from __future__ import annotations

import argparse
from pathlib import Path

import torch

from jit_reg.checkpoint import EMA, CheckpointManager
from jit_reg.config import ExperimentConfig, load_config
from jit_reg.data import data_fingerprint, load_imagenet, preprocessing_metadata
from jit_reg.distributed import initialize_distributed
from jit_reg.endpoint_cache import EndpointCache
from jit_reg.latent_statistics import (
    EndpointLatentNormalizer,
    latent_statistics_sha256,
    load_latent_statistics,
)
from jit_reg.model import SemanticDiT
from jit_reg.model.compile import compile_model_regions
from jit_reg.optim import build_optimizer
from jit_reg.pixel_decoder import (
    decoder_latent_statistics_sha256,
    decoder_weights_path,
)
from jit_reg.rng import RNGStreams
from jit_reg.sample_runtime import checkpoint_sha256
from jit_reg.training import TrainState, make_train_loader, train
from jit_reg.validation import ValidationPlan, create_validation_plan
from jit_reg.validation_runtime import validate_ema
from jit_reg.vision import load_dino_encoder, vision_fingerprint
from jit_reg.wandb_utils import RunLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train matched semantic-to-target or semantic-to-RGB flow"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dino-cascade.yaml"))
    parser.add_argument("--endpoint", choices=("target", "pixel"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def _configure(args: argparse.Namespace) -> ExperimentConfig:
    overrides = list(args.override)
    if args.endpoint is not None:
        overrides.append(f"flow.endpoint={args.endpoint}")
    config = load_config(args.config, overrides)
    if args.output:
        config.checkpoint.directory = str(args.output / "checkpoints")
        config.validation.plan_path = str(args.output / "validation-plan.pt")
    config.validate()
    return config


def main() -> None:
    args = build_parser().parse_args()
    config = _configure(args)
    if config.flow.endpoint == "target" and config.decoder.checkpoint is None:
        raise ValueError("target flow requires decoder.checkpoint for reconstruction validation")
    if config.decoder.checkpoint is not None:
        config.decoder.checkpoint = str(Path(config.decoder.checkpoint).resolve())
    rank, world_size, device = initialize_distributed()
    statistics_path = Path(config.vision.statistics_path).resolve()
    config.vision.statistics_path = str(statistics_path)
    statistics = load_latent_statistics(
        statistics_path,
        expected_channels=config.vision.hidden_size,
        expected_vision_fingerprint=vision_fingerprint(config.vision),
        expected_data_fingerprint=data_fingerprint(config.data, config.vision.image_size),
    )
    statistics_digest = latent_statistics_sha256(statistics_path)
    if (
        config.flow.endpoint == "target"
        and config.decoder.checkpoint is not None
        and decoder_latent_statistics_sha256(config.decoder.checkpoint)
        != statistics_digest
    ):
        raise ValueError("pixel decoder was trained with different latent statistics")
    normalizer = EndpointLatentNormalizer(statistics).to(device)
    decoder_digest = (
        checkpoint_sha256(decoder_weights_path(config.decoder.checkpoint))
        if config.decoder.checkpoint is not None
        else "not-used-by-pixel-flow"
    )
    streams = RNGStreams.create(config.flow.train.seed, device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(streams.initialization.initial_seed())
        model = SemanticDiT(config.model).to(device)
    load_pixels = not config.vision.cache.enabled or config.flow.endpoint == "pixel"
    dataset = load_imagenet(
        config.data, config.vision.image_size, load_pixels=load_pixels
    )
    endpoint_cache = (
        EndpointCache(config, len(dataset), statistics_digest)
        if config.vision.cache.enabled
        else None
    )
    endpoint_source = endpoint_cache.fingerprint if endpoint_cache is not None else "online"
    feature_encoder = (
        None
        if endpoint_cache is not None
        else load_dino_encoder(
            config.vision,
            device,
            include_semantic=True,
            normalizer=normalizer,
        )
    )
    plan_path = Path(config.validation.plan_path)
    if rank == 0 and not plan_path.exists():
        create_validation_plan(
            config.validation.final_samples,
            config.validation.periodic_samples,
            config.flow.train.seed + 7,
            min(
                config.data.validation_size,
                config.data.max_samples or config.data.validation_size,
            ),
        ).save(plan_path)
    if world_size > 1:
        torch.distributed.barrier()
    plan = ValidationPlan.load(plan_path)
    ema = EMA(model, config.flow.train.ema_decay)
    optimizer = build_optimizer(model, config.flow.train.optimizer)
    manager = CheckpointManager(
        config.checkpoint.directory, config.checkpoint.keep_last_n
    )
    state = TrainState()
    resume_run_id: str | None = None
    fingerprint = data_fingerprint(config.data, config.vision.image_size)
    if args.resume:
        resumed = manager.load(
            args.resume.resolve(),
            model,
            ema,
            fingerprint,
            plan.plan_hash,
            vision_fingerprint(config.vision),
            decoder_digest,
            statistics_digest,
            endpoint_source,
            config.flow.endpoint,
        )
        optimizer.load_state_dict(resumed.optimizer)
        streams.load_state_dict(resumed.rng_streams)
        state = TrainState(
            resumed.epoch,
            resumed.global_step,
            resumed.last_completed_validation_epoch,
        )
        resume_run_id = resumed.wandb_run_id
    execution_model = compile_model_regions(model, config.compile)
    if world_size > 1:
        execution_model = torch.nn.parallel.DistributedDataParallel(
            execution_model,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
    loader, sampler = make_train_loader(
        dataset, config, streams, endpoint_cache
    )
    logged_config = config.as_dict()
    logged_config["stage"] = f"semantic_to_{config.flow.endpoint}_flow"
    logged_config["provenance"] = {
        "dataset": preprocessing_metadata(config.data, config.vision.image_size),
        "data_fingerprint": fingerprint,
        "validation_plan_hash": plan.plan_hash,
        "vision_fingerprint": vision_fingerprint(config.vision),
        "latent_statistics_sha256": statistics_digest,
        "decoder_sha256": decoder_digest,
        "endpoint_source": endpoint_source,
    }
    logger = RunLogger(
        config.wandb, logged_config, rank == 0, resume_run_id, args.log_interval
    )
    output_root = Path(config.checkpoint.directory).parent / "validation"

    def validate_callback(epoch: int, final: bool) -> dict[str, float]:
        metrics, output = validate_ema(
            ema=ema,
            config=config,
            plan=plan,
            completed_epoch=epoch,
            final=final,
            output_root=output_root,
            normalizer=normalizer,
        )
        preview = output / "preview.png"
        if preview.is_file():
            logger.log_grid(str(preview), f"EMA epoch {epoch}", state.global_step)
        return metrics

    try:
        train(
            canonical_model=model,
            execution_model=execution_model,
            feature_encoder=feature_encoder,
            normalizer=normalizer,
            loader=loader,
            sampler=sampler,
            optimizer=optimizer,
            ema=ema,
            streams=streams,
            config=config,
            checkpoint_manager=manager,
            data_fingerprint=fingerprint,
            decoder_sha256=decoder_digest,
            latent_statistics_sha256=statistics_digest,
            endpoint_cache_fingerprint=endpoint_source,
            validation_plan_hash=plan.plan_hash,
            validation_plan_path=plan_path,
            logger=logger,
            state=state,
            validation_callback=validate_callback,
        )
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
