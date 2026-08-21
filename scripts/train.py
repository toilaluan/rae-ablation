from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from jit_reg.checkpoint import EMA, CheckpointManager
from jit_reg.config import ExperimentConfig, load_config
from jit_reg.data import data_fingerprint, load_imagenet, preprocessing_metadata
from jit_reg.distributed import initialize_distributed
from jit_reg.endpoint_cache import EndpointCache
from jit_reg.flops import model_flops_per_step
from jit_reg.latent_statistics import (
    EndpointLatentNormalizer,
    latent_statistics_sha256,
    load_latent_statistics,
)
from jit_reg.model import DDT
from jit_reg.model.compile import compile_model_regions
from jit_reg.optim import build_optimizer
from jit_reg.pixel_decoder import decoder_latent_statistics_sha256, decoder_weights_path
from jit_reg.rng import RNGStreams
from jit_reg.sample_runtime import SamplingParameters, checkpoint_sha256
from jit_reg.training import TrainState, make_train_loader, train
from jit_reg.validation import ValidationPlan, create_validation_plan
from jit_reg.validation_runtime import validate_ema
from jit_reg.vision import load_dino_encoder, vision_fingerprint
from jit_reg.wandb_utils import RunLogger

torch._dynamo.config.cache_size_limit = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train semantic-waypoint DDT on online or cached DINOv3 endpoints"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dino-waypoint.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def _sampling_parameters(config: ExperimentConfig) -> SamplingParameters:
    s = config.sampling
    return SamplingParameters(
        batch_size=s.batch_size,
        semantic_steps=s.semantic_steps,
        target_steps=s.target_steps,
        waypoint_time=config.waypoint.time,
        solver=s.solver,
        timestep_shift=s.timestep_shift,
        ig_scale=s.ig_scale,
        ig_start=s.ig_start,
        ig_end=s.ig_end,
        cfg_scale=s.cfg_scale,
        cfg_start=s.cfg_start,
        cfg_end=s.cfg_end,
        log_interval=s.log_interval,
    )


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.override)
    if args.output:
        config.checkpoint.directory = str(args.output / "checkpoints")
        config.validation.plan_path = str(args.output / "validation-plan.pt")
    config.validate()
    if config.decoder.checkpoint is None:
        raise ValueError("decoder.checkpoint must point to a trained pixel decoder")
    config.decoder.checkpoint = str(Path(config.decoder.checkpoint).resolve())
    decoder_digest = checkpoint_sha256(decoder_weights_path(config.decoder.checkpoint))
    sampling_parameters = _sampling_parameters(config)
    sampling_parameters.validate()
    if not Path(config.evaluation.reference_npz).is_file():
        raise FileNotFoundError(config.evaluation.reference_npz)

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
    if decoder_latent_statistics_sha256(config.decoder.checkpoint) != statistics_digest:
        raise ValueError("decoder checkpoint was trained with different latent statistics")
    streams = RNGStreams.create(config.train.seed, device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(streams.initialization.initial_seed())
        model = DDT(config.model, config.repa).to(device)
    dataset = load_imagenet(
        config.data,
        config.vision.image_size,
        load_pixels=not config.vision.cache.enabled,
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
            normalizer=EndpointLatentNormalizer(statistics),
        )
    )
    fingerprint = data_fingerprint(config.data, config.vision.image_size)
    feature_id = vision_fingerprint(config.vision)
    plan_path = Path(config.validation.plan_path)
    if rank == 0 and not plan_path.exists():
        create_validation_plan(
            config.validation.final_samples,
            config.validation.periodic_samples,
            config.model.num_classes,
            config.train.seed + 7,
        ).save(plan_path)
    if world_size > 1:
        torch.distributed.barrier()
    plan = ValidationPlan.load(plan_path)

    ema = EMA(model, config.train.ema_decay)
    optimizer = build_optimizer(model, config.train.optimizer)
    manager = CheckpointManager(config.checkpoint.directory, config.checkpoint.keep_last_n)
    state = TrainState()
    resume_run_id: str | None = None
    if args.resume:
        resumed = manager.load(
            args.resume.resolve(),
            model,
            ema,
            fingerprint,
            plan.plan_hash,
            feature_id,
            decoder_digest,
            statistics_digest,
            endpoint_source,
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
    loader, sampler = make_train_loader(dataset, config, streams, endpoint_cache)
    logged_config = config.as_dict()
    logged_config["provenance"] = {
        "dataset_and_preprocessing": preprocessing_metadata(
            config.data, config.vision.image_size
        ),
        "data_fingerprint": fingerprint,
        "validation_plan_hash": plan.plan_hash,
        "vision_fingerprint": feature_id,
        "latent_statistics_sha256": statistics_digest,
        "decoder_sha256": decoder_digest,
        "diffusion_input_shape": list(config.model.input_shape),
        "endpoint_source": endpoint_source,
    }
    logged_config["model_parameters"] = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    logger = RunLogger(
        config.wandb, logged_config, rank == 0, resume_run_id, args.log_interval
    )
    logger.event(
        f"training initialized: variant={config.model.model_variant}, "
        f"waypoint={config.waypoint.time}, endpoint_source={endpoint_source}, "
        f"input_shape={config.model.input_shape}, "
        f"parameters={logged_config['model_parameters']['total']:,}"
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
            sampling=sampling_parameters,
            endpoint_source=endpoint_source,
        )
        metrics["compute/cumulative_model_flops"] = float(
            state.global_step * model_flops_per_step(config)
        )
        preview = output / "preview.png"
        if preview.is_file():
            logger.log_grid(str(preview), f"EMA epoch {epoch}", state.global_step)
        return metrics

    try:
        if config.validation.validate_now:
            started = time.perf_counter()
            logger.log(
                validate_callback(state.epoch, state.epoch >= config.train.epochs),
                step=state.global_step,
            )
            logger.event(
                f"forced validation finished in {time.perf_counter() - started:.1f}s"
            )
        train(
            canonical_model=model,
            execution_model=execution_model,
            feature_encoder=feature_encoder,
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
