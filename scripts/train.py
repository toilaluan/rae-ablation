from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from jit_reg.checkpoint import EMA, CheckpointManager
from jit_reg.config import load_config
from jit_reg.data import data_fingerprint, load_imagenet, preprocessing_metadata
from jit_reg.distributed import initialize_distributed
from jit_reg.flops import model_flops_per_step, total_training_flops_per_step
from jit_reg.model import DDT
from jit_reg.model.compile import compile_model_regions
from jit_reg.optim import build_optimizer
from jit_reg.pixel_codec import load_pixel_codec_artifact
from jit_reg.rng import RNGStreams
from jit_reg.sample_runtime import SamplingParameters
from jit_reg.token_cache import TokenCache
from jit_reg.training import TrainState, make_train_loader, train
from jit_reg.validation import ValidationPlan, create_validation_plan
from jit_reg.validation_runtime import validate_ema
from jit_reg.wandb_utils import RunLogger

torch._dynamo.config.cache_size_limit = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DDT on mixed pixel and RAE tokens")
    parser.add_argument("--config", type=Path, default=Path("configs/ddt-token-mix.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def _sampling_parameters(config: object) -> SamplingParameters:
    s = config.sampling  # type: ignore[attr-defined]
    return SamplingParameters(
        batch_size=s.batch_size,
        sampling_steps=s.sampling_steps,
        solver=s.solver,
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
    sampling_parameters = _sampling_parameters(config)
    sampling_parameters.validate()
    if not Path(config.evaluation.reference_npz).is_file():
        raise FileNotFoundError(config.evaluation.reference_npz)

    rank, world_size, device = initialize_distributed()
    streams = RNGStreams.create(config.train.seed, device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(streams.initialization.initial_seed())
        model = DDT(config.model).to(device)

    dataset = load_imagenet(config.data, 256, load_pixels=True)
    token_cache = TokenCache(config, len(dataset))
    codec_path = (
        args.resume.resolve() / "pixel-codec"
        if args.resume is not None
        else Path(config.pixel_codec.checkpoint)
    )
    pixel_codec = load_pixel_codec_artifact(
        codec_path, device, expected_config=config.pixel_codec
    )
    fingerprint = data_fingerprint(config.data, 256)
    if pixel_codec.metadata.get("token_cache_fingerprint") != token_cache.fingerprint:
        raise ValueError("pixel codec was fitted against a different RAE token cache")
    if pixel_codec.metadata.get("data_fingerprint") != fingerprint:
        raise ValueError("pixel codec was fitted against different image preprocessing")
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
            token_cache.fingerprint,
            pixel_codec.weights_sha256,
        )
        optimizer.load_state_dict(resumed.optimizer)
        streams.load_state_dict(resumed.rng_streams)
        state = TrainState(resumed.epoch, resumed.global_step, resumed.last_completed_validation_epoch)
        resume_run_id = resumed.wandb_run_id

    execution_model = compile_model_regions(model, config.compile)
    if world_size > 1:
        execution_model = torch.nn.parallel.DistributedDataParallel(
            execution_model, device_ids=[device.index] if device.type == "cuda" else None
        )
    loader, sampler = make_train_loader(dataset, config, streams, token_cache)
    logged_config = config.as_dict()
    logged_config["provenance"] = {
        "dataset_and_preprocessing": preprocessing_metadata(config.data, 256),
        "data_fingerprint": fingerprint,
        "validation_plan_hash": plan.plan_hash,
        "token_cache_fingerprint": token_cache.fingerprint,
        "pixel_codec_sha256": pixel_codec.weights_sha256,
        "rae": {
            "base_repository": config.rae.base_repository,
            "base_revision": config.rae.base_revision,
            "compressor_repository": config.rae.compressor_repository,
            "compressor_revision": config.rae.compressor_revision,
            "remote_code_schema": config.rae.remote_code_schema,
        },
        "diffusion_input_shape": list(config.model.input_shape),
        "mixing": config.mixing.__dict__,
    }
    logged_config["model_parameters"] = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    logger = RunLogger(config.wandb, logged_config, rank == 0, resume_run_id, args.log_interval)
    logger.event(
        f"training initialized: variant={config.model.model_variant}, input_shape={config.model.input_shape}, "
        f"ratio={'fixed:' + str(config.mixing.fixed_ratio) if config.mixing.fixed_ratio is not None else 'logit-normal+endpoints'}, "
        f"pixel_codec={pixel_codec.weights_sha256[:12]}, "
        f"parameters={logged_config['model_parameters']['total']:,}"
    )
    output_root = Path(config.checkpoint.directory).parent / "validation"

    def validate_callback(epoch: int, final: bool) -> dict[str, float]:
        metrics, _output = validate_ema(
            ema=ema,
            config=config,
            plan=plan,
            completed_epoch=epoch,
            final=final,
            output_root=output_root,
            sampling=sampling_parameters,
            token_cache=token_cache,
            pixel_codec=pixel_codec,
        )
        per_step = model_flops_per_step(config)
        metrics["compute/cumulative_model_flops"] = float(state.global_step * per_step)
        metrics["compute/cumulative_total_flops"] = float(
            state.global_step * total_training_flops_per_step(config)
        )
        return metrics

    try:
        if config.validation.validate_now:
            started = time.perf_counter()
            logger.log(validate_callback(state.epoch, state.epoch >= config.train.epochs), step=state.global_step)
            logger.event(f"forced validation finished in {time.perf_counter() - started:.1f}s")
        train(
            canonical_model=model,
            execution_model=execution_model,
            loader=loader,
            sampler=sampler,
            optimizer=optimizer,
            ema=ema,
            streams=streams,
            config=config,
            checkpoint_manager=manager,
            data_fingerprint=fingerprint,
            validation_plan_hash=plan.plan_hash,
            validation_plan_path=plan_path,
            logger=logger,
            token_cache=token_cache,
            pixel_codec=pixel_codec,
            state=state,
            validation_callback=validate_callback,
        )
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
