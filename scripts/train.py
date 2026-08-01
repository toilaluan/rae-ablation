from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from jit_reg.checkpoint import EMA, CheckpointManager
from jit_reg.config import load_config
from jit_reg.data import (
    data_fingerprint,
    load_imagenet,
    preprocessing_metadata,
)
from jit_reg.distributed import initialize_distributed
from jit_reg.feature_cache import FeatureCache
from jit_reg.flops import model_flops_per_step
from jit_reg.latent_cache import LatentCache
from jit_reg.model.compile import compile_model_regions
from jit_reg.model.sit import SiT
from jit_reg.optim import build_optimizer
from jit_reg.repa import DINOv3FeatureExtractor
from jit_reg.rng import RNGStreams
from jit_reg.sample_runtime import SamplingParameters
from jit_reg.training import TrainState, make_train_loader, train
from jit_reg.vae import load_flux2_codec
from jit_reg.validation import ValidationPlan, create_validation_plan
from jit_reg.validation_runtime import validate_ema
from jit_reg.wandb_utils import RunLogger

torch._dynamo.config.cache_size_limit = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train pixel- or FLUX.2-latent SiT on ImageNet-256"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sit-baseline.yaml"),
        help="schema-v11 experiment YAML",
    )
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    if args.output:
        config.checkpoint.directory = str(args.output / "checkpoints")
        config.validation.plan_path = str(args.output / "validation-plan.pt")
    config.validate()
    sampling_config = config.sampling
    sampling_parameters = SamplingParameters(
        batch_size=sampling_config.batch_size,
        sampling_steps=sampling_config.sampling_steps,
        solver=sampling_config.solver,
        guidance_scale=sampling_config.guidance_scale,
        timestep_shift=sampling_config.timestep_shift,
        cfg_start=sampling_config.cfg_start,
        cfg_end=sampling_config.cfg_end,
        log_interval=sampling_config.log_interval,
    )
    sampling_parameters.validate()
    if not Path(config.evaluation.reference_npz).is_file():
        raise FileNotFoundError(config.evaluation.reference_npz)

    rank, world_size, device = initialize_distributed()
    if rank == 0:
        print(f"initializing training on {world_size} rank(s), device={device}", flush=True)
    streams = RNGStreams.create(config.train.seed, device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(streams.initialization.initial_seed())
        model = SiT(config.model, config.repa).to(device)
    cache_active = config.repa.enabled and config.repa.cache.enabled
    latent_cache_active = (
        config.model.input_space == "flux2_latents"
        and config.vae.cache.enabled
    )
    pixels_required = (
        config.model.input_space == "pixels"
        or (
            config.model.input_space == "flux2_latents"
            and not latent_cache_active
        )
        or (config.repa.enabled and not cache_active)
    )
    dataset = load_imagenet(
        config.data,
        config.model.image_size,
        load_pixels=pixels_required,
    )
    feature_cache = FeatureCache(config, len(dataset)) if cache_active else None
    latent_cache = (
        LatentCache(config, len(dataset)) if latent_cache_active else None
    )
    fingerprint = data_fingerprint(config.data, config.model.image_size)
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
            feature_cache.fingerprints if feature_cache is not None else None,
            latent_cache.fingerprint if latent_cache is not None else None,
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
    feature_encoder = None
    if config.repa.enabled and feature_cache is None:
        feature_encoder = (
            DINOv3FeatureExtractor(config.repa).to(device).bfloat16()
        )
        if config.compile.enabled:
            feature_encoder = torch.compile(
                feature_encoder,
                backend=config.compile.backend,
                mode=config.compile.mode,
                fullgraph=config.compile.fullgraph,
                dynamic=config.compile.dynamic,
            )
    vae_codec = (
        load_flux2_codec(
            config.vae,
            image_size=config.model.image_size,
            device=device,
        )
        if config.model.input_space == "flux2_latents"
        and latent_cache is None
        else None
    )
    loader, sampler = make_train_loader(dataset, config, streams)
    logged_config = config.as_dict()
    logged_config["provenance"] = {
        "dataset_and_preprocessing": preprocessing_metadata(config.data, config.model.image_size),
        "data_fingerprint": fingerprint,
        "validation_plan_hash": plan.plan_hash,
        "feature_target": (
            config.repa.encoder.model_name
            if config.repa.enabled
            else None
        ),
        "feature_cache_fingerprints": (
            feature_cache.fingerprints if feature_cache is not None else None
        ),
        "latent_cache_fingerprint": (
            latent_cache.fingerprint if latent_cache is not None else None
        ),
        "input_space": config.model.input_space,
        "diffusion_input_shape": list(config.model.input_shape),
        "vae": (
            {
                "model_name": config.vae.model_name,
                "revision": config.vae.revision,
                "subfolder": config.vae.subfolder,
            }
            if config.model.input_space == "flux2_latents"
            else None
        ),
    }
    logged_config["model_parameters"] = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    logger = RunLogger(
        config.wandb,
        logged_config,
        rank == 0,
        resume_run_id,
        args.log_interval,
    )
    logger.event(
        f"training initialized: variant={config.model.model_variant}, "
        f"input_shape={config.model.input_shape}, "
        f"repa={config.repa.enabled}, selector={config.exploration.selector}, "
        f"K={config.exploration.k}, "
        "parameters="
        f"{logged_config['model_parameters']['total']:,}"
    )
    if args.resume:
        logger.event(f"resumed from {args.resume}: epoch={state.epoch}, step={state.global_step}")
    output_root = Path(config.checkpoint.directory).parent / "validation"

    def validate_callback(epoch: int, final: bool) -> dict[str, float]:
        metrics, validation_output = validate_ema(
            ema=ema,
            config=config,
            plan=plan,
            completed_epoch=epoch,
            final=final,
            output_root=output_root,
            sampling=sampling_parameters,
        )
        metrics.update(
            {
                "progress/epoch": float(epoch),
                "progress/images_seen": float(
                    state.global_step * config.train.global_batch_size
                ),
                "compute/cumulative_model_flops": float(
                    state.global_step * model_flops_per_step(config)
                ),
            }
        )
        if rank == 0:
            labels = ", ".join(str(value) for value in plan.labels[:16].tolist())
            logger.log_grid(
                str(validation_output / "preview.png"),
                f"First 16 evaluated ImageNet class IDs: {labels}",
                state.global_step,
            )
        return metrics

    try:
        if config.validation.validate_now:
            logger.event(f"forced validation started after epoch {state.epoch}")
            forced_validation_started = time.perf_counter()
            forced_metrics = validate_callback(state.epoch, state.epoch >= config.train.epochs)
            forced_validation_seconds = time.perf_counter() - forced_validation_started
            forced_metrics["throughput/validation_seconds"] = forced_validation_seconds
            state.last_completed_validation_epoch = state.epoch
            config.validation.validate_now = False
            logger.log(forced_metrics, step=state.global_step)
            logger.event(f"forced validation finished in {forced_validation_seconds:.1f}s")
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
            feature_encoder=feature_encoder,
            feature_cache=feature_cache,
            vae_codec=vae_codec,
            latent_cache=latent_cache,
            state=state,
            validation_callback=validate_callback,
        )
        logger.event(f"training complete: epoch={state.epoch}, step={state.global_step}")
    finally:
        logger.event("training shutdown")
        logger.finish()


if __name__ == "__main__":
    main()
