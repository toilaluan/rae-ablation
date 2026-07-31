from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from self_flow.checkpoint import EMA, CheckpointManager
from self_flow.config import load_config
from self_flow.data import data_fingerprint, load_imagenet, preprocessing_metadata
from self_flow.distributed import initialize_distributed
from self_flow.latent_cache import LatentCache
from self_flow.model import SiT
from self_flow.model.compile import compile_model_regions
from self_flow.optim import build_optimizer
from self_flow.rng import RNGStreams
from self_flow.sample_runtime import SamplingParameters
from self_flow.training import TrainState, make_train_loader, train
from self_flow.vae import load_flux2_codec
from self_flow.validation import ValidationPlan, create_validation_plan
from self_flow.validation_runtime import validate_model
from self_flow.wandb_utils import RunLogger

torch._dynamo.config.cache_size_limit = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Self-Flow SiT-B/2 on FLUX.2 latents")
    parser.add_argument("--config", type=Path, default=Path("configs/vanilla.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.override)
    if args.output:
        config.checkpoint.directory = str(args.output / "checkpoints")
        config.validation.plan_path = str(args.output / "validation-plan.pt")
    config.validate()
    if not Path(config.evaluation.reference_npz).is_file():
        raise FileNotFoundError(config.evaluation.reference_npz)

    rank, world_size, device = initialize_distributed()
    streams = RNGStreams.create(config.train.seed, device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(streams.initialization.initial_seed())
        model = SiT(
            config.model,
            align_layers=config.objective.align_layers,
            projector_hidden_size=config.objective.projector_hidden_size,
        ).to(device)
    ema = EMA(model, config.objective.ema_decay) if config.objective.kind == "ema" else None
    latent_cache_active = config.vae.cache.enabled
    dataset = load_imagenet(
        config.data,
        config.model.image_size,
        load_pixels=not latent_cache_active,
    )
    latent_cache = LatentCache(config, len(dataset)) if latent_cache_active else None
    codec = (
        load_flux2_codec(config.vae, image_size=config.model.image_size, device=device)
        if latent_cache is None
        else None
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

    optimizer = build_optimizer(model, config.train.optimizer)
    manager = CheckpointManager(config.checkpoint.directory, config.checkpoint.keep_last_n)
    state = TrainState()
    resume_run_id: str | None = None
    if args.resume:
        resumed = manager.load(
            args.resume.resolve(),
            model,
            ema,
            expected_data_fingerprint=fingerprint,
            expected_validation_plan_hash=plan.plan_hash,
            expected_latent_cache_fingerprint=latent_cache.fingerprint if latent_cache else None,
        )
        optimizer.load_state_dict(resumed.optimizer)
        streams.load_state_dict(resumed.rng_streams)
        state = TrainState(
            resumed.epoch, resumed.global_step, resumed.last_completed_validation_epoch
        )
        resume_run_id = resumed.wandb_run_id

    execution_model = compile_model_regions(model, config.compile)
    if world_size > 1:
        execution_model = torch.nn.parallel.DistributedDataParallel(
            execution_model,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
    loader, sampler = make_train_loader(dataset, config, streams)
    logged_config = config.as_dict()
    logged_config["provenance"] = {
        "dataset_and_preprocessing": preprocessing_metadata(config.data, config.model.image_size),
        "data_fingerprint": fingerprint,
        "validation_plan_hash": plan.plan_hash,
        "latent_cache_fingerprint": latent_cache.fingerprint if latent_cache else None,
        "diffusion_input_shape": list(config.model.input_shape),
        "vae": {
            "model_name": config.vae.model_name,
            "revision": config.vae.revision,
            "subfolder": config.vae.subfolder,
        },
    }
    logged_config["model_parameters"] = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    logger = RunLogger(config.wandb, logged_config, rank == 0, resume_run_id, args.log_interval)
    logger.event(
        f"training initialized: {config.model.model_variant}, objective={config.objective.kind}, "
        f"parameters={logged_config['model_parameters']['total']:,}"
    )
    sampling = SamplingParameters(
        batch_size=config.sampling.batch_size,
        sampling_steps=config.sampling.sampling_steps,
        solver=config.sampling.solver,
        guidance_scale=config.sampling.guidance_scale,
        timestep_shift=config.sampling.timestep_shift,
        cfg_start=config.sampling.cfg_start,
        cfg_end=config.sampling.cfg_end,
        log_interval=config.sampling.log_interval,
    )
    output_root = Path(config.checkpoint.directory).parent / "validation"

    def validate_callback(epoch: int, final: bool) -> dict[str, float]:
        metrics, validation_output = validate_model(
            model=model,
            ema=ema,
            config=config,
            plan=plan,
            completed_epoch=epoch,
            final=final,
            output_root=output_root,
            sampling=sampling,
        )
        if rank == 0:
            labels = ", ".join(str(value) for value in plan.labels[:16].tolist())
            logger.log_grid(
                str(validation_output / "preview.png"),
                f"ImageNet class IDs: {labels}",
                state.global_step,
            )
        return metrics

    try:
        if config.validation.validate_now:
            started = time.perf_counter()
            forced = validate_callback(state.epoch, state.epoch >= config.train.epochs)
            forced["throughput/validation_seconds"] = time.perf_counter() - started
            state.last_completed_validation_epoch = state.epoch
            config.validation.validate_now = False
            logger.log(forced, step=state.global_step)
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
            vae_codec=codec,
            latent_cache=latent_cache,
            state=state,
            validation_callback=validate_callback,
        )
        logger.event(f"training complete: epoch={state.epoch}, step={state.global_step}")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
