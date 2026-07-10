from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rae_ablation.cache import ShardedLatentDataset, read_manifest
from rae_ablation.checkpoint import EMA, CheckpointManager
from rae_ablation.config import load_config
from rae_ablation.constants import (
    COMPACT_REPO,
    COMPACT_REVISION,
    IMAGENET_REPO,
    IMAGENET_REVISION,
    RAE_REPO,
    RAE_REVISION,
)
from rae_ablation.distributed import initialize_distributed
from rae_ablation.latents import LatentStatistics
from rae_ablation.model.compile import compile_model_regions
from rae_ablation.model.dit import DiffusionTransformer
from rae_ablation.optim import build_optimizer
from rae_ablation.rng import RNGStreams
from rae_ablation.sample_runtime import SamplingParameters
from rae_ablation.training import TrainState, make_train_loader, train
from rae_ablation.validation import ValidationPlan, create_validation_plan
from rae_ablation.validation_runtime import validate_ema
from rae_ablation.wandb_utils import RunLogger

torch._dynamo.config.cache_size_limit = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the matched latent diffusion model")
    parser.add_argument("--config", type=Path, default=Path("configs/canonical.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--codec", choices=("rae", "rae_compact"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--validate-now", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-sampling-steps", type=int, default=50)
    parser.add_argument("--eval-guidance-scale", type=float, default=1.78)
    parser.add_argument("--eval-guidance-start", type=float, default=0.1)
    parser.add_argument("--eval-log-interval", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.override)
    if args.codec:
        config.codec = args.codec
    if args.cache_dir:
        config.cache.directory = str(args.cache_dir)
    if args.output:
        config.checkpoint.directory = str(args.output / "checkpoints")
        config.validation.plan_path = str(args.output / "validation-plan.pt")
    if args.validation_interval is not None:
        config.validation.interval = args.validation_interval
    if args.checkpoint_interval is not None:
        config.checkpoint.interval = args.checkpoint_interval
    if args.max_steps is not None:
        config.train.max_steps = args.max_steps
    config.validation.validate_now = args.validate_now
    config.wandb.enabled = config.wandb.enabled or args.wandb
    config.validate()
    sampling = SamplingParameters(
        batch_size=args.eval_batch_size,
        sampling_steps=args.eval_sampling_steps,
        guidance_scale=args.eval_guidance_scale,
        guidance_start=args.eval_guidance_start,
        log_interval=args.eval_log_interval,
    )
    sampling.validate()
    if not args.reference_npz.is_file():
        raise FileNotFoundError(args.reference_npz)

    rank, world_size, device = initialize_distributed()
    if rank == 0:
        print(f"initializing training on {world_size} rank(s), device={device}", flush=True)
    streams = RNGStreams.create(config.train.seed, device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(streams.initialization.initial_seed())
        model = DiffusionTransformer(config.model).to(device)
    dataset = ShardedLatentDataset(config.cache.directory, config.codec)
    manifest = read_manifest(Path(config.cache.directory) / "manifest.json")
    statistics = LatentStatistics.load(
        Path(config.cache.directory) / f"stats-{config.codec}.safetensors"
    )
    plan_path = Path(config.validation.plan_path)
    if rank == 0 and not plan_path.exists():
        create_validation_plan(
            config.validation.final_samples,
            config.validation.periodic_samples,
            config.model.num_classes,
            config.train.seed + 7,
            channels=config.model.in_channels,
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
            args.resume.resolve(), model, ema, manifest.fingerprint, plan.plan_hash
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
    logged_config["online_validation"] = {
        "reference_npz": str(args.reference_npz),
        "batch_size": sampling.batch_size,
        "sampling_steps": sampling.sampling_steps,
        "guidance_scale": sampling.guidance_scale,
        "guidance_start": sampling.guidance_start,
        "log_interval": sampling.log_interval,
    }
    logged_config["provenance"] = {
        "rae": {"repository": RAE_REPO, "revision": RAE_REVISION},
        "compact": {"repository": COMPACT_REPO, "revision": COMPACT_REVISION},
        "dataset": {"repository": IMAGENET_REPO, "revision": IMAGENET_REVISION},
        "cache_fingerprint": manifest.fingerprint,
        "validation_plan_hash": plan.plan_hash,
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
        f"training initialized: codec={config.codec}, parameters="
        f"{logged_config['model_parameters']['total']:,}"
    )
    if args.resume:
        logger.event(
            f"resumed from {args.resume}: epoch={state.epoch}, step={state.global_step}"
        )
    output_root = Path(config.checkpoint.directory).parent / "validation"

    def validate_callback(epoch: int, final: bool) -> dict[str, float]:
        metrics, validation_output = validate_ema(
            ema=ema,
            config=config,
            statistics=statistics,
            plan=plan,
            completed_epoch=epoch,
            final=final,
            output_root=output_root,
            reference_npz=args.reference_npz,
            sampling=sampling,
            eval_batch_size=args.eval_batch_size,
            eval_log_interval=args.eval_log_interval,
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
        if config.validation.validate_now and state.epoch > 0:
            logger.event(f"forced validation started after epoch {state.epoch}")
            forced_metrics = validate_callback(state.epoch, state.epoch >= config.train.epochs)
            state.last_completed_validation_epoch = state.epoch
            config.validation.validate_now = False
            logger.log(forced_metrics, step=state.global_step)
            for name, value in forced_metrics.items():
                if "FID" in name:
                    logger.event(f"{name}: {value:.6f}")

        train(
            canonical_model=model,
            execution_model=execution_model,
            loader=loader,
            sampler=sampler,
            statistics=statistics,
            optimizer=optimizer,
            ema=ema,
            streams=streams,
            config=config,
            checkpoint_manager=manager,
            cache_fingerprint=manifest.fingerprint,
            validation_plan_hash=plan.plan_hash,
            logger=logger,
            state=state,
            validation_callback=validate_callback,
        )
        logger.event(f"training complete: epoch={state.epoch}, step={state.global_step}")
    finally:
        logger.event("training shutdown")
        logger.finish()


if __name__ == "__main__":
    main()
