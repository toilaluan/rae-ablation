from __future__ import annotations

import argparse
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
from jit_reg.model.compile import compile_model_regions
from jit_reg.model.sit import SiT
from jit_reg.optim import build_optimizer
from jit_reg.reg import DINOv3FeatureExtractor
from jit_reg.rng import RNGStreams
from jit_reg.sample_runtime import SamplingParameters
from jit_reg.training import TrainState, make_train_loader, train
from jit_reg.validation import ValidationPlan, create_validation_plan
from jit_reg.validation_runtime import validate_ema
from jit_reg.wandb_utils import RunLogger

torch._dynamo.config.cache_size_limit = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train pixel-space JiT on ImageNet-256")
    parser.add_argument("--config", type=Path, default=Path("configs/canonical.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    reg = parser.add_mutually_exclusive_group()
    reg.add_argument("--reg", action="store_true", help="enable standard DINOv3 REG and REPA")
    reg.add_argument(
        "--reg-widehead",
        action="store_true",
        help="enable REG with the wide intermediate clean-token predictor",
    )
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
    parser.add_argument("--eval-timestep-shift", type=float, default=1.0)
    parser.add_argument("--eval-reg-timestep-shift", type=float)
    parser.add_argument("--eval-cfg-start", type=float, default=0.0)
    parser.add_argument("--eval-cfg-end", type=float, default=1.0)
    parser.add_argument("--eval-log-interval", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.override)
    if args.reg:
        config.model.reg.enabled = True
        config.model.reg.cls_predictor = "standard"
    elif args.reg_widehead:
        config.model.reg.enabled = True
        config.model.reg.cls_predictor = "wide_intermediate"
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
        timestep_shift=args.eval_timestep_shift,
        reg_timestep_shift=args.eval_reg_timestep_shift,
        cfg_start=args.eval_cfg_start,
        cfg_end=args.eval_cfg_end,
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
        model = SiT(config.model).to(device)
    dataset = load_imagenet(config.data, config.model.image_size)
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
        resumed = manager.load(args.resume.resolve(), model, ema, fingerprint, plan.plan_hash)
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
    reg_encoder = (
        torch.compile(DINOv3FeatureExtractor(config.model.reg)).to(device).bfloat16() if config.model.reg.enabled else None
    )
    loader, sampler = make_train_loader(dataset, config, streams)
    logged_config = config.as_dict()
    logged_config["online_validation"] = {
        "reference_npz": str(args.reference_npz),
        "batch_size": sampling.batch_size,
        "sampling_steps": sampling.sampling_steps,
        "guidance_scale": sampling.guidance_scale,
        "timestep_shift": sampling.timestep_shift,
        "reg_timestep_shift": sampling.reg_timestep_shift,
        "cfg_interval": [sampling.cfg_start, sampling.cfg_end],
        "log_interval": sampling.log_interval,
    }
    logged_config["provenance"] = {
        "dataset_and_preprocessing": preprocessing_metadata(config.data, config.model.image_size),
        "data_fingerprint": fingerprint,
        "validation_plan_hash": plan.plan_hash,
        "reg_target": config.model.reg.model_name if config.model.reg.enabled else None,
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
        f"reg={config.model.reg.enabled}, reg_variant="
        f"{config.model.reg.cls_predictor if config.model.reg.enabled else 'disabled'}, "
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
        if config.validation.validate_now:
            logger.event(f"forced validation started after epoch {state.epoch}")
            forced_metrics = validate_callback(state.epoch, state.epoch >= config.train.epochs)
            state.last_completed_validation_epoch = state.epoch
            config.validation.validate_now = False
            logger.log(forced_metrics, step=state.global_step)
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
            logger=logger,
            reg_encoder=reg_encoder,
            state=state,
            validation_callback=validate_callback,
        )
        logger.event(f"training complete: epoch={state.epoch}, step={state.global_step}")
    finally:
        logger.event("training shutdown")
        logger.finish()


if __name__ == "__main__":
    main()
