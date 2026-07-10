from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from .cache import ShardedLatentDataset, build_cache, read_manifest, verify_cache
from .checkpoint import EMA, CheckpointManager
from .codecs import load_codecs
from .compare import compare_runs
from .config import load_config
from .constants import (
    COMPACT_REPO,
    COMPACT_REVISION,
    IMAGENET_REPO,
    IMAGENET_REVISION,
    RAE_REPO,
    RAE_REVISION,
)
from .evaluation import evaluate_image_directories, save_evaluation
from .latents import LatentStatistics
from .model.compile import compile_model_regions
from .model.dit import DiffusionTransformer
from .optim import build_optimizer
from .rng import RNGStreams
from .sampling import euler_sample
from .training import TrainState, make_train_loader, train
from .validation import ValidationPlan, create_validation_plan
from .validation_runtime import validate_ema
from .wandb_utils import WandbLogger


def _base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=Path("configs/canonical.yaml"))
    parser.add_argument(
        "overrides", nargs="*", help="OmegaConf dot-list overrides, e.g. model.hidden_size=256"
    )
    return parser


def _initialize_distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world_size, device


def cache_main() -> None:
    parser = _base_parser("Cache pinned ImageNet RAE latents")
    parser.add_argument("--codec", choices=("rae", "rae_compact", "both"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if args.codec:
        config.cache.codec = args.codec
    if args.output:
        config.cache.directory = str(args.output)
    if args.max_samples is not None:
        config.cache.max_samples = args.max_samples
    if args.verify_only:
        completed = verify_cache(config.cache.directory)
        print(f"verified {len(completed)} shards")
        return
    _initialize_distributed()
    manifest = build_cache(config.cache)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"cache fingerprint: {manifest.fingerprint}")


def _training_parser() -> argparse.ArgumentParser:
    parser = _base_parser("Train the matched latent diffusion model")
    parser.add_argument("--codec", choices=("rae", "rae_compact"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--validate-now", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    return parser


def train_main() -> None:
    args = _training_parser().parse_args()
    config = load_config(args.config, args.overrides)
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

    rank, world_size, device = _initialize_distributed()
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
            args.resume,
            model,
            ema,
            manifest.fingerprint,
            plan.plan_hash,
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
    logger = WandbLogger(config.wandb, logged_config, rank == 0, resume_run_id)
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
        )
        if rank == 0:
            labels = ", ".join(str(value) for value in plan.visualization_labels.tolist())
            logger.log_grid(
                str(validation_output.parent / f"{validation_output.name}-fixed-grid.png"),
                f"ImageNet class IDs: {labels}",
                state.global_step,
            )
        return metrics

    if config.validation.validate_now and state.epoch > 0:
        forced_metrics = validate_callback(state.epoch, state.epoch >= config.train.epochs)
        state.last_completed_validation_epoch = state.epoch
        config.validation.validate_now = False
        logger.log(forced_metrics, step=state.global_step)

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
    logger.finish()


def sample_main() -> None:
    parser = _base_parser("Sample an EMA export")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--codec", choices=("rae", "rae_compact"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if args.codec:
        config.codec = args.codec
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from safetensors.torch import load_file

    model = DiffusionTransformer(config.model).to(device)
    model.load_state_dict(load_file(str(args.checkpoint), device=str(device)))
    model.eval()
    model = compile_model_regions(model, config.compile)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    labels = torch.tensor(args.classes, device=device)
    noise = torch.randn(
        (len(args.classes), *config.latent_shape), generator=generator, device=device
    )
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        latent = euler_sample(
            model,
            noise,
            labels,
            config.validation.sampling_steps,
            config.validation.guidance_scale,
            config.validation.guidance_start,
        )
        stats = LatentStatistics.load(
            Path(config.cache.directory) / f"stats-{config.codec}.safetensors"
        )
        codec = load_codecs(config.codec, str(device))
        if isinstance(codec, tuple):
            codec = codec[0] if config.codec == "rae" else codec[1]
        images = codec.decode(stats.denormalize(latent)).clamp(0, 1)
    from torchvision.utils import save_image

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(images, args.output, nrow=max(1, int(len(args.classes) ** 0.5)))


def eval_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated images")
    parser.add_argument("generated", type=Path)
    parser.add_argument("real", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fid-only", action="store_true")
    args = parser.parse_args()
    plan = ValidationPlan.load(args.plan)
    metrics = evaluate_image_directories(
        args.generated, args.real, cuda=torch.cuda.is_available(), inception_score=not args.fid_only
    )
    save_evaluation(args.output, metrics, plan.plan_hash)
    print(json.dumps(metrics, indent=2, sort_keys=True))


def compare_main() -> None:
    parser = argparse.ArgumentParser(description="Compare matched RAE ablation results")
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    comparison: dict[str, Any] = compare_runs(args.first, args.second)
    rendered = json.dumps(comparison, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)
