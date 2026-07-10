from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from .cache import ShardedLatentDataset, build_cache, read_manifest, verify_cache
from .checkpoint import EMA, CheckpointManager
from .codecs import load_codecs
from .compare import compare_runs
from .config import ExperimentConfig, load_config
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
from .validation import ValidationPlan, create_validation_plan, noise_for_samples
from .validation_runtime import validate_ema
from .wandb_utils import WandbLogger

torch._dynamo.config.cache_size_limit = 1000

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


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, values: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _save_sample_batch(images: torch.Tensor, sample_ids: torch.Tensor, output: Path) -> None:
    from PIL import Image

    pixels = images.detach().float().clamp(0, 1).mul(255).round().byte().cpu()
    for image, sample_id in zip(pixels, sample_ids.tolist(), strict=True):
        target = output / f"{sample_id:06d}.png"
        temporary = output / f".{sample_id:06d}.png.partial"
        array = image.permute(1, 2, 0).numpy()
        Image.fromarray(array, mode="RGB").save(temporary, format="PNG")
        temporary.replace(target)


def _validate_fid_plan(
    plan: ValidationPlan, sample_count: int, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sample_count < 1 or sample_count % num_classes:
        raise ValueError("FID sample count must be positive and divisible by model.num_classes")
    if min(plan.labels.numel(), plan.sample_ids.numel(), plan.noise_seeds.numel()) < sample_count:
        raise ValueError(f"validation plan does not contain {sample_count} samples")
    labels = plan.labels[:sample_count].to(torch.int64)
    sample_ids = plan.sample_ids[:sample_count].to(torch.int64)
    noise_seeds = plan.noise_seeds[:sample_count].to(torch.int64)
    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError("FID plan contains a label outside model.num_classes")
    counts = torch.bincount(labels, minlength=num_classes)
    expected = sample_count // num_classes
    if counts.numel() != num_classes or not torch.all(counts == expected):
        raise ValueError(f"FID plan must contain exactly {expected} samples per class")
    if sample_ids.unique().numel() != sample_count or torch.any(sample_ids < 0):
        raise ValueError("FID plan sample IDs must be unique and non-negative")
    return labels, sample_ids, noise_seeds


def _sample_fid_directory(
    *,
    checkpoint: Path,
    output: Path,
    plan_path: Path,
    sample_count: int,
    config: ExperimentConfig,
) -> None:
    from safetensors.torch import load_file

    rank, world_size, device = _initialize_distributed()
    plan = ValidationPlan.load(plan_path)
    labels, sample_ids, noise_seeds = _validate_fid_plan(
        plan, sample_count, config.model.num_classes
    )
    expected_names = {f"{sample_id:06d}.png" for sample_id in sample_ids.tolist()}
    manifest_path = output / ".sampling.json"
    manifest: dict[str, Any] | None = None
    completion_state = [False]
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "checkpoint_sha256": _checkpoint_sha256(checkpoint),
            "validation_plan_hash": plan.plan_hash,
            "codec": config.codec,
            "sample_count": sample_count,
            "num_classes": config.model.num_classes,
            "samples_per_class": sample_count // config.model.num_classes,
            "sampling_steps": config.validation.sampling_steps,
            "guidance_scale": config.validation.guidance_scale,
            "guidance_start": config.validation.guidance_start,
        }
        existing_images = {path.name for path in output.glob("*.png")}
        unexpected = existing_images - expected_names
        if unexpected:
            example = sorted(unexpected)[0]
            raise ValueError(f"output contains an unexpected PNG: {example}")
        if existing_images and not manifest_path.exists():
            raise ValueError("refusing to resume FID sampling without .sampling.json")
        if manifest_path.exists():
            existing_manifest = json.loads(manifest_path.read_text())
            comparable = {key: existing_manifest.get(key) for key in manifest}
            if comparable != manifest:
                raise ValueError("existing FID sampling manifest does not match this request")
        completion_state[0] = existing_images == expected_names
        _write_json_atomic(manifest_path, {**manifest, "complete": completion_state[0]})
        for partial in output.glob(".*.png.partial"):
            partial.unlink()
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(completion_state, src=0)
        torch.distributed.barrier()
    if completion_state[0]:
        if rank == 0:
            print(
                f"found {sample_count} complete images "
                f"({sample_count // config.model.num_classes} per class) in {output}"
            )
        return

    model = DiffusionTransformer(config.model).to(device)
    model.load_state_dict(load_file(str(checkpoint), device=str(device)))
    model.eval()
    execution_model = compile_model_regions(model, config.compile)
    statistics = LatentStatistics.load(
        Path(config.cache.directory) / f"stats-{config.codec}.safetensors"
    )
    codec = load_codecs(config.codec, str(device))
    if isinstance(codec, tuple):
        codec = codec[0] if config.codec == "rae" else codec[1]

    local_positions = torch.arange(rank, sample_count, world_size)
    for begin in range(0, local_positions.numel(), config.validation.batch_size):
        positions = local_positions[begin : begin + config.validation.batch_size]
        pending = torch.tensor(
            [
                position
                for position in positions.tolist()
                if not (output / f"{sample_ids[position].item():06d}.png").exists()
            ],
            dtype=torch.int64,
        )
        if pending.numel() == 0:
            continue
        batch_labels = labels[pending].to(device)
        noise = noise_for_samples(
            noise_seeds[pending],
            config.latent_shape,
            device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            normalized = euler_sample(
                execution_model,
                noise,
                batch_labels,
                config.validation.sampling_steps,
                config.validation.guidance_scale,
                config.validation.guidance_start,
            )
            images = codec.decode(statistics.denormalize(normalized))
        _save_sample_batch(images, sample_ids[pending], output)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        actual_names = {path.name for path in output.glob("*.png")}
        if actual_names != expected_names:
            missing = len(expected_names - actual_names)
            raise RuntimeError(f"FID sampling finished with {missing} missing images")
        assert manifest is not None
        _write_json_atomic(manifest_path, {**manifest, "complete": True})
        print(
            f"wrote {sample_count} images ({sample_count // config.model.num_classes} per class) "
            f"to {output}"
        )


def _validate_generated_sampling(generated: Path, plan: ValidationPlan) -> None:
    manifest_path = generated / ".sampling.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise ValueError("generated FID sample directory is incomplete")
    if manifest.get("validation_plan_hash") != plan.plan_hash:
        raise ValueError("generated samples use a different validation plan")
    count = manifest.get("sample_count")
    num_classes = manifest.get("num_classes")
    samples_per_class = manifest.get("samples_per_class")
    if not all(isinstance(value, int) for value in (count, num_classes, samples_per_class)):
        raise ValueError("generated FID sampling manifest has invalid counts")
    if count != num_classes * samples_per_class:
        raise ValueError("generated FID sampling manifest is not class balanced")
    if len(list(generated.glob("*.png"))) != count:
        raise ValueError("generated FID image count does not match its sampling manifest")


def sample_main() -> None:
    parser = _base_parser("Sample an EMA export")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--codec", choices=("rae", "rae_compact"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--classes", type=int, nargs="+")
    mode.add_argument(
        "--fid-samples",
        type=int,
        nargs="?",
        const=50_000,
        metavar="N",
        help="write a balanced PNG directory (default: 50000 images)",
    )
    parser.add_argument("--plan", type=Path, help="immutable validation plan for FID sampling")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if args.codec:
        config.codec = args.codec
    if args.cache_dir:
        config.cache.directory = str(args.cache_dir)
    if args.fid_samples is not None:
        if args.plan is None:
            parser.error("--plan is required with --fid-samples")
        _sample_fid_directory(
            checkpoint=args.checkpoint,
            output=args.output,
            plan_path=args.plan,
            sample_count=args.fid_samples,
            config=config,
        )
        return
    assert args.classes is not None
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
    _validate_generated_sampling(args.generated, plan)
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
