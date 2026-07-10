from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .checkpoint import EMA
from .codecs import load_codecs
from .config import ExperimentConfig
from .constants import IMAGENET_REPO, IMAGENET_REVISION
from .evaluation import evaluate_image_directories
from .latents import LatentStatistics
from .model.compile import compile_model_regions
from .model.dit import DiffusionTransformer
from .rng import preserving_global_rng
from .sampling import euler_sample
from .validation import ValidationPlan, noise_for_samples


def _distributed() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _save_rgb(batch: torch.Tensor, ids: torch.Tensor, directory: Path) -> None:
    batch = batch.detach().float().clamp(0, 1).mul(255).round().byte().cpu()
    for image, sample_id in zip(batch, ids.tolist(), strict=True):
        array = image.permute(1, 2, 0).numpy()
        Image.fromarray(array, mode="RGB").save(directory / f"{sample_id:06d}.png")


def _save_labeled_grid(
    batch: torch.Tensor, labels: torch.Tensor, names: list[str], path: Path
) -> None:
    pixels = batch.detach().float().clamp(0, 1).mul(255).round().byte().cpu()
    _, height, width = pixels[0].shape
    caption_height = 20
    canvas = Image.new("RGB", (4 * width, 4 * (height + caption_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (tensor, class_id) in enumerate(zip(pixels, labels.tolist(), strict=True)):
        row, column = divmod(index, 4)
        image = Image.fromarray(tensor.permute(1, 2, 0).numpy(), mode="RGB")
        x, y = column * width, row * (height + caption_height)
        canvas.paste(image, (x, y))
        name = names[class_id] if class_id < len(names) else "unknown"
        draw.text((x + 2, y + height + 2), f"{class_id}: {name[:28]}", fill="black")
    canvas.save(path)


def materialize_real_validation(directory: str | Path) -> Path:
    """Cache the unaugmented Hugging Face ImageNet validation pixels as PNG."""
    directory = Path(directory)
    marker = directory / "complete.json"
    if marker.exists():
        metadata = json.loads(marker.read_text())
        if metadata.get("revision") != IMAGENET_REVISION or metadata.get("count") != 50_000:
            raise ValueError("real-image feature source was made from a different dataset revision")
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    dataset = load_dataset(IMAGENET_REPO, split="validation", revision=IMAGENET_REVISION)
    for index, row in enumerate(dataset):
        row["image"].convert("RGB").save(directory / f"{index:06d}.png")
    marker.write_text(json.dumps({"revision": IMAGENET_REVISION, "count": len(dataset)}) + "\n")
    names = getattr(dataset.features["label"], "names", None)
    if names:
        (directory / "class-names.json").write_text(json.dumps(names) + "\n")
    return directory


@torch.inference_mode()
@preserving_global_rng
def validate_ema(
    *,
    ema: EMA,
    config: ExperimentConfig,
    statistics: LatentStatistics,
    plan: ValidationPlan,
    completed_epoch: int,
    final: bool,
    output_root: str | Path,
) -> tuple[dict[str, float], Path]:
    rank, world_size = _distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    count = config.validation.final_samples if final else config.validation.periodic_samples
    name = "final-50k" if final else "periodic-5k"
    output = Path(output_root) / f"epoch-{completed_epoch:04d}-{name}"
    output.mkdir(parents=True, exist_ok=True)
    real = (
        materialize_real_validation(Path(output_root) / "real-imagenet-validation")
        if rank == 0
        else None
    )

    model = DiffusionTransformer(config.model).to(device)
    model.load_state_dict(ema.state_dict())
    model.eval()
    execution_model = compile_model_regions(model, config.compile)
    codec = load_codecs(config.codec, str(device))
    if isinstance(codec, tuple):
        codec = codec[0] if config.codec == "rae" else codec[1]
    labels = plan.labels[:count]
    seeds = plan.noise_seeds[:count]
    local_indices = torch.arange(rank, count, world_size)
    started = time.perf_counter()
    for begin in range(0, local_indices.numel(), config.validation.batch_size):
        positions = local_indices[begin : begin + config.validation.batch_size]
        batch_labels = labels[positions].to(device)
        noise = noise_for_samples(
            seeds[positions],
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
        _save_rgb(images, positions, output)

    if rank == 0:
        grid_noise = (
            plan.visualization_noise_16
            if config.latent_shape[-1] == 16
            else plan.visualization_noise_8
        ).to(device)
        grid_labels = plan.visualization_labels.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            grid_latents = euler_sample(
                execution_model,
                grid_noise,
                grid_labels,
                config.validation.sampling_steps,
                config.validation.guidance_scale,
                config.validation.guidance_start,
            )
            grid_images = codec.decode(statistics.denormalize(grid_latents)).clamp(0, 1)
        assert real is not None
        names_path = real / "class-names.json"
        class_names: list[str] = json.loads(names_path.read_text()) if names_path.exists() else []
        grid_path = output.parent / f"{output.name}-fixed-grid.png"
        _save_labeled_grid(grid_images, plan.visualization_labels, class_names, grid_path)
    _barrier()

    metrics: dict[str, float] = {}
    if rank == 0:
        assert real is not None
        raw = evaluate_image_directories(
            output, real, cuda=device.type == "cuda", inception_score=final
        )
        suffix = "50K" if final else "5K"
        metrics[f"validation/FID-{suffix}"] = raw["FID"]
        if final:
            metrics["validation/Inception Score"] = raw["Inception Score"]
        metrics["validation/images_per_second"] = count / max(time.perf_counter() - started, 1e-9)
        metrics["validation/sampling_nfe"] = float(config.validation.sampling_steps)
        metrics["validation/peak_memory_gib"] = (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
        )
    _barrier()
    return metrics, output
