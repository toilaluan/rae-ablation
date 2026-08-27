from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

from .config import ExperimentConfig
from .data import JointBatchPreprocessor
from .objective import pixel_reconstruction_loss
from .reconstruction import ReconstructionBatchPreprocessor
from .rng import RNGStreams
from .sampling import class_balanced_labels
from .training import compute_training_loss


@dataclass
class ValidationPlan:
    labels: Tensor
    noise_seeds: Tensor
    plan_hash: str

    def prefix(self, count: int) -> tuple[Tensor, Tensor]:
        return self.labels[:count], self.noise_seeds[:count]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        if target.exists():
            existing = self.load(target)
            if existing.plan_hash != self.plan_hash:
                raise ValueError("refusing to overwrite a different immutable validation plan")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".partial")
        torch.save(self.__dict__, temporary)
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> ValidationPlan:
        return cls(**torch.load(path, map_location="cpu", weights_only=True))


@dataclass(frozen=True)
class ReconstructionValidationResult:
    metrics: dict[str, float]
    sample_count: int
    originals: Tensor | None
    reconstructions: Tensor | None
    labels: Tensor | None


def _plan_digest(values: dict[str, Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(values, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def create_validation_plan(
    final_samples: int = 50_000,
    periodic_samples: int = 10_000,
    num_classes: int = 1_000,
    seed: int = 44_721,
) -> ValidationPlan:
    labels = class_balanced_labels(final_samples, num_classes)
    class_balanced_labels(periodic_samples, num_classes)
    if periodic_samples > final_samples:
        raise ValueError("periodic plan must be a prefix of final plan")
    generator = torch.Generator().manual_seed(seed)
    noise_seeds = torch.randint(0, 2**63 - 1, (final_samples,), generator=generator)
    values = {"labels": labels, "noise_seeds": noise_seeds}
    return ValidationPlan(**values, plan_hash=_plan_digest(values))


@torch.inference_mode()
def validate_joint_loss(
    model: torch.nn.Module,
    loader: object,
    batch_preprocessor: JointBatchPreprocessor,
    config: ExperimentConfig,
    lpips_model: torch.nn.Module,
    *,
    epoch: int,
    device: torch.device,
) -> dict[str, float]:
    streams = RNGStreams.create(config.experiment.seed + 1_000_003 + epoch, device)
    totals: list[float] = []
    full_velocities: list[float] = []
    base_velocities: list[float] = []
    pixel_l1: list[float] = []
    pixel_lpips: list[float] = []
    was_training = model.training
    model.eval()
    try:
        for batch_index, raw_batch in enumerate(loader):  # type: ignore[union-attr]
            if batch_index >= config.validation.batches:
                break
            batch = batch_preprocessor(raw_batch)
            result = compute_training_loss(
                model,
                batch,
                streams,
                config,
                lpips_model,
            )
            totals.append(float(result.loss.total))
            full_velocities.append(float(result.loss.full_velocity_mse))
            base_velocities.append(float(result.loss.base_velocity_mse))
            pixel_l1.append(float(result.loss.pixel_l1))
            pixel_lpips.append(float(result.loss.pixel_lpips))
    finally:
        model.train(was_training)
    if not totals:
        raise ValueError("validation loader produced no batches")
    return {
        "loss": sum(totals) / len(totals),
        "full_velocity_mse": sum(full_velocities) / len(full_velocities),
        "base_velocity_mse": sum(base_velocities) / len(base_velocities),
        "pixel_l1": sum(pixel_l1) / len(pixel_l1),
        "pixel_lpips": sum(pixel_lpips) / len(pixel_lpips),
    }


def _distributed_preview(
    local: list[tuple[int, int, Tensor, Tensor]],
) -> list[tuple[int, int, Tensor, Tensor]]:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return sorted(local, key=lambda item: item[0])
    gathered: list[object] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local)
    combined: list[tuple[int, int, Tensor, Tensor]] = []
    for values in gathered:
        if not isinstance(values, list):
            raise TypeError("distributed reconstruction preview payload is invalid")
        combined.extend(values)
    return sorted(combined, key=lambda item: item[0])


@torch.inference_mode()
def validate_reconstruction(
    model: torch.nn.Module,
    loader: object,
    batch_preprocessor: ReconstructionBatchPreprocessor,
    config: ExperimentConfig,
    lpips_model: torch.nn.Module,
    *,
    device: torch.device,
) -> ReconstructionValidationResult:
    preview_count = min(
        config.validation.reconstruction.preview_samples,
        config.validation.reconstruction.samples_per_class * config.model.num_classes,
    )
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    local_preview: list[tuple[int, int, Tensor, Tensor]] = []
    was_training = model.training
    model.eval()
    try:
        for raw_batch in loader:  # type: ignore[union-attr]
            batch = batch_preprocessor(raw_batch)
            timestep = torch.full(
                (batch.semantic.shape[0],),
                config.sampling.final_decode_timestep,
                device=device,
                dtype=torch.float32,
            )
            force_class_drop = torch.zeros_like(batch.labels, dtype=torch.bool)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(
                    batch.semantic,
                    timestep,
                    batch.labels,
                    decode_pixels=True,
                    force_class_drop=force_class_drop,
                )
                prediction = getattr(output, "rgb", None)
                if not isinstance(prediction, Tensor):
                    raise TypeError("reconstruction validation requires an RGB model output")
                loss = pixel_reconstruction_loss(
                    prediction, batch.pixels, lpips_model, config.objective
                )
            count = batch.pixels.shape[0]
            mse = (prediction.float() - batch.pixels.float()).square().flatten(1).mean(1)
            psnr = -10 * torch.log10(mse.clamp_min(1e-12))
            totals += torch.tensor(
                [
                    float(loss.pixel_l1) * count,
                    float(loss.pixel_lpips) * count,
                    float(psnr.sum()),
                    count,
                ],
                dtype=torch.float64,
                device=device,
            )
            wanted = batch.positions < preview_count
            if wanted.any():
                positions = batch.positions[wanted].tolist()
                labels = batch.labels[wanted.to(batch.labels.device)].tolist()
                originals = (
                    batch.pixels[wanted.to(batch.pixels.device)]
                    .float()
                    .clamp(0, 1)
                    .mul(255)
                    .round()
                    .byte()
                    .cpu()
                )
                reconstructions = (
                    prediction[wanted.to(prediction.device)]
                    .float()
                    .clamp(0, 1)
                    .mul(255)
                    .round()
                    .byte()
                    .cpu()
                )
                local_preview.extend(
                    (position, label, original, reconstruction)
                    for position, label, original, reconstruction in zip(
                        positions,
                        labels,
                        originals,
                        reconstructions,
                        strict=True,
                    )
                )
    finally:
        model.train(was_training)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals)
    count = int(totals[-1].item())
    if count < 1:
        raise ValueError("reconstruction validation loader produced no samples")
    preview = _distributed_preview(local_preview)[:preview_count]
    if len(preview) != preview_count:
        raise ValueError("reconstruction validation did not produce the fixed preview")
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    if rank == 0:
        originals = torch.stack([item[2] for item in preview])
        reconstructions = torch.stack([item[3] for item in preview])
        labels = torch.tensor([item[1] for item in preview], dtype=torch.int64)
    else:
        originals = reconstructions = labels = None
    return ReconstructionValidationResult(
        metrics={
            "l1": totals[0].item() / count,
            "lpips": totals[1].item() / count,
            "psnr": totals[2].item() / count,
        },
        sample_count=count,
        originals=originals,
        reconstructions=reconstructions,
        labels=labels,
    )


def save_reconstruction_preview(
    result: ReconstructionValidationResult,
    path: str | Path,
) -> None:
    if (
        result.originals is None
        or result.reconstructions is None
        or result.labels is None
    ):
        raise ValueError("reconstruction preview is only available on rank zero")
    count = result.originals.shape[0]
    if (
        result.originals.shape != result.reconstructions.shape
        or result.originals.shape[1:] != (3, 128, 128)
        or result.labels.shape != (count,)
    ):
        raise ValueError("reconstruction preview tensors have invalid shapes")
    header = 20
    row_height = 146
    canvas = Image.new("RGB", (256, header + count * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((42, 3), "Original", fill="black")
    draw.text((164, 3), "Reconstruction", fill="black")
    originals = result.originals.permute(0, 2, 3, 1).numpy()
    reconstructions = result.reconstructions.permute(0, 2, 3, 1).numpy()
    for index, (original, reconstruction, label) in enumerate(
        zip(originals, reconstructions, result.labels.tolist(), strict=True)
    ):
        y = header + index * row_height
        canvas.paste(Image.fromarray(np.asarray(original)), (0, y))
        canvas.paste(Image.fromarray(np.asarray(reconstruction)), (128, y))
        draw.text((2, y + 130), f"class {label}", fill="black")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)


def save_reconstruction_evaluation(
    path: str | Path,
    result: ReconstructionValidationResult,
    *,
    timestep: float,
    cache_fingerprint: str,
    ema_sha256: str,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".partial-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            {
                "metrics": result.metrics,
                "num_samples": result.sample_count,
                "final_decode_timestep": timestep,
                "reconstruction_cache_fingerprint": cache_fingerprint,
                "ema_sha256": ema_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(temporary, target)
