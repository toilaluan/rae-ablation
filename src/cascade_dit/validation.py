from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .config import ExperimentConfig
from .data import EndpointBatchPreprocessor, EndpointPhase
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
def validate_phase_loss(
    model: torch.nn.Module,
    loader: object,
    batch_preprocessor: EndpointBatchPreprocessor,
    config: ExperimentConfig,
    *,
    phase: EndpointPhase,
    epoch: int,
    device: torch.device,
) -> dict[str, float]:
    streams = RNGStreams.create(config.experiment.seed + 1_000_003 + epoch, device)
    totals: list[float] = []
    velocities: list[float] = []
    auxiliary: list[float] = []
    weighted_auxiliary: list[float] = []
    ratios: list[Tensor] = []
    pure_semantic_count = 0
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
                phase=phase,
            )
            totals.append(float(result.loss.total))
            velocities.append(float(result.loss.velocity_mse))
            auxiliary.append(float(result.loss.pixel_auxiliary_mse))
            weighted_auxiliary.append(float(result.loss.weighted_pixel_auxiliary_mse))
            pure_semantic_count += result.loss.pure_semantic_count
            ratios.append(batch.pixel_ratio.detach().float().cpu())
    finally:
        model.train(was_training)
    if not totals:
        raise ValueError("validation loader produced no batches")
    all_ratios = torch.cat(ratios)
    mixed_count = int(((all_ratios > 0) & (all_ratios < 1)).sum())
    return {
        "loss": sum(totals) / len(totals),
        "velocity_mse": sum(velocities) / len(velocities),
        "pixel_auxiliary_mse": sum(auxiliary) / len(auxiliary),
        "weighted_pixel_auxiliary_mse": sum(weighted_auxiliary) / len(weighted_auxiliary),
        "pure_semantic_count": float(pure_semantic_count),
        "mixed_count": float(mixed_count),
        "pixel_ratio_mean": float(all_ratios.mean()),
        "pixel_ratio_min": float(all_ratios.min()),
        "pixel_ratio_max": float(all_ratios.max()),
    }
