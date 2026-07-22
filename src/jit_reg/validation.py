from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .rng import preserve_global_rng
from .sampling import class_balanced_labels


@dataclass
class ValidationPlan:
    labels: Tensor
    sample_ids: Tensor
    noise_seeds: Tensor
    plan_hash: str

    def prefix(self, count: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.labels[:count], self.sample_ids[:count], self.noise_seeds[:count]

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
    periodic_samples: int = 5_000,
    num_classes: int = 1_000,
    seed: int = 44_721,
) -> ValidationPlan:
    labels = class_balanced_labels(final_samples, num_classes)
    class_balanced_labels(periodic_samples, num_classes)
    if periodic_samples > final_samples:
        raise ValueError("periodic plan must be a prefix of final plan")
    sample_ids = torch.arange(final_samples, dtype=torch.int64)
    generator = torch.Generator().manual_seed(seed)
    noise_seeds = torch.randint(0, 2**63 - 1, (final_samples,), generator=generator)
    values = {"labels": labels, "sample_ids": sample_ids, "noise_seeds": noise_seeds}
    return ValidationPlan(**values, plan_hash=_plan_digest(values))


def noise_for_samples(
    seeds: Tensor,
    shape: tuple[int, ...],
    device: str | torch.device,
    dtype: torch.dtype = torch.float16,
    *,
    seed_offset: int = 0,
) -> Tensor:
    """Generate each example independently, invariant to batching and rank partitioning."""
    samples = []
    modulus = 2**63 - 1
    for seed in seeds.cpu().tolist():
        generator = torch.Generator(device=device).manual_seed((seed + seed_offset) % modulus)
        samples.append(torch.randn(shape, generator=generator, device=device, dtype=dtype))
    return torch.stack(samples)


def should_validate(
    completed_epoch: int,
    total_epochs: int,
    interval: int,
    last_completed_validation_epoch: int | None = None,
    validate_now: bool = False,
) -> bool:
    if completed_epoch == last_completed_validation_epoch and not validate_now:
        return False
    return (
        validate_now
        or completed_epoch == total_epochs
        or (interval > 0 and completed_epoch % interval == 0)
    )


def run_preserving_training_state(model: torch.nn.Module, function: object) -> object:
    was_training = model.training
    with preserve_global_rng():
        model.eval()
        try:
            return function()  # type: ignore[operator]
        finally:
            model.train(was_training)
