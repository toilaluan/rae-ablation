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
    visualization_labels: Tensor
    visualization_noise_16: Tensor
    visualization_noise_8: Tensor
    periodic_samples: int
    plan_hash: str

    def periodic(self) -> tuple[Tensor, Tensor, Tensor]:
        n = self.periodic_samples
        return self.labels[:n], self.sample_ids[:n], self.noise_seeds[:n]

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


def _plan_digest(values: dict[str, object]) -> str:
    buffer = io.BytesIO()
    torch.save(values, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def create_validation_plan(
    final_samples: int = 50_000,
    periodic_samples: int = 5_000,
    num_classes: int = 1_000,
    seed: int = 44_721,
    visualization_classes: list[int] | None = None,
    channels: int = 1024,
) -> ValidationPlan:
    # Construct both to validate that the final plan and its periodic prefix are balanced.
    labels = class_balanced_labels(final_samples, num_classes)
    class_balanced_labels(periodic_samples, num_classes)
    if periodic_samples > final_samples:
        raise ValueError("periodic plan must be a prefix of final plan")
    sample_ids = torch.arange(final_samples, dtype=torch.int64)
    seed_generator = torch.Generator().manual_seed(seed)
    noise_seeds = torch.randint(0, 2**63 - 1, (final_samples,), generator=seed_generator)
    classes = torch.tensor(visualization_classes or list(range(16)), dtype=torch.int64)
    if classes.numel() != 16:
        raise ValueError("visualization plan requires a fixed 4x4 class list")
    noise_generator = torch.Generator().manual_seed(seed + 1)
    noise16 = torch.randn((16, channels, 16, 16), generator=noise_generator).half()
    noise8 = torch.randn((16, channels, 8, 8), generator=noise_generator).half()
    values: dict[str, object] = {
        "labels": labels,
        "sample_ids": sample_ids,
        "noise_seeds": noise_seeds,
        "visualization_labels": classes,
        "visualization_noise_16": noise16,
        "visualization_noise_8": noise8,
        "periodic_samples": periodic_samples,
    }
    return ValidationPlan(**values, plan_hash=_plan_digest(values))  # type: ignore[arg-type]


def noise_for_samples(
    seeds: Tensor,
    shape: tuple[int, int, int],
    device: str | torch.device,
    dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Generate each sample independently, invariant to batching and rank partitioning."""
    samples = []
    for seed in seeds.cpu().tolist():
        generator = torch.Generator(device=device).manual_seed(seed)
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
    return validate_now or completed_epoch == total_epochs or completed_epoch % interval == 0


def run_preserving_training_state(model: torch.nn.Module, function: object) -> object:
    """Run validation without changing model mode or any global training RNG."""
    was_training = model.training
    with preserve_global_rng():
        model.eval()
        try:
            return function()  # type: ignore[operator]
        finally:
            model.train(was_training)
