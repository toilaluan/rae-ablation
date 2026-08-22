from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .rng import preserve_global_rng


@dataclass
class ValidationPlan:
    indices: Tensor
    plan_hash: str

    def prefix(self, count: int) -> Tensor:
        if count < 1 or count > self.indices.numel():
            raise ValueError("validation sample count is outside the immutable plan")
        return self.indices[:count]

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
        plan = cls(**torch.load(path, map_location="cpu", weights_only=True))
        if plan.plan_hash != _plan_digest({"indices": plan.indices}):
            raise ValueError("validation plan hash does not match its indices")
        return plan


def _plan_digest(values: dict[str, Tensor]) -> str:
    indices = values["indices"].to(dtype=torch.int64, device="cpu").contiguous()
    return hashlib.sha256(indices.numpy().tobytes()).hexdigest()


def create_validation_plan(
    final_samples: int = 50_000,
    periodic_samples: int = 10_000,
    seed: int = 7,
    dataset_size: int = 50_000,
) -> ValidationPlan:
    if not 0 < periodic_samples <= final_samples <= dataset_size:
        raise ValueError("periodic plan must be a non-empty prefix of final plan")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator, dtype=torch.int64)[
        :final_samples
    ]
    values = {"indices": indices}
    return ValidationPlan(indices=indices, plan_hash=_plan_digest(values))


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
