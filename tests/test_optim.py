from __future__ import annotations

import pytest
import torch

from cascade_dit.optim import build_training_scheduler


def _optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=1.0)


def test_constant_scheduler_keeps_lr_after_optional_warmup() -> None:
    optimizer = _optimizer()
    scheduler = build_training_scheduler(
        optimizer,
        kind="constant",
        steps_per_epoch=2,
        total_epochs=3,
        warmup_epochs=1,
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)


def test_unknown_scheduler_is_rejected() -> None:
    with pytest.raises(ValueError, match="constant or cosine"):
        build_training_scheduler(
            _optimizer(), kind="linear", steps_per_epoch=1, total_epochs=1
        )
