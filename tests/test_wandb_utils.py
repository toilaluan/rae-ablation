from __future__ import annotations

from typing import Any

from cascade_dit.config import WandbConfig
from cascade_dit.wandb_utils import RunLogger


def _logger() -> RunLogger:
    return RunLogger(WandbConfig(enabled=False), {}, True, log_interval=1)


def test_training_console_uses_new_metric_names(capsys) -> None:
    metrics = {
        "train/epoch": 0.25,
        "train/loss": 1.0,
        "train/image_latent_velocity_loss": 0.2,
        "train/semantic_full_velocity_loss": 0.3,
        "train/semantic_base_velocity_loss": 0.4,
        "train/predicted_semantic_condition_fraction": 0.25,
        "train/lr": 2e-4,
        "train/gradient_norm": 1.2,
    }
    _logger().log_training(metrics, step=1)
    output = capsys.readouterr().out
    assert "image=0.200000" in output
    assert "semantic_full=0.300000" in output
    assert "predicted_condition=0.250" in output


def test_wandb_training_logs_only_at_interval_or_forced_step() -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.calls: list[tuple[dict[str, Any], int | None]] = []

        def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
            self.calls.append((metrics, step))

    logger = RunLogger(WandbConfig(enabled=False), {}, False, log_interval=3)
    run = FakeRun()
    logger.run = run
    logger.log_training({"value": 1}, step=1)
    logger.log_training({"value": 3}, step=3)
    logger.log_training({"value": 4}, step=4, force=True)
    assert run.calls == [({"value": 3}, 3), ({"value": 4}, 4)]
