from __future__ import annotations

from typing import Any

from jit_reg.config import WandbConfig
from jit_reg.wandb_utils import RunLogger


def _logger() -> RunLogger:
    return RunLogger(WandbConfig(enabled=False), {}, True, log_interval=1)


def test_training_console_metrics_match_endpoint_training(capsys) -> None:
    metrics = {
        "train/epoch": 0.25,
        "train/loss": 1.0,
        "train/endpoint_loss": 0.2,
        "train/timestep_mean": 0.6,
        "train/lr": 2e-4,
        "train/gradient_norm": 1.2,
        "throughput/images_per_second": 100.0,
        "throughput/step_seconds": 2.0,
        "throughput/data_wait_seconds": 0.1,
        "system/peak_memory_gib": 12.0,
    }
    _logger().log_training(metrics, step=1)
    output = capsys.readouterr().out
    assert "endpoint=0.200000" in output
    assert "t=0.600" in output


def test_console_logging_does_not_interrupt_training_on_missing_metric(capsys) -> None:
    _logger().log_training({}, step=1)
    assert "loss=n/a" in capsys.readouterr().out


def test_decoder_console_logging_uses_current_metric_keys(capsys) -> None:
    metrics = {
        "decoder/loss": 1.0,
        "decoder/l1": 0.1,
        "decoder/lpips": 0.2,
        "decoder/generator_adversarial": 0.3,
        "decoder/discriminator": 0.4,
        "decoder/gan_active": 1.0,
        "decoder/lr": 2e-4,
        "throughput/images_per_second": 10.0,
        "throughput/step_seconds": 0.5,
    }
    _logger().log_decoder_training(metrics, step=1)
    output = capsys.readouterr().out
    assert "loss=1.000000" in output
    assert "gan=1" in output


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
    logger.log_training({"value": 2}, step=2)
    logger.log_training({"value": 3}, step=3)
    logger.log_training({"value": 4}, step=4, force=True)
    assert run.calls == [({"value": 3}, 3), ({"value": 4}, 4)]
