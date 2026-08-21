from __future__ import annotations

from jit_reg.config import WandbConfig
from jit_reg.wandb_utils import RunLogger


def _logger() -> RunLogger:
    return RunLogger(WandbConfig(enabled=False), {}, True, log_interval=1)


def test_training_console_metrics_match_waypoint_training(capsys) -> None:
    metrics = {
        "train/epoch": 0.25,
        "train/loss": 1.0,
        "train/full_velocity_loss": 0.2,
        "train/base_velocity_loss": 0.3,
        "train/repa_loss": 0.4,
        "train/repa_cosine_similarity": 0.5,
        "train/local_timestep_mean": 0.6,
        "train/global_timestep_mean": 0.7,
        "train/semantic_stage_fraction": 0.5,
        "train/lr": 2e-4,
        "train/gradient_norm": 1.2,
        "throughput/images_per_second": 100.0,
        "throughput/step_seconds": 2.0,
        "throughput/data_wait_seconds": 0.1,
        "system/peak_memory_gib": 12.0,
    }
    _logger().log_training(metrics, step=1)
    output = capsys.readouterr().out
    assert "full_velocity=0.200000" in output
    assert "base_velocity=0.300000" in output
    assert "local_t=0.600" in output
    assert "semantic=0.500" in output
    assert "velocity=" not in output.replace("full_velocity=", "").replace(
        "base_velocity=", ""
    )


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
