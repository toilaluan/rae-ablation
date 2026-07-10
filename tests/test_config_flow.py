from __future__ import annotations

import math

import pytest
import torch
from omegaconf.errors import ConfigKeyError

from rae_ablation.config import ExperimentConfig, load_config
from rae_ablation.flow import (
    flow_losses,
    internal_guidance,
    noisy_interpolate,
    shift_timestep,
    timestep_shift_alpha,
)
from rae_ablation.optim import learning_rate_at_epoch


def test_structured_config_and_unknown_key() -> None:
    config = load_config("configs/canonical.yaml", ["model.hidden_size=64", "model.num_heads=4"])
    assert isinstance(config, ExperimentConfig)
    assert config.model.hidden_size == 64
    with pytest.raises(ConfigKeyError):
        load_config("configs/canonical.yaml", ["model.typo=1"])


def test_config_validation() -> None:
    config = ExperimentConfig()
    config.model.hidden_size = 63
    with pytest.raises(ValueError, match="divisible"):
        config.validate()


def test_shift_values_and_velocity_losses() -> None:
    assert timestep_shift_alpha((1024, 16, 16)) == 8
    assert timestep_shift_alpha((1024, 8, 8)) == 4
    u = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(shift_timestep(u, 4), torch.tensor([0.0, 0.8, 1.0]))
    clean = torch.randn(2, 3, 2, 2)
    noise = torch.randn_like(clean)
    timestep = torch.tensor([0.2, 0.8])
    x_t = noisy_interpolate(clean, noise, timestep)
    assert x_t.shape == clean.shape
    losses = flow_losses(clean, noise, timestep, clean, clean)
    assert losses.total.item() == pytest.approx(0, abs=1e-12)


def test_internal_guidance_in_x_space() -> None:
    early = torch.ones(2, 1, 1, 1)
    full = 3 * early
    guided = internal_guidance(early, full, torch.tensor([0.05, 0.5]), scale=2, start=0.1)
    assert guided.flatten().tolist() == [3, 5]


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [(0, 2e-4), (25, 2e-4), (37.5, 1.1e-4), (50, 2e-5), (80, 2e-5)],
)
def test_learning_rate_recipe(epoch: float, expected: float) -> None:
    assert math.isclose(learning_rate_at_epoch(epoch), expected)
