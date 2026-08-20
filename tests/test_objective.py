from __future__ import annotations

import pytest
import torch

from jit_reg.config import MixingConfig
from jit_reg.objective import (
    diffusion_losses,
    mix_endpoints,
    noisy_interpolate,
    sample_mix_ratio,
    sample_timesteps,
    shift_timestep,
    target_velocity,
)


def test_flow_math_and_direct_velocity_loss() -> None:
    clean = torch.tensor([[[[2.0]]]])
    noise = torch.tensor([[[[-1.0]]]])
    timestep = torch.tensor([0.25])
    assert noisy_interpolate(clean, noise, timestep).item() == pytest.approx(1.25)
    assert target_velocity(clean, noise).item() == pytest.approx(-3.0)
    semantic = torch.ones_like(clean)
    projection = semantic.flatten(2).transpose(1, 2)
    losses = diffusion_losses(
        clean,
        noise,
        target_velocity(clean, noise),
        projection,
        semantic,
        repa_coefficient=0.5,
    )
    assert losses.velocity.item() == 0
    assert losses.repa.item() == pytest.approx(-1.0)
    assert losses.total.item() == pytest.approx(-0.5)


def test_token_mixing_broadcasts_r_and_baseline_is_exact() -> None:
    semantic = torch.zeros(2, 3, 2, 2)
    target = torch.full_like(semantic, 10)
    ratio = torch.tensor([0.25, 1.0])
    mixed = mix_endpoints(semantic, target, ratio)
    assert torch.all(mixed[0] == 2.5)
    assert torch.equal(mixed[1], target[1])
    baseline = sample_mix_ratio(
        16,
        torch.Generator().manual_seed(4),
        "cpu",
        MixingConfig(mode="target_baseline"),
    )
    assert torch.equal(baseline, torch.ones(16))


def test_logit_normal_has_configurable_endpoint_atom() -> None:
    endpoint = sample_mix_ratio(
        32,
        torch.Generator().manual_seed(7),
        "cpu",
        MixingConfig(endpoint_probability=1.0),
    )
    assert torch.equal(endpoint, torch.ones(32))
    continuous = sample_mix_ratio(
        32,
        torch.Generator().manual_seed(7),
        "cpu",
        MixingConfig(endpoint_probability=0.0),
    )
    assert torch.all((continuous > 0) & (continuous < 1))


def test_logit_normal_timestep_shift() -> None:
    assert shift_timestep(torch.tensor(0.5), 4).item() == pytest.approx(0.8)
    values = sample_timesteps(32, torch.Generator().manual_seed(7), "cpu")
    assert torch.all((values > 0) & (values < 1))
