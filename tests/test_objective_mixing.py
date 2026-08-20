from __future__ import annotations

import pytest
import torch

from jit_reg.objective import (
    ddt_losses,
    mix_endpoints,
    noisy_interpolate,
    sample_mix_ratios,
    sample_timesteps,
    target_velocity,
    x_to_velocity,
)


def test_x_prediction_flow_math_and_weighted_dual_loss() -> None:
    clean = torch.tensor([[[[2.0]]]])
    noise = torch.tensor([[[[-1.0]]]])
    timestep = torch.tensor([0.25])
    noisy = noisy_interpolate(clean, noise, timestep)
    assert noisy.item() == pytest.approx(1.25)
    assert x_to_velocity(noisy, clean, timestep).item() == pytest.approx(-3.0)
    assert target_velocity(clean, noise).item() == pytest.approx(-3.0)
    losses = ddt_losses(
        clean,
        noise,
        noisy,
        timestep,
        clean,
        torch.zeros_like(clean),
        base_loss_weight=0.5,
    )
    assert losses.full_velocity == 0
    assert losses.total == pytest.approx(0.5 * losses.base_velocity.item())


def test_timestep_distribution_is_unshifted_logit_normal() -> None:
    generator = torch.Generator().manual_seed(7)
    values = sample_timesteps(32, generator, "cpu", loc=0.8, scale=0.8)
    assert torch.all((values > 0) & (values < 1))
    assert values.mean() > 0.5


def test_mix_sampling_has_exact_endpoint_masses_and_is_reproducible() -> None:
    first = torch.Generator().manual_seed(17)
    second = torch.Generator().manual_seed(17)
    left = sample_mix_ratios(
        10_000,
        first,
        "cpu",
        rae_endpoint_probability=0.1,
        pixel_endpoint_probability=0.1,
    )
    right = sample_mix_ratios(
        10_000,
        second,
        "cpu",
        rae_endpoint_probability=0.1,
        pixel_endpoint_probability=0.1,
    )
    assert torch.equal(left, right)
    assert (left == 0).float().mean().item() == pytest.approx(0.1, abs=0.015)
    assert (left == 1).float().mean().item() == pytest.approx(0.1, abs=0.015)
    interior = left[(left > 0) & (left < 1)]
    assert interior.mean().item() == pytest.approx(0.5, abs=0.02)


def test_endpoint_normalization_is_not_changed_after_mixing() -> None:
    pixel = torch.full((3, 2, 1, 1), 2.0)
    rae = torch.full_like(pixel, -2.0)
    result = mix_endpoints(pixel, rae, torch.tensor([0.0, 0.25, 1.0]))
    assert torch.equal(result[:, 0, 0, 0], torch.tensor([-2.0, -1.0, 2.0]))


def test_fixed_pixel_control_always_samples_one() -> None:
    ratios = sample_mix_ratios(
        8, torch.Generator().manual_seed(1), "cpu", fixed_ratio=1.0
    )
    assert torch.equal(ratios, torch.ones(8))
