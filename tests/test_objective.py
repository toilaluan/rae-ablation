from __future__ import annotations

import pytest
import torch

from jit_reg.objective import (
    flow_losses,
    flow_path,
    sample_timesteps,
    shift_timestep,
    x_to_velocity,
)


def test_gaussian_flow_path_connects_endpoint_to_noise() -> None:
    endpoint = torch.tensor([[[[2.0]]], [[[10.0]]]])
    noise = torch.tensor([[[[6.0]]], [[[14.0]]]])
    path = flow_path(endpoint, noise, torch.tensor([0.0, 1.0]))
    assert torch.equal(path.noisy, torch.tensor([[[[2.0]]], [[[14.0]]]]))
    assert torch.equal(path.clean, endpoint)
    assert torch.equal(path.noise, noise)

    midpoint = flow_path(endpoint, noise, torch.full((2,), 0.5))
    assert torch.equal(midpoint.noisy, 0.5 * (endpoint + noise))


def test_clean_endpoint_loss_and_velocity_recovery() -> None:
    prediction = torch.tensor([[[[1.0]]]])
    target = torch.tensor([[[[3.0]]]])
    losses = flow_losses(prediction, target)
    assert losses.total.item() == pytest.approx(4.0)
    assert losses.endpoint.item() == pytest.approx(4.0)
    velocity = x_to_velocity(
        torch.tensor([[[[1.25]]]]),
        torch.tensor([[[[2.0]]]]),
        torch.tensor([0.25]),
    )
    assert velocity.item() == pytest.approx(-3.0)


def test_timestep_sampling_and_shift() -> None:
    assert shift_timestep(torch.tensor(0.5), 4).item() == pytest.approx(0.8)
    values = sample_timesteps(32, torch.Generator().manual_seed(7), "cpu")
    assert torch.all((values > 0) & (values < 1))
    assert torch.equal(
        values,
        sample_timesteps(32, torch.Generator().manual_seed(7), "cpu"),
    )


def test_flow_rejects_unpaired_shapes_and_zero_velocity_timestep() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        flow_path(torch.zeros(1, 2, 1, 1), torch.zeros(1, 3, 1, 1), torch.zeros(1))
    with pytest.raises(ValueError, match="strictly positive"):
        x_to_velocity(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1), torch.zeros(1))
    with pytest.raises(ValueError, match="prediction and target"):
        flow_losses(torch.zeros(1, 2), torch.zeros(1, 3))
