from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cascade_dit.config import ObjectiveConfig
from cascade_dit.objective import (
    clean_prediction_loss,
    gaussian_flow_path,
    sample_timesteps,
    shift_timestep,
    x0_to_velocity,
)
from cascade_dit.sample_runtime import noise_for_seeds
from cascade_dit.sampling import sample_tokens


def test_clean_x0_is_converted_to_velocity_with_denominator_floor() -> None:
    clean = torch.tensor([[[2.0]]])
    noise = torch.tensor([[[-1.0]]])
    path = gaussian_flow_path(clean, noise, torch.tensor([0.25]))
    assert path.noisy.item() == pytest.approx(1.25)
    assert path.velocity.item() == pytest.approx(-3)
    assert x0_to_velocity(path.noisy, clean, path.timestep).item() == pytest.approx(-3)
    loss = clean_prediction_loss(path, clean, ObjectiveConfig())
    assert loss.total.item() == pytest.approx(0)
    near_zero = x0_to_velocity(torch.ones_like(clean), torch.zeros_like(clean), torch.tensor([0.0]))
    assert near_zero.item() == pytest.approx(20)


def test_shifted_logit_normal_timesteps_are_seeded_and_bounded() -> None:
    first = sample_timesteps(64, torch.Generator().manual_seed(4), "cpu")
    second = sample_timesteps(64, torch.Generator().manual_seed(4), "cpu")
    assert torch.equal(first, second)
    assert torch.all((first > 0) & (first < 1))
    assert shift_timestep(torch.tensor(0.5), 4).item() == pytest.approx(0.8)


class CleanOracle(nn.Module):
    def __init__(self, clean: torch.Tensor) -> None:
        super().__init__()
        self.clean = clean
        self.batch_sizes: list[int] = []

    def forward(
        self,
        values: torch.Tensor,
        timestep: torch.Tensor,
        labels: torch.Tensor,
        representation_weights: torch.Tensor,
    ) -> SimpleNamespace:
        del timestep, labels
        assert torch.all(representation_weights[:, -1] == 1)
        self.batch_sizes.append(values.shape[0])
        return SimpleNamespace(prediction=self.clean.expand_as(values))


def test_one_step_x0_sampling_recovers_clean_and_cfg_is_interval_gated() -> None:
    clean = torch.full((1, 3, 4), 2.0)
    noise = torch.randn_like(clean)
    model = CleanOracle(clean)
    weights = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    sampled = sample_tokens(
        model, noise, torch.tensor([2]), weights, steps=1, num_classes=10
    )
    assert torch.allclose(sampled, clean)
    sample_tokens(
        model,
        noise,
        torch.tensor([2]),
        weights,
        steps=2,
        guidance_scale=2,
        guidance_start=0,
        guidance_end=0.25,
        num_classes=10,
    )
    assert 2 in model.batch_sizes
    assert 1 in model.batch_sizes


def test_noise_is_batch_invariant_per_sample_seed() -> None:
    seeds = torch.tensor([12, 33])
    together = noise_for_seeds(seeds, (2, 3), "cpu")
    apart = torch.cat([noise_for_seeds(seed[None], (2, 3), "cpu") for seed in seeds])
    assert torch.equal(together, apart)
