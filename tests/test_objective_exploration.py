from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jit_reg.exploration import select_candidates
from jit_reg.model.sit import SiTOutput
from jit_reg.objective import (
    noisy_interpolate,
    per_sample_repa_cosine_loss,
    per_sample_velocity_mse,
    sit_losses,
    target_velocity,
    x_to_velocity,
)


def test_x_prediction_is_converted_to_velocity_only_for_loss() -> None:
    clean = torch.tensor([[[[2.0]]]])
    noise = torch.tensor([[[[-1.0]]]])
    timestep = torch.tensor([0.25])
    noisy = noisy_interpolate(clean, noise, timestep)
    assert noisy.item() == pytest.approx(-0.25)
    assert x_to_velocity(noisy, clean, timestep).item() == pytest.approx(3.0)
    assert target_velocity(clean, noise).item() == pytest.approx(3.0)
    losses = sit_losses(clean, noise, noisy, timestep, clean)
    assert losses.velocity == 0


class IdentityXModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_classes=3)
        self.grad_enabled: list[bool] = []
        self.batch_sizes: list[int] = []

    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        labels: torch.Tensor,
        *,
        return_repa: bool = False,
    ) -> SiTOutput:
        del timestep, labels, return_repa
        self.grad_enabled.append(torch.is_grad_enabled())
        self.batch_sizes.append(noisy.shape[0])
        return SiTOutput(noisy)


def test_final_selector_is_per_sample_no_grad_and_chunk_invariant() -> None:
    clean = torch.zeros(2, 1, 1, 1)
    noise = torch.tensor(
        [
            [[[[3.0]]], [[[0.2]]], [[[1.0]]], [[[2.0]]]],
            [[[[0.1]]], [[[4.0]]], [[[2.0]]], [[[1.0]]]],
        ]
    )
    timestep = torch.full((2,), 0.5)
    noisy = 0.5 * noise
    labels = torch.tensor([0, 1])
    full_model = IdentityXModel()
    full = select_candidates(
        full_model,
        noisy,
        noise,
        clean,
        timestep,
        labels,
        selector="final_velocity",
    )
    chunked = select_candidates(
        IdentityXModel(),
        noisy,
        noise,
        clean,
        timestep,
        labels,
        selector="final_velocity",
        candidate_chunk_size=2,
    )
    assert torch.equal(full.indices, torch.tensor([1, 0]))
    assert torch.equal(full.indices, chunked.indices)
    assert torch.equal(full.noise, chunked.noise)
    assert full_model.grad_enabled == [False]
    assert full_model.batch_sizes == [8]
    assert torch.all(full.selected_score <= full.candidate_zero_score)


class REPAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prefix_batches: list[int] = []

    def forward_repa(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        del timestep, labels
        self.prefix_batches.append(noisy.shape[0])
        value = noisy.flatten(1)[:, :1]
        return torch.stack((value, torch.ones_like(value)), dim=-1)


def test_repa_selector_uses_early_cosine_target() -> None:
    clean = torch.zeros(1, 1, 1, 1)
    noise = torch.tensor([[[[[0.0]]], [[[1.0]]], [[[3.0]]], [[[-2.0]]]]])
    noisy = noise.clone()
    target = torch.tensor([[[3.0, 1.0]]])
    model = REPAModel()
    result = select_candidates(
        model,
        noisy,
        noise,
        clean,
        torch.tensor([0.0]),
        torch.tensor([1]),
        selector="repa",
        repa_target=target,
        candidate_chunk_size=2,
    )
    assert result.indices.item() == 2
    assert model.prefix_batches == [2, 2]


def test_velocity_and_repa_scores_reduce_per_sample() -> None:
    clean = torch.zeros(2, 1, 1, 1)
    noise = torch.tensor([[[[1.0]]], [[[2.0]]]])
    noisy = noise * 0.5
    score = per_sample_velocity_mse(
        clean, noise, noisy, torch.full((2,), 0.5), noisy
    )
    assert torch.equal(score, torch.tensor([1.0, 4.0]))
    target = torch.tensor([[[1.0, 0.0]]])
    assert per_sample_repa_cosine_loss(target, target).item() == pytest.approx(0)
