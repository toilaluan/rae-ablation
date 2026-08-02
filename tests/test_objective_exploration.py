from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jit_reg.exploration import select_candidates
from jit_reg.model import DDTOutput
from jit_reg.objective import (
    ddt_losses,
    noisy_interpolate,
    sample_timesteps,
    shift_timestep,
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
    losses = ddt_losses(clean, noise, noisy, timestep, clean, torch.zeros_like(clean), base_loss_weight=0.5)
    assert losses.full_velocity == 0
    assert losses.total == pytest.approx(0.5 * losses.base_velocity.item())


def test_logit_normal_timestep_shift_is_compact_token_shift_four() -> None:
    assert shift_timestep(torch.tensor(0.5), 4).item() == pytest.approx(0.8)
    generator = torch.Generator().manual_seed(7)
    values = sample_timesteps(32, generator, "cpu")
    assert torch.all((values > 0) & (values < 1))


class ScoringModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_classes=3)
        self.grad_enabled: list[bool] = []
        self.base_batches: list[int] = []
        self.full_batches: list[int] = []

    def forward_base_prediction(self, noisy: torch.Tensor, timestep: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        del timestep, labels
        self.grad_enabled.append(torch.is_grad_enabled())
        self.base_batches.append(noisy.shape[0])
        return (noisy - 3).abs()

    def forward_full_prediction(self, noisy: torch.Tensor, timestep: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        del timestep, labels
        self.grad_enabled.append(torch.is_grad_enabled())
        self.full_batches.append(noisy.shape[0])
        return (noisy - 2).abs()

    def forward(self, noisy: torch.Tensor, timestep: torch.Tensor, labels: torch.Tensor) -> DDTOutput:
        return DDTOutput(
            self.forward_full_prediction(noisy, timestep, labels),
            self.forward_base_prediction(noisy, timestep, labels),
        )


def test_hybrid_base_then_full_selection_is_chunk_invariant_and_no_grad() -> None:
    clean = torch.zeros(1, 1, 1, 1)
    noise = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1, 1)
    timestep = torch.ones(1)
    labels = torch.zeros(1, dtype=torch.long)
    model = ScoringModel()
    result = select_candidates(
        model,
        noise,
        noise,
        clean,
        timestep,
        labels,
        k_early=4,
        k_full=2,
        early_candidate_chunk_size=2,
        full_candidate_chunk_size=1,
    )
    unchunked = select_candidates(
        ScoringModel(), noise, noise, clean, timestep, labels, k_early=4, k_full=2
    )
    assert result.indices.item() == 2
    assert torch.equal(result.indices, unchunked.indices)
    assert model.base_batches == [2, 2, 2, 2]
    assert model.full_batches == [1, 1]
    assert model.grad_enabled == [False] * 6


class TiedModel(ScoringModel):
    def forward_full_prediction(self, noisy: torch.Tensor, timestep: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        del timestep, labels
        return torch.zeros_like(noisy)


def test_full_ties_choose_lower_original_candidate_index() -> None:
    values = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1, 1)
    result = select_candidates(
        TiedModel(),
        values,
        values,
        torch.zeros(1, 1, 1, 1),
        torch.ones(1),
        torch.zeros(1, dtype=torch.long),
        k_early=4,
        k_full=2,
    )
    assert result.indices.item() == 2
