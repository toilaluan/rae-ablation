from __future__ import annotations

import pytest
import torch
from torch import nn

from cascade_dit.config import ObjectiveConfig
from cascade_dit.model import DDTOutput
from cascade_dit.objective import (
    flow_matching_loss,
    gaussian_flow_path,
    sample_timesteps,
    shift_timestep,
    x0_to_velocity,
)
from cascade_dit.sample_runtime import noise_for_seeds
from cascade_dit.sampling import (
    internal_guidance_x0,
    sample_semantic_then_pixel,
    sample_tokens,
)


class LPIPSStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.inputs = (prediction.detach(), target.detach())
        return (prediction - target).abs().mean((1, 2, 3), keepdim=True)


def test_joint_loss_has_four_terms_and_lpips_receives_minus_one_to_one() -> None:
    clean = torch.zeros(1, 2, 3)
    path = gaussian_flow_path(clean, torch.ones_like(clean), torch.tensor([0.5]))
    pixels = torch.full((1, 3, 4, 4), 0.25)
    target = torch.full_like(pixels, 0.75)
    stub = LPIPSStub()
    config = ObjectiveConfig(
        base_velocity_weight=2,
        pixel_l1_weight=3,
        pixel_lpips_weight=4,
    )
    loss = flow_matching_loss(
        path,
        DDTOutput(clean, torch.full_like(clean, 0.25), pixels),
        target,
        stub,
        config,
    )
    assert loss.full_velocity_mse.item() == pytest.approx(0)
    assert loss.base_velocity_mse.item() == pytest.approx(0.25)
    assert loss.pixel_l1.item() == pytest.approx(0.5)
    assert loss.pixel_lpips.item() == pytest.approx(1)
    assert loss.total.item() == pytest.approx(6)
    assert stub.inputs is not None
    prediction_input, target_input = stub.inputs
    assert torch.all(prediction_input == -0.5)
    assert torch.all(target_input == 0.5)


def test_shifted_logit_normal_timesteps_are_seeded_and_bounded() -> None:
    first = sample_timesteps(64, torch.Generator().manual_seed(4), "cpu")
    second = sample_timesteps(64, torch.Generator().manual_seed(4), "cpu")
    assert torch.equal(first, second)
    assert torch.all((first > 0) & (first < 1))
    assert shift_timestep(torch.tensor(0.5), 4).item() == pytest.approx(0.8)
    near_zero = x0_to_velocity(
        torch.ones(1, 1, 1), torch.zeros(1, 1, 1), torch.tensor([0.0])
    )
    assert near_zero.item() == pytest.approx(20)


def test_internal_guidance_interpolation_and_inclusive_boundaries() -> None:
    full = torch.full((4, 1, 1), 2.0)
    base = torch.full_like(full, 1.0)
    times = torch.tensor([0.099, 0.1, 1.0, 0.0])
    guided = internal_guidance_x0(
        full, base, times, scale=1.78, t_min=0.1, t_max=1
    )
    assert guided[:, 0, 0].tolist() == pytest.approx([2, 2.78, 2.78, 2])


class JointOracle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def forward(
        self,
        values: torch.Tensor,
        timestep: torch.Tensor,
        _labels: torch.Tensor,
        *,
        decode_pixels: bool,
    ) -> DDTOutput:
        self.calls.append(
            (values.detach().clone(), timestep.detach().clone(), decode_pixels)
        )
        full = torch.full_like(values, 2)
        base = torch.full_like(values, 1)
        rgb = (
            torch.full((values.shape[0], 3, 128, 128), 6.0)
            if decode_pixels
            else None
        )
        return DDTOutput(full, base, rgb)


def test_sampling_uses_one_conditional_call_per_ode_call_and_one_t0_decode() -> None:
    model = JointOracle()
    noise = torch.zeros(2, 3, 4)
    sampled = sample_semantic_then_pixel(
        model,
        noise,
        torch.tensor([1, 2]),
        steps=3,
        internal_guidance_scale=1.78,
    )
    ode_calls = [call for call in model.calls if not call[2]]
    pixel_calls = [call for call in model.calls if call[2]]
    assert len(ode_calls) == 3
    assert all(values.shape[0] == 2 for values, _time, _decode in ode_calls)
    assert len(pixel_calls) == 1
    assert torch.equal(pixel_calls[0][1], torch.zeros(2))
    assert torch.all(sampled == 6)


def test_one_step_sampling_applies_internal_guidance_and_noise_is_batch_invariant() -> None:
    model = JointOracle()
    sampled = sample_tokens(
        model,
        torch.zeros(1, 3, 4),
        torch.tensor([2]),
        steps=1,
        internal_guidance_scale=2,
    )
    assert torch.allclose(sampled, torch.full_like(sampled, 3))
    assert len(model.calls) == 1

    seeds = torch.tensor([12, 33])
    together = noise_for_seeds(seeds, (2, 3), "cpu")
    apart = torch.cat([noise_for_seeds(seed[None], (2, 3), "cpu") for seed in seeds])
    assert torch.equal(together, apart)
