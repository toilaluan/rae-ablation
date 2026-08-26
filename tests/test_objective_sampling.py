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
from cascade_dit.sampling import sample_semantic_then_pixel, sample_tokens


def test_ratio_mixed_x0_is_converted_to_velocity_after_interpolation() -> None:
    semantic = torch.full((2, 1, 1), 2.0)
    pixel = torch.full((2, 1, 1), 6.0)
    ratio = torch.tensor([0.25, 0.75])
    clean = torch.lerp(semantic, pixel, ratio[:, None, None])
    path = gaussian_flow_path(clean, torch.full_like(clean, -1), torch.tensor([0.25, 0.5]))
    output = DDTOutput(clean, semantic_x0=semantic, pixel_x0=pixel)
    loss = flow_matching_loss(path, output, pixel, ratio, ObjectiveConfig())
    assert loss.velocity_mse.item() == pytest.approx(0)
    assert loss.pixel_auxiliary_mse.item() == pytest.approx(0)


def test_auxiliary_mse_masks_only_pure_semantic_items_and_applies_weight() -> None:
    clean = torch.zeros(3, 1, 1)
    path = gaussian_flow_path(clean, torch.ones_like(clean), torch.full((3,), 0.5))
    ratio = torch.tensor([0.0, 0.5, 1.0])
    pixel_target = torch.tensor([[[2.0]], [[100.0]], [[100.0]]])
    pixel_prediction = torch.tensor([[[0.0]], [[0.0]], [[0.0]]])
    config = ObjectiveConfig(pixel_auxiliary_mse_weight=3.0)
    loss = flow_matching_loss(
        path,
        DDTOutput(clean, pixel_x0=pixel_prediction),
        pixel_target,
        ratio,
        config,
    )
    assert loss.pixel_auxiliary_mse.item() == pytest.approx(4)
    assert loss.weighted_pixel_auxiliary_mse.item() == pytest.approx(12)
    assert loss.total.item() == pytest.approx(12)
    assert loss.pure_semantic_count == 1


def test_auxiliary_empty_mask_is_safe() -> None:
    clean = torch.zeros(2, 1, 1)
    path = gaussian_flow_path(clean, torch.ones_like(clean), torch.full((2,), 0.5))
    loss = flow_matching_loss(
        path,
        DDTOutput(clean, pixel_x0=torch.ones_like(clean)),
        torch.zeros_like(clean),
        torch.tensor([0.2, 1.0]),
        ObjectiveConfig(),
    )
    assert loss.pixel_auxiliary_mse.item() == 0
    assert loss.pure_semantic_count == 0


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


class EndpointOracle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        values: torch.Tensor,
        timestep: torch.Tensor,
        _labels: torch.Tensor,
        pixel_ratio: torch.Tensor,
        *,
        prediction_mode: str,
    ) -> DDTOutput:
        self.calls.append(
            (prediction_mode, timestep.detach().clone(), pixel_ratio.detach().clone())
        )
        prediction = torch.full_like(values, 2 if prediction_mode == "semantic" else 6)
        return DDTOutput(prediction)


def test_semantic_sampling_has_one_final_t0_ratio_zero_pixel_decode() -> None:
    model = EndpointOracle()
    noise = torch.randn(2, 3, 4)
    sampled = sample_semantic_then_pixel(
        model, noise, torch.tensor([1, 2]), steps=3, num_classes=10
    )
    semantic_calls = [call for call in model.calls if call[0] == "semantic"]
    pixel_calls = [call for call in model.calls if call[0] == "pixel"]
    assert len(semantic_calls) == 3
    assert len(pixel_calls) == 1
    assert torch.equal(pixel_calls[0][1], torch.zeros(2))
    assert torch.equal(pixel_calls[0][2], torch.zeros(2))
    assert torch.all(sampled == 6)


def test_pixel_sampling_uses_ratio_one_and_noise_is_batch_invariant() -> None:
    model = EndpointOracle()
    noise = torch.randn(1, 3, 4)
    ratio = torch.ones(1)
    sampled = sample_tokens(
        model,
        noise,
        torch.tensor([2]),
        ratio,
        prediction_mode="pixel",
        steps=1,
        num_classes=10,
    )
    assert torch.allclose(sampled, torch.full_like(noise, 6))
    assert torch.equal(model.calls[0][2], ratio)

    seeds = torch.tensor([12, 33])
    together = noise_for_seeds(seeds, (2, 3), "cpu")
    apart = torch.cat([noise_for_seeds(seed[None], (2, 3), "cpu") for seed in seeds])
    assert torch.equal(together, apart)
