from __future__ import annotations

import pytest
import torch

from jit_reg.config import CascadeConfig
from jit_reg.objective import (
    SEMANTIC_STAGE,
    TARGET_STAGE,
    cascade_path,
    diffusion_losses,
    pool_semantic,
    sample_stages,
    sample_timesteps,
    shift_timestep,
    x_to_velocity,
)


def test_semantic_pooling_is_four_by_four_then_restored() -> None:
    semantic = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    restored = pool_semantic(semantic, CascadeConfig())
    pooled = torch.nn.functional.interpolate(semantic, size=(4, 4), mode="area")
    expected = torch.nn.functional.interpolate(
        pooled, size=(8, 8), mode="bilinear", align_corners=False
    )
    assert restored.shape == semantic.shape
    assert torch.equal(restored, expected)


def test_cascade_path_has_clean_semantic_interface_and_local_velocities() -> None:
    config = CascadeConfig(pool_size=2)
    target = torch.full((2, 1, 2, 2), 10.0)
    semantic = torch.tensor(
        [[[[0.0, 2.0], [4.0, 6.0]]], [[[2.0, 4.0], [6.0, 8.0]]]]
    )
    noise = torch.full_like(target, -2.0)
    stages = torch.tensor([SEMANTIC_STAGE, TARGET_STAGE])
    path = cascade_path(
        target,
        semantic,
        noise,
        torch.tensor([0.0, 1.0]),
        stages,
        config,
    )
    # Semantic t=0 and target t=1 meet at the same clean semantic endpoint.
    assert torch.equal(path.noisy, path.semantic_endpoint)
    assert path.timestep.tolist() == [0.0, 1.0]
    assert torch.equal(path.clean[0], path.semantic_endpoint[0])
    assert torch.equal(path.clean[1], target[1])
    assert torch.equal(path.velocity[0], noise[0] - path.semantic_endpoint[0])
    assert torch.equal(path.velocity[1], path.semantic_endpoint[1] - target[1])

    endpoints = cascade_path(
        target,
        semantic,
        noise,
        torch.tensor([1.0, 0.0]),
        stages,
        config,
    )
    assert torch.equal(endpoints.noisy[0], noise[0])
    assert torch.equal(endpoints.noisy[1], target[1])


def test_x_prediction_velocity_and_weighted_losses() -> None:
    noisy = torch.tensor([[[[1.25]]]])
    clean = torch.tensor([[[[2.0]]]])
    target_velocity = torch.tensor([[[[-3.0]]]])
    timestep = torch.tensor([0.25])
    assert x_to_velocity(noisy, clean, timestep).item() == pytest.approx(-3.0)
    semantic = torch.ones_like(noisy)
    projection = semantic.flatten(2).transpose(1, 2)
    losses = diffusion_losses(
        noisy,
        target_velocity,
        timestep,
        torch.tensor([SEMANTIC_STAGE]),
        clean,
        torch.zeros_like(clean),
        projection,
        semantic,
        base_loss_weight=0.5,
        repa_coefficient=0.5,
    )
    assert losses.full_velocity.item() == 0
    assert losses.repa.item() == pytest.approx(-1.0)
    assert losses.total.item() == pytest.approx(
        0.5 * losses.base_semantic_velocity.item() - 0.5
    )


def test_base_loss_uses_only_semantic_stage_examples() -> None:
    noisy = torch.ones(2, 1, 1, 1)
    velocity_target = torch.zeros_like(noisy)
    timestep = torch.ones(2)
    stages = torch.tensor([SEMANTIC_STAGE, TARGET_STAGE])
    full_prediction = noisy.clone()
    base_prediction = torch.zeros_like(noisy, requires_grad=True)
    semantic = torch.ones_like(noisy)
    projection = semantic.flatten(2).transpose(1, 2)
    losses = diffusion_losses(
        noisy,
        velocity_target,
        timestep,
        stages,
        full_prediction,
        base_prediction,
        projection,
        semantic,
        repa_coefficient=0,
    )
    assert losses.base_semantic_velocity.item() == pytest.approx(1.0)
    losses.total.backward()
    assert base_prediction.grad is not None
    assert base_prediction.grad[0].abs().sum() > 0
    assert base_prediction.grad[1].abs().sum() == 0


def test_base_loss_is_safe_when_batch_has_no_semantic_examples() -> None:
    noisy = torch.ones(2, 1, 1, 1)
    base_prediction = torch.zeros_like(noisy, requires_grad=True)
    semantic = torch.ones_like(noisy)
    losses = diffusion_losses(
        noisy,
        torch.zeros_like(noisy),
        torch.ones(2),
        torch.full((2,), TARGET_STAGE),
        noisy.clone(),
        base_prediction,
        semantic.flatten(2).transpose(1, 2),
        semantic,
        repa_coefficient=0,
    )
    assert losses.base_semantic_velocity.item() == 0
    losses.total.backward()
    assert base_prediction.grad is not None
    assert torch.count_nonzero(base_prediction.grad) == 0


def test_disabled_repa_returns_zero_metrics_without_projection() -> None:
    noisy = torch.ones(1, 1, 1, 1)
    losses = diffusion_losses(
        noisy,
        torch.zeros_like(noisy),
        torch.ones(1),
        torch.tensor([SEMANTIC_STAGE]),
        noisy,
        noisy,
        None,
        noisy,
        repa_coefficient=0,
    )
    assert losses.repa.item() == 0
    assert losses.cosine_similarity.item() == 0
    with pytest.raises(ValueError, match="projector is disabled"):
        diffusion_losses(
            noisy,
            torch.zeros_like(noisy),
            torch.ones(1),
            torch.tensor([SEMANTIC_STAGE]),
            noisy,
            noisy,
            None,
            noisy,
            repa_coefficient=0.5,
        )


def test_stage_sampling_and_logit_normal_timestep_shift() -> None:
    stages = sample_stages(
        128, torch.Generator().manual_seed(7), "cpu", semantic_probability=0.5
    )
    assert set(stages.tolist()) == {SEMANTIC_STAGE, TARGET_STAGE}
    assert stages.dtype == torch.long
    assert shift_timestep(torch.tensor(0.5), 4).item() == pytest.approx(0.8)
    values = sample_timesteps(32, torch.Generator().manual_seed(7), "cpu")
    assert torch.all((values > 0) & (values < 1))
