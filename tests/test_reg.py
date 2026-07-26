from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from jit_reg.model.sit import SiTOutput
from jit_reg.objective import (
    jit_losses,
    noisy_interpolate,
    sample_reg_timesteps,
    shift_timestep,
)
from jit_reg.reg import (
    FeatureTargets,
    negative_cosine_repa_loss,
    prepare_feature_targets,
    repa_target_for_scope,
    scale_reg_targets,
)
from jit_reg.sampling import ODESample, ode_sample


def test_optional_component_target_extraction() -> None:
    spatial = torch.randn(2, 16, 8)
    baseline = prepare_feature_targets(
        spatial, 2, include_reg=False, include_repa=False
    )
    assert baseline == FeatureTargets(None, None)

    reg = prepare_feature_targets(
        spatial,
        2,
        include_reg=True,
        pool_size=2,
        include_repa=False,
    )
    assert reg.reg is not None and reg.reg.shape == (2, 4, 8)
    assert reg.repa is None
    assert torch.allclose(reg.reg.mean(dim=-1), torch.zeros(2, 4), atol=1e-6)
    assert torch.allclose(
        reg.reg.std(dim=-1, unbiased=False), torch.ones(2, 4), atol=2e-4
    )

    repa = prepare_feature_targets(
        spatial, 2, include_reg=False, include_repa=True
    )
    assert repa.reg is None
    assert repa.repa is not None and repa.repa.shape == (2, 4, 8)
    assert torch.allclose(repa.repa.norm(dim=-1), torch.ones(2, 4))


def test_training_reg_scale_preserves_repa_and_reaches_requested_std() -> None:
    canonical = FeatureTargets(
        F.layer_norm(torch.randn(2, 4, 8), (8,)),
        F.normalize(torch.randn(2, 4, 8), dim=-1),
    )
    scaled = scale_reg_targets(canonical, 0.53)
    assert scaled.reg is not None
    assert scaled.reg.std(dim=-1, unbiased=False).mean() == pytest.approx(
        0.53, abs=2e-4
    )
    assert scaled.repa is canonical.repa


def test_repa_target_scopes() -> None:
    targets = FeatureTargets(torch.randn(2, 1, 8), torch.randn(2, 4, 8))
    assert repa_target_for_scope(targets, "image").shape == (2, 4, 8)
    full = repa_target_for_scope(targets, "image_and_reg")
    assert full is not None and full.shape == (2, 5, 8)
    assert torch.equal(full[:, :1], targets.reg)


def test_negative_cosine_repa_loss() -> None:
    target = F.normalize(torch.randn(2, 4, 8), dim=-1)
    assert negative_cosine_repa_loss(target, target) == pytest.approx(-1.0)


def test_log_uniform_reg_timing_is_deterministic_and_equal_time_is_exact() -> None:
    image_time = torch.linspace(0.1, 0.9, 8)
    first = sample_reg_timesteps(
        image_time, 0.5, 2.0, torch.Generator().manual_seed(7)
    )
    second = sample_reg_timesteps(
        image_time, 0.5, 2.0, torch.Generator().manual_seed(7)
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.all((first[1] >= 0.5) & (first[1] <= 2.0))
    assert torch.allclose(first[0], shift_timestep(image_time, first[1]))

    equal_time, ratio = sample_reg_timesteps(
        image_time, 1.0, 1.0, torch.Generator().manual_seed(1)
    )
    assert torch.equal(equal_time, image_time)
    assert torch.equal(ratio, torch.ones_like(image_time))


def test_reg_velocity_loss_uses_reg_timestep() -> None:
    clean_image = torch.zeros(1, 1)
    image_noise = torch.zeros_like(clean_image)
    image_time = torch.tensor([0.5])
    clean_reg = torch.ones(1, 1)
    reg_noise = torch.zeros_like(clean_reg)
    reg_prediction = torch.zeros_like(clean_reg)
    reg_time = torch.tensor([0.25])
    losses = jit_losses(
        clean_image,
        image_noise,
        image_time,
        clean_image,
        clean_reg=clean_reg,
        reg_noise=reg_noise,
        reg_prediction=reg_prediction,
        reg_timestep=reg_time,
        reg_weight=1,
        repa_weight=0,
    )
    assert losses.image_velocity == 0
    assert losses.reg_velocity == pytest.approx((1 / 0.75) ** 2)


def test_reg_metrics_are_generic() -> None:
    clean_reg = torch.tensor([[[1.0], [2.0], [1.0]]])
    prediction = torch.zeros_like(clean_reg)
    losses = jit_losses(
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.tensor([0.5]),
        torch.zeros(1, 1),
        clean_reg=clean_reg,
        reg_noise=torch.zeros_like(clean_reg),
        reg_prediction=prediction,
        reg_timestep=torch.tensor([0.5]),
    )
    assert losses.reg_x0_mse == pytest.approx(2.0)
    assert losses.reg_velocity == pytest.approx(8.0)
    assert losses.reg_x0_cosine == 0
    assert not any("cls" in field or "spatial" in field for field in losses._fields)


class RecordingCleanModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_classes=3)
        self.calls: list[tuple[torch.Tensor, torch.Tensor | None, int]] = []

    def forward(
        self,
        image: torch.Tensor,
        timestep: torch.Tensor,
        labels: torch.Tensor,
        *,
        noisy_reg: torch.Tensor | None = None,
        reg_timestep: torch.Tensor | None = None,
        return_repa: bool = False,
    ) -> SiTOutput:
        del labels, return_repa
        self.calls.append(
            (
                timestep.detach().cpu(),
                reg_timestep.detach().cpu() if reg_timestep is not None else None,
                image.shape[0],
            )
        )
        return SiTOutput(
            image=torch.full_like(image, 0.25),
            reg=torch.full_like(noisy_reg, -0.5) if noisy_reg is not None else None,
        )


@pytest.mark.parametrize("solver", ["euler", "heun"])
def test_sampler_jointly_integrates_dual_schedules_and_x0(solver: str) -> None:
    model = RecordingCleanModel()
    image_noise = torch.randn(1, 1, 2, 2)
    reg_noise = torch.randn(1, 1, 3)
    output = ode_sample(
        model,
        image_noise,
        torch.tensor([1]),
        reg_noise=reg_noise,
        steps=2,
        solver=solver,
        guidance_scale=1,
        timestep_shift=1,
        reg_timestep_ratio=0.5,
        cfg_interval=(1, 1),
    )
    assert isinstance(output, ODESample)
    assert torch.allclose(output.image, torch.full_like(output.image, 0.25))
    assert torch.allclose(output.reg, torch.full_like(output.reg, -0.5))
    assert len(model.calls) == (2 if solver == "euler" else 3)
    image_half = [call for call in model.calls if call[0][0] == pytest.approx(0.5)]
    assert image_half
    assert image_half[0][1] is not None
    assert image_half[0][1][0] == pytest.approx(1 / 3)


def test_cfg_window_uses_image_time_with_heterogeneous_reg_time() -> None:
    model = RecordingCleanModel()
    ode_sample(
        model,
        torch.randn(1, 1, 2, 2),
        torch.tensor([1]),
        reg_noise=torch.randn(1, 1, 3),
        steps=2,
        guidance_scale=2,
        reg_timestep_ratio=0.1,
        cfg_interval=(0.4, 0.6),
    )
    assert [call[2] for call in model.calls] == [1, 2]


def test_noisy_interpolation_uses_its_own_timestep() -> None:
    clean = torch.ones(2, 1)
    noise = torch.zeros_like(clean)
    assert torch.allclose(
        noisy_interpolate(clean, noise, torch.tensor([0.25, 0.75])),
        torch.tensor([[0.25], [0.75]]),
    )
