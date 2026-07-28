from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from jit_reg.model.ddt import DDTOutput, DDTREGOutput
from jit_reg.objective import (
    denoising_step_bounds,
    jit_losses,
    local_denoising_schedule,
    noisy_interpolate,
    sample_timesteps,
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
    spatial = torch.randn(
        2,
        16,
        8,
        generator=torch.Generator().manual_seed(0),
    )
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


def test_independent_reg_timing_is_deterministic() -> None:
    image_time = sample_timesteps(
        8, torch.Generator().manual_seed(7), "cpu", loc=-0.8, scale=0.8
    )
    first = sample_timesteps(
        8, torch.Generator().manual_seed(11), "cpu", loc=-0.8, scale=0.8
    )
    second = sample_timesteps(
        8, torch.Generator().manual_seed(11), "cpu", loc=-0.8, scale=0.8
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, image_time)


def test_denoising_intervals_quantize_and_freeze_local_clocks() -> None:
    assert denoising_step_bounds((0.0, 0.7), 50) == (0, 35)
    assert denoising_step_bounds((0.3, 1.0), 50) == (15, 50)
    assert denoising_step_bounds((0.25, 0.75), 2) == (1, 2)
    reg = local_denoising_schedule((0.0, 0.7), 10, device="cpu")
    image = local_denoising_schedule((0.3, 1.0), 10, device="cpu")
    assert torch.equal(reg[:1], torch.tensor([0.0]))
    assert torch.all(reg[7:] == 1)
    assert torch.all(image[:4] == 0)
    assert image[-1] == 1
    with pytest.raises(ValueError, match="collapses"):
        denoising_step_bounds((0.01, 0.02), 10)


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
        reg_early_prediction=reg_prediction,
        reg_timestep=reg_time,
        reg_weight=1,
        repa_weight=0,
    )
    assert losses.image_velocity == 0
    assert losses.reg_velocity == pytest.approx((1 / 0.75) ** 2)
    assert losses.reg_early_velocity == pytest.approx((1 / 0.75) ** 2)
    assert losses.total == pytest.approx(2 * (1 / 0.75) ** 2)


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
        reg_early_prediction=prediction,
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
    ) -> DDTOutput:
        del labels, return_repa
        self.calls.append(
            (
                timestep.detach().cpu(),
                reg_timestep.detach().cpu() if reg_timestep is not None else None,
                image.shape[0],
            )
        )
        return DDTOutput(
            image=torch.full_like(image, 0.25),
            reg=torch.full_like(noisy_reg, -0.5) if noisy_reg is not None else None,
            reg_early=(
                torch.full_like(noisy_reg, -0.5)
                if noisy_reg is not None
                else None
            ),
        )


class RecordingIndependentModel(RecordingCleanModel):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_classes=3,
            features=SimpleNamespace(
                reg=SimpleNamespace(attention_mode="asymmetric")
            ),
        )
        self.reg_only_calls: list[
            tuple[torch.Tensor, int]
        ] = []

    def forward_reg(
        self,
        noisy_reg: torch.Tensor,
        labels: torch.Tensor,
        *,
        reg_timestep: torch.Tensor,
    ) -> DDTREGOutput:
        del labels
        self.reg_only_calls.append(
            (
                reg_timestep.detach().cpu(),
                noisy_reg.shape[0],
            )
        )
        prediction = torch.full_like(noisy_reg, -0.5)
        return DDTREGOutput(prediction, prediction)


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
        reg_timestep_shift=0.5,
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
        reg_timestep_shift=0.1,
        cfg_interval=(0.4, 0.6),
    )
    assert [call[2] for call in model.calls] == [1, 2]


def test_sampler_runs_asynchronous_reg_lead_and_image_follow() -> None:
    model = RecordingCleanModel()
    output = ode_sample(
        model,
        torch.randn(1, 1, 2, 2),
        torch.tensor([1]),
        reg_noise=torch.randn(1, 1, 3),
        steps=10,
        guidance_scale=1,
        img_denoising_interval=(0.3, 1.0),
        reg_denoising_interval=(0.0, 0.7),
        cfg_interval=(1, 1),
    )
    assert isinstance(output, ODESample)
    assert len(model.calls) == 10
    assert all(call[0][0] == 0 for call in model.calls[:3])
    assert all(call[1] is not None and call[1][0] == 1 for call in model.calls[7:])
    assert torch.allclose(output.image, torch.full_like(output.image, 0.25))
    assert torch.allclose(output.reg, torch.full_like(output.reg, -0.5))


@pytest.mark.parametrize(
    ("solver", "expected_reg_only_calls"),
    [("euler", 3), ("heun", 6)],
)
def test_independent_sampler_uses_reg_only_forward_while_image_is_frozen(
    solver: str,
    expected_reg_only_calls: int,
) -> None:
    model = RecordingIndependentModel()
    image_noise = torch.randn(1, 1, 2, 2)
    reg_noise = torch.randn(1, 1, 3)
    labels = torch.tensor([1])
    output = ode_sample(
        model,
        image_noise,
        labels,
        reg_noise=reg_noise,
        steps=10,
        solver=solver,
        guidance_scale=1,
        img_denoising_interval=(0.3, 1.0),
        reg_denoising_interval=(0.0, 0.7),
        cfg_interval=(0, 1),
    )
    fallback = RecordingIndependentModel()
    fallback.config.features.reg.attention_mode = "joint"
    fallback_output = ode_sample(
        fallback,
        image_noise,
        labels,
        reg_noise=reg_noise,
        steps=10,
        solver=solver,
        guidance_scale=1,
        img_denoising_interval=(0.3, 1.0),
        reg_denoising_interval=(0.0, 0.7),
        cfg_interval=(0, 1),
    )
    assert isinstance(output, ODESample)
    assert isinstance(fallback_output, ODESample)
    assert len(model.reg_only_calls) == expected_reg_only_calls
    assert all(call[1] == 1 for call in model.reg_only_calls)
    assert model.calls
    assert torch.allclose(output.image, fallback_output.image)
    assert torch.allclose(output.reg, fallback_output.reg)


def test_independent_sampler_skips_steps_when_both_states_are_frozen() -> None:
    model = RecordingIndependentModel()
    output = ode_sample(
        model,
        torch.randn(1, 1, 2, 2),
        torch.tensor([1]),
        reg_noise=torch.randn(1, 1, 3),
        steps=10,
        solver="euler",
        guidance_scale=1,
        img_denoising_interval=(0.6, 1.0),
        reg_denoising_interval=(0.0, 0.2),
        cfg_interval=(1, 1),
    )
    assert isinstance(output, ODESample)
    assert len(model.reg_only_calls) == 2
    assert len(model.calls) == 4


def test_noisy_interpolation_uses_its_own_timestep() -> None:
    clean = torch.ones(2, 1)
    noise = torch.zeros_like(clean)
    assert torch.allclose(
        noisy_interpolate(clean, noise, torch.tensor([0.25, 0.75])),
        torch.tensor([[0.25], [0.75]]),
    )


class DistinctGuidanceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_classes=3)

    def forward(
        self,
        image: torch.Tensor,
        timestep: torch.Tensor,
        labels: torch.Tensor,
        *,
        noisy_reg: torch.Tensor | None = None,
        reg_timestep: torch.Tensor | None = None,
        return_repa: bool = False,
    ) -> DDTOutput:
        del timestep, reg_timestep, return_repa
        conditional = labels != self.config.num_classes
        image_values = torch.where(conditional, 2.0, 0.0)
        final_values = torch.where(conditional, 3.0, 100.0)
        early_values = torch.where(conditional, 1.0, 200.0)
        assert noisy_reg is not None
        return DDTOutput(
            image=image_values[:, None, None, None].expand_as(image),
            reg=final_values[:, None, None].expand_as(noisy_reg),
            reg_early=early_values[:, None, None].expand_as(noisy_reg),
        )


@pytest.mark.parametrize(("cfg_interval", "expected_image"), [((0, 1), 4.0), ((1, 1), 2.0)])
@pytest.mark.parametrize(("scale", "expected_reg"), [(0.0, 1.0), (1.0, 3.0), (1.78, 4.56)])
def test_reg_internal_guidance_scale_identities(
    scale: float,
    expected_reg: float,
    cfg_interval: tuple[int, int],
    expected_image: float,
) -> None:
    output = ode_sample(
        DistinctGuidanceModel(),
        torch.zeros(1, 1, 1, 1),
        torch.tensor([1]),
        reg_noise=torch.zeros(1, 1, 1),
        steps=1,
        guidance_scale=2,
        reg_guidance_scale=scale,
        cfg_interval=cfg_interval,
    )
    assert isinstance(output, ODESample)
    assert torch.allclose(
        output.image,
        torch.full_like(output.image, expected_image),
    )
    assert torch.allclose(
        output.reg,
        torch.full_like(output.reg, expected_reg),
    )
