from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jit_reg.model import DDTOutput
from jit_reg.objective import SEMANTIC_STAGE, TARGET_STAGE
from jit_reg.sample_runtime import SamplingParameters
from jit_reg.sampling import ode_sample


class GuidanceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_classes=2)
        self.stages: list[torch.Tensor] = []
        self.timesteps: list[torch.Tensor] = []

    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        stage: torch.Tensor,
        labels: torch.Tensor,
    ) -> DDTOutput:
        self.stages.append(stage.detach().clone())
        self.timesteps.append(timestep.detach().clone())
        shape = (labels.shape[0], *((1,) * (noisy.ndim - 1)))
        label = labels.reshape(shape).to(noisy.dtype)
        full = torch.ones_like(noisy) * (2 + label)
        base = torch.ones_like(noisy) * label
        projection = torch.zeros(labels.shape[0], 1, 1, device=noisy.device)
        return DDTOutput(full, base, projection)


class SamplingOnlyModel(GuidanceModel):
    def forward_predictions(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        stage: torch.Tensor,
        labels: torch.Tensor,
    ) -> SimpleNamespace:
        output = super().forward(noisy, timestep, stage, labels)
        return SimpleNamespace(
            full_prediction=output.full_prediction,
            base_prediction=output.base_prediction,
        )

    def forward(self, *args: object, **kwargs: object) -> DDTOutput:
        raise AssertionError("sampling should use forward_predictions")


def test_default_step_allocation_has_explicit_solver_nfe() -> None:
    assert SamplingParameters().sampling_nfe == 50
    assert SamplingParameters(solver="heun").sampling_nfe == 99


def test_euler_uses_exact_per_stage_step_allocation_and_boundary() -> None:
    model = SamplingOnlyModel()
    ode_sample(
        model,
        torch.zeros(1, 1, 1, 1),
        torch.tensor([1]),
        semantic_steps=2,
        target_steps=3,
        waypoint_time=0.5,
        timestep_shift=1,
        ig_scale=1,
        num_classes=2,
    )
    assert [int(stage[0]) for stage in model.stages] == [
        SEMANTIC_STAGE,
        SEMANTIC_STAGE,
        TARGET_STAGE,
        TARGET_STAGE,
        TARGET_STAGE,
    ]
    assert [float(timestep[0]) for timestep in model.timesteps] == pytest.approx(
        [1.0, 0.75, 0.5, 1 / 3, 1 / 6]
    )


def test_heun_reports_two_n_minus_one_model_calls_without_crossing_stage() -> None:
    model = GuidanceModel()
    ode_sample(
        model,
        torch.zeros(1, 1, 1, 1),
        torch.tensor([1]),
        semantic_steps=2,
        target_steps=3,
        timestep_shift=1,
        ig_scale=1,
        solver="heun",
        num_classes=2,
    )
    assert len(model.stages) == 2 * 5 - 1
    values = [int(stage[0]) for stage in model.stages]
    assert values[:4] == [SEMANTIC_STAGE] * 4
    assert values[4:] == [TARGET_STAGE] * 5


def test_internal_guidance_precedes_cfg_across_both_legs() -> None:
    noise = torch.zeros(1, 1, 1, 1)
    pure = ode_sample(
        GuidanceModel(),
        noise,
        torch.tensor([1]),
        semantic_steps=1,
        target_steps=1,
        timestep_shift=1,
        ig_scale=2,
        num_classes=2,
    )
    assert pure.item() == pytest.approx(5.0)
    combined = ode_sample(
        GuidanceModel(),
        noise,
        torch.tensor([1]),
        semantic_steps=1,
        target_steps=1,
        timestep_shift=1,
        ig_scale=2,
        cfg_scale=2,
        num_classes=2,
    )
    # IG(cond)=5 and IG(uncond)=6, then CFG=6+2*(5-6)=4.
    assert combined.item() == pytest.approx(4.0)
