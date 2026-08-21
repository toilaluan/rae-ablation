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
        self.label_batches: list[torch.Tensor] = []

    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        stage: torch.Tensor,
        labels: torch.Tensor,
    ) -> DDTOutput:
        self.stages.append(stage.detach().clone())
        self.timesteps.append(timestep.detach().clone())
        self.label_batches.append(labels.detach().clone())
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


def test_semantic_uses_internal_guidance_and_target_uses_only_cfg() -> None:
    noise = torch.zeros(1, 1, 1, 1)
    pure_model = GuidanceModel()
    pure = ode_sample(
        pure_model,
        noise,
        torch.tensor([1]),
        semantic_steps=1,
        target_steps=1,
        timestep_shift=1,
        ig_scale=2,
        num_classes=2,
    )
    # Semantic IG predicts 5 at the waypoint; target uses the conditional full=3.
    assert pure.item() == pytest.approx(3.0)
    assert [batch.numel() for batch in pure_model.label_batches] == [1, 1]
    cfg_model = GuidanceModel()
    combined = ode_sample(
        cfg_model,
        noise,
        torch.tensor([1]),
        semantic_steps=1,
        target_steps=1,
        timestep_shift=1,
        ig_scale=2,
        cfg_scale=2,
        num_classes=2,
    )
    # Semantic remains IG=5 without duplication. Target CFG is 4+2*(3-4)=2.
    assert combined.item() == pytest.approx(2.0)
    assert [batch.numel() for batch in cfg_model.label_batches] == [1, 2]
