from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jit_reg.sampling import ode_sample


class GuidanceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_classes=2)
        self.ratios: list[torch.Tensor] = []

    def forward_velocity(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        ratio: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        del timestep
        self.ratios.append(ratio.detach().clone())
        shape = (labels.shape[0], *((1,) * (noisy.ndim - 1)))
        return labels.reshape(shape).to(noisy.dtype).expand_as(noisy)


def test_sampling_fixes_r_to_one_and_applies_standard_cfg() -> None:
    model = GuidanceModel()
    result = ode_sample(
        model,
        torch.zeros(1, 1, 1, 1),
        torch.tensor([1]),
        steps=1,
        timestep_shift=1,
        cfg_scale=2,
        num_classes=2,
    )
    # Integration runs from t=1 to t=0: v_cfg=2 + 2*(1-2)=0.
    assert result.item() == pytest.approx(0.0)
    assert torch.equal(model.ratios[0], torch.ones(2))


def test_no_cfg_direct_velocity_integration() -> None:
    model = GuidanceModel()
    result = ode_sample(
        model,
        torch.zeros(1, 1, 1, 1),
        torch.tensor([1]),
        steps=1,
        timestep_shift=1,
        num_classes=2,
    )
    assert result.item() == pytest.approx(-1.0)
