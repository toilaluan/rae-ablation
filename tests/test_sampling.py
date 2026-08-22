from __future__ import annotations

import pytest
import torch
from torch import nn

from jit_reg.sampling import flow_sample, one_step_sample


class EndpointModel(nn.Module):
    def __init__(self, endpoint: float = 2.0) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.timesteps: list[torch.Tensor] = []
        self.inputs: list[torch.Tensor] = []
        self.semantics: list[torch.Tensor] = []

    def forward(
        self, noisy: torch.Tensor, timestep: torch.Tensor, semantic: torch.Tensor
    ) -> torch.Tensor:
        self.timesteps.append(timestep.detach().clone())
        self.inputs.append(noisy.detach().clone())
        self.semantics.append(semantic.detach().clone())
        return torch.full_like(noisy, self.endpoint)


def test_one_step_starts_from_noise_and_conditions_on_semantic() -> None:
    model = EndpointModel()
    semantic = torch.full((2, 1, 1, 1), 5.0)
    noise = torch.full_like(semantic, -3.0)
    result = one_step_sample(model, semantic, noise)
    assert torch.all(result.endpoint == 2)
    assert result.nfe == 1
    assert torch.equal(model.inputs[0], noise)
    assert torch.equal(model.semantics[0], semantic)
    assert torch.equal(model.timesteps[0], torch.ones(2))


def test_euler_integrates_noise_to_endpoint_with_exact_step_count() -> None:
    model = EndpointModel()
    semantic = torch.full((1, 1, 1, 1), 99.0)
    noise = torch.full_like(semantic, 5.0)
    result = flow_sample(
        model,
        semantic,
        noise,
        steps=2,
        timestep_shift=1,
        solver="euler",
    )
    assert result.endpoint.item() == pytest.approx(2.0)
    assert result.nfe == 2
    assert [float(value[0]) for value in model.timesteps] == pytest.approx([1.0, 0.5])


def test_heun_skips_the_singular_endpoint_correction() -> None:
    model = EndpointModel()
    semantic = torch.zeros(1, 1, 1, 1)
    result = flow_sample(
        model,
        semantic,
        torch.ones_like(semantic),
        steps=3,
        timestep_shift=1,
        solver="heun",
    )
    assert result.endpoint.item() == pytest.approx(2.0)
    assert result.nfe == 5
    assert len(model.timesteps) == 5


def test_invalid_sampling_configuration_is_rejected() -> None:
    values = torch.zeros(1, 1, 1, 1)
    with pytest.raises(ValueError, match="positive steps"):
        flow_sample(EndpointModel(), values, values, steps=0)
    with pytest.raises(ValueError, match="matching shapes"):
        one_step_sample(EndpointModel(), values, torch.zeros(1, 2, 1, 1))
