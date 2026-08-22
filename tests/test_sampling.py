from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cascade_dit.sample_runtime import SamplingParameters, sampling_output_name
from cascade_dit.sampling import sample_cascade, sample_image_latents
from cascade_dit.validation import noise_for_samples


class RecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_classes=2)
        self.semantic_states: list[torch.Tensor] = []
        self.image_states: list[torch.Tensor] = []
        self.semantic_timesteps: list[torch.Tensor] = []
        self.image_timesteps: list[torch.Tensor] = []
        self.semantic_conditions: list[torch.Tensor | None] = []

    def forward_semantic_predictions(
        self, values: torch.Tensor, timestep: torch.Tensor, labels: torch.Tensor
    ) -> SimpleNamespace:
        del labels
        self.semantic_states.append(values.detach().clone())
        self.semantic_timesteps.append(timestep.detach().clone())
        return SimpleNamespace(full=torch.zeros_like(values), base=torch.zeros_like(values))

    def forward_image_prediction(
        self,
        values: torch.Tensor,
        timestep: torch.Tensor,
        labels: torch.Tensor,
        semantic_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del labels
        self.image_states.append(values.detach().clone())
        self.image_timesteps.append(timestep.detach().clone())
        self.semantic_conditions.append(
            semantic_tokens.detach().clone() if semantic_tokens is not None else None
        )
        return torch.zeros_like(values)


def test_sampling_nfe_uses_two_routes_or_latent_only() -> None:
    assert SamplingParameters().sampling_nfe == 50
    assert SamplingParameters(solver="heun").sampling_nfe == 98
    assert SamplingParameters(semantic_steps=0).sampling_nfe == 40
    assert SamplingParameters(semantic_steps=0, solver="heun").sampling_nfe == 79


def test_cascade_initializes_fresh_image_noise() -> None:
    model = RecordingModel()
    semantic_noise = torch.full((1, 1, 1, 1), 3.0)
    image_noise = torch.full((1, 1, 1, 1), 7.0)
    generated = sample_cascade(
        model,
        semantic_noise,
        image_noise,
        torch.tensor([1]),
        semantic_steps=1,
        image_steps=1,
        num_classes=2,
    )
    assert torch.equal(model.semantic_states[0], semantic_noise)
    assert torch.equal(model.image_states[0], image_noise)
    assert generated.semantic_tokens is not None


def test_image_noise_is_batch_and_architecture_invariant() -> None:
    seeds = torch.tensor([12, 33])
    together = noise_for_samples(seeds, (2, 3, 3), "cpu", dtype=torch.float32)
    apart = torch.cat(
        [
            noise_for_samples(seed[None], (2, 3, 3), "cpu", dtype=torch.float32)
            for seed in seeds
        ]
    )
    assert torch.equal(together, apart)
    model = RecordingModel()
    sample_image_latents(
        model,
        together,
        torch.tensor([0, 1]),
        steps=1,
        num_classes=2,
    )
    assert torch.equal(model.image_states[0], together)


def test_cascade_uses_independent_semantic_and_image_timestep_shifts() -> None:
    model = RecordingModel()
    sample_cascade(
        model,
        torch.ones(1, 1, 1, 1),
        torch.ones(1, 1, 1, 1),
        torch.tensor([1]),
        semantic_steps=2,
        image_steps=2,
        semantic_timestep_shift=4.0,
        image_timestep_shift=1.0,
        num_classes=2,
    )
    assert model.semantic_timesteps[1].item() == pytest.approx(0.8)
    assert model.image_timesteps[1].item() == pytest.approx(0.5)


def test_cfg_unconditional_branch_receives_zero_semantic_condition() -> None:
    model = RecordingModel()
    semantic = torch.full((1, 3, 2, 2), 7.0)
    sample_image_latents(
        model,
        torch.ones(1, 1, 1, 1),
        torch.tensor([1]),
        semantic,
        steps=1,
        class_guidance_scale=2.0,
        num_classes=2,
    )
    condition = model.semantic_conditions[0]
    assert condition is not None
    assert torch.equal(condition[0], semantic[0])
    assert torch.count_nonzero(condition[1]) == 0


def test_equal_sampling_shifts_keep_legacy_output_name() -> None:
    equal = SamplingParameters(
        semantic_timestep_shift=2.0,
        image_timestep_shift=2.0,
    )
    distinct = SamplingParameters(
        semantic_timestep_shift=3.0,
        image_timestep_shift=2.0,
    )
    assert "-shift2p0-" in sampling_output_name(1000, equal)
    name = sampling_output_name(1000, distinct)
    assert "-semantic-shift3p0-image-shift2p0-" in name
