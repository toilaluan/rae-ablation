from __future__ import annotations

import math

import pytest
import torch
from omegaconf.errors import ConfigKeyError
from torch import nn

from rae_ablation.config import ExperimentConfig, load_config
from rae_ablation.flow import (
    flow_losses,
    internal_guidance,
    noisy_interpolate,
    sample_timesteps,
    shift_timestep,
)
from rae_ablation.model.ddt import DDTOutput
from rae_ablation.optim import learning_rate_at_epoch
from rae_ablation.sampling import euler_sample


def test_structured_config_and_unknown_key() -> None:
    config = load_config(
        "configs/canonical.yaml",
        ["model.encoder_hidden_size=64", "model.encoder_num_heads=4"],
    )
    assert isinstance(config, ExperimentConfig)
    assert config.model.encoder_hidden_size == 64
    assert config.model.decoder_hidden_size == 2048
    assert not config.model.use_patch_forcing
    assert config.model.attnres_block_size == 4
    assert config.model.num_input_residuals == 4
    with pytest.raises(ConfigKeyError):
        load_config("configs/canonical.yaml", ["model.typo=1"])
    with pytest.raises(ConfigKeyError):
        load_config("configs/canonical.yaml", ["model.architecture=depth_attention_dit"])
    with pytest.raises(ConfigKeyError):
        load_config("configs/canonical.yaml", ["train.use_patch_forcing=true"])


def test_ddt_and_timestep_config_validation() -> None:
    config = ExperimentConfig()
    config.model.encoder_hidden_size = 63
    with pytest.raises(ValueError, match="divisible"):
        config.validate()

    config = ExperimentConfig()
    config.model.base_model_depth = config.model.encoder_depth + 1
    with pytest.raises(ValueError, match="base_model_depth"):
        config.validate()

    config = ExperimentConfig()
    config.train.timestep_scale = 0
    with pytest.raises(ValueError, match="timestep_scale"):
        config.validate()

    config = ExperimentConfig()
    config.model.attnres_block_size = 3
    with pytest.raises(ValueError, match="attnres_block_size"):
        config.validate()

    config = ExperimentConfig()
    config.model.attnres_block_size = 4
    config.model.num_input_residuals = 1
    with pytest.raises(ValueError, match="num_input_residuals"):
        config.validate()

    config = ExperimentConfig()
    config.model.attnres_block_size = 4
    config.compile.fullgraph = True
    with pytest.raises(ValueError, match="fullgraph"):
        config.validate()


def test_uniform_logit_normal_timesteps_are_bounded_and_deterministic() -> None:
    first = sample_timesteps(
        4,
        (3, 2, 2),
        torch.Generator().manual_seed(7),
        "cpu",
        loc=-0.25,
        scale=1.5,
    )
    second = sample_timesteps(
        4,
        (3, 2, 2),
        torch.Generator().manual_seed(7),
        "cpu",
        loc=-0.25,
        scale=1.5,
    )
    assert torch.equal(first.conditioning, second.conditioning)
    assert torch.equal(first.interpolation, first.conditioning)
    assert torch.all((first.conditioning > 0) & (first.conditioning < 1))


def test_patch_forcing_matches_sampling_mechanism() -> None:
    seed = 11
    batch, height, width = 2, 8, 8
    actual = sample_timesteps(
        batch,
        (3, height, width),
        torch.Generator().manual_seed(seed),
        "cpu",
        use_patch_forcing=True,
    )

    generator = torch.Generator().manual_seed(seed)
    maximum = torch.randn(batch, generator=generator).sigmoid()
    maximum_per_patch = maximum.reshape(batch, 1, 1, 1)
    deviation = torch.randn((batch, 1, height, width), generator=generator) * (
        maximum_per_patch / 2
    )
    candidate = maximum_per_patch - deviation.abs()
    replacement = torch.rand(candidate.shape, generator=generator) * maximum_per_patch
    expected = torch.where(candidate < 0, replacement, candidate)

    assert torch.any(candidate < 0)
    assert torch.equal(actual.conditioning, expected)
    assert torch.equal(actual.interpolation, expected)
    assert torch.equal(actual.conditioning, actual.interpolation)
    assert actual.interpolation.shape == (batch, 1, height, width)
    assert torch.all(actual.interpolation >= 0)
    assert torch.all(actual.interpolation <= maximum_per_patch)


def test_noise_data_endpoints_and_velocity_losses() -> None:
    clean = torch.randn(2, 3, 2, 2)
    noise = torch.randn_like(clean)
    endpoints = torch.tensor([0.0, 1.0])
    mixed = noisy_interpolate(clean, noise, endpoints)
    assert torch.equal(mixed[0], noise[0])
    assert torch.equal(mixed[1], clean[1])

    timestep = torch.tensor([0.2, 0.8])
    losses = flow_losses(clean, noise, timestep, clean, clean)
    assert losses.total.item() == pytest.approx(0, abs=1e-12)

    patchwise = torch.rand(2, 1, 2, 2).mul(0.8).add(0.1)
    patch_losses = flow_losses(clean, noise, patchwise, clean, clean)
    assert patch_losses.total.item() == pytest.approx(0, abs=1e-12)


def test_shifted_euler_runs_from_noise_to_data() -> None:
    assert torch.allclose(
        shift_timestep(torch.tensor([0.0, 0.5, 1.0]), 4),
        torch.tensor([0.0, 0.8, 1.0]),
    )

    class CleanPredictor(nn.Module):
        def forward(
            self, latent: torch.Tensor, timestep: torch.Tensor, _labels: torch.Tensor
        ) -> DDTOutput:
            prediction = torch.full_like(latent, 2)
            return DDTOutput(prediction, prediction)

    result = euler_sample(
        CleanPredictor(),
        torch.zeros(1, 1, 1, 1),
        torch.zeros(1, dtype=torch.long),
        steps=1,
    )
    assert torch.equal(result, torch.full_like(result, 2))


def test_internal_guidance_is_always_on() -> None:
    early = torch.ones(2, 1, 1, 1)
    full = 3 * early
    assert torch.equal(internal_guidance(early, full, scale=2), 5 * early)
    assert torch.equal(internal_guidance(early, full, scale=1), full)


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [(0, 2e-4), (25, 2e-4), (37.5, 1.1e-4), (50, 2e-5), (80, 2e-5)],
)
def test_learning_rate_recipe(epoch: float, expected: float) -> None:
    assert math.isclose(learning_rate_at_epoch(epoch), expected)
