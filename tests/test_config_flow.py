from __future__ import annotations

import math

import pytest
import torch
from omegaconf.errors import ConfigKeyError
from torch import nn

from jit_reg.config import ExperimentConfig, load_config
from jit_reg.model.sit import SiTOutput, classifier_free_labels
from jit_reg.objective import (
    jit_losses,
    noisy_interpolate,
    sample_timesteps,
    shift_timestep,
)
from jit_reg.optim import learning_rate_at_epoch
from jit_reg.sampling import EulerSample, euler_sample


def test_structured_config_and_unknown_key() -> None:
    config = load_config(
        "configs/canonical.yaml",
        ["model.hidden_size=64", "model.num_heads=4"],
    )
    assert isinstance(config, ExperimentConfig)
    assert config.model.patch_size == 32
    assert not config.model.reg.enabled
    with pytest.raises(ConfigKeyError):
        load_config("configs/canonical.yaml", ["model.typo=1"])
    with pytest.raises(ConfigKeyError):
        load_config("configs/canonical.yaml", ["cache.workers=0"])


@pytest.mark.parametrize(
    ("image_size", "patch_size", "message"),
    [(256, 0, "positive"), (255, 32, "divisible")],
)
def test_patch_configuration_rejection(
    image_size: int, patch_size: int, message: str
) -> None:
    config = ExperimentConfig()
    config.model.image_size = image_size
    config.model.patch_size = patch_size
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_head_dimension_and_reg_block_rejection() -> None:
    config = ExperimentConfig()
    config.model.hidden_size = 63
    with pytest.raises(ValueError, match="divisible"):
        config.validate()
    config = ExperimentConfig()
    config.model.reg.enabled = True
    config.model.depth = 3
    with pytest.raises(ValueError, match="repa_block"):
        config.validate()


def test_logit_normal_timesteps_are_bounded_and_deterministic() -> None:
    first = sample_timesteps(
        4, torch.Generator().manual_seed(7), "cpu", loc=-0.25, scale=1.5
    )
    second = sample_timesteps(
        4, torch.Generator().manual_seed(7), "cpu", loc=-0.25, scale=1.5
    )
    assert torch.equal(first, second)
    assert torch.all((first > 0) & (first < 1))


@pytest.mark.parametrize("timestep", [torch.tensor([0.2, 0.8]), torch.tensor([0.96, 0.99])])
def test_clean_prediction_has_zero_velocity_loss_across_clipping_interval(
    timestep: torch.Tensor,
) -> None:
    clean = torch.randn(2, 3, 2, 2)
    noise = torch.randn_like(clean)
    mixed = noisy_interpolate(clean, noise, torch.tensor([0.0, 1.0]))
    assert torch.equal(mixed[0], noise[0])
    assert torch.equal(mixed[1], clean[1])
    losses = jit_losses(clean, noise, timestep, clean)
    assert losses.total.item() == pytest.approx(0, abs=1e-12)


def test_reg_weighted_loss_composition() -> None:
    clean = torch.randn(2, 3, 2, 2)
    noise = torch.randn_like(clean)
    cls = torch.randn(2, 8)
    cls_noise = torch.randn_like(cls)
    target = torch.randn(2, 5, 8)
    losses = jit_losses(
        clean,
        noise,
        torch.tensor([0.3, 0.99]),
        clean,
        clean_cls=cls,
        cls_noise=cls_noise,
        cls_prediction=cls,
        repa_prediction=target,
        repa_target=target,
    )
    assert losses.image_velocity.item() == pytest.approx(0)
    assert losses.cls_velocity.item() == pytest.approx(0)
    assert losses.repa.item() == pytest.approx(-1)
    assert losses.total.item() == pytest.approx(-0.5)


def test_label_dropout_uses_null_class_and_explicit_rng() -> None:
    labels = torch.arange(100)
    first = classifier_free_labels(
        labels, 100, 0.1, torch.Generator().manual_seed(4)
    )
    second = classifier_free_labels(
        labels, 100, 0.1, torch.Generator().manual_seed(4)
    )
    assert torch.equal(first, second)
    assert torch.any(first == 100)
    assert torch.all((first == labels) | (first == 100))


class _CFGModel(nn.Module):
    class Config:
        num_classes = 10

    config = Config()

    def forward(
        self,
        image: torch.Tensor,
        _timestep: torch.Tensor,
        labels: torch.Tensor,
        noisy_cls: torch.Tensor | None = None,
        return_repa: bool = False,
    ) -> SiTOutput:
        del return_repa
        conditional = (labels < self.config.num_classes).to(image.dtype)
        image_value = (2 * conditional).view(-1, 1, 1, 1).expand_as(image)
        cls = None
        if noisy_cls is not None:
            cls_value = 1 + 2 * conditional
            cls = cls_value[:, None].expand_as(noisy_cls)
        return SiTOutput(image_value, cls)


def test_euler_cfg_scale_interval_boundaries_and_joint_reg_state() -> None:
    assert torch.allclose(
        shift_timestep(torch.tensor([0.0, 0.5, 1.0]), 4),
        torch.tensor([0.0, 0.8, 1.0]),
    )
    model = _CFGModel()
    noise = torch.zeros(1, 1, 1, 1)
    labels = torch.tensor([2])
    conditional = euler_sample(model, noise, labels, steps=1, guidance_scale=1)
    assert torch.equal(conditional, torch.full_like(noise, 2))
    guided = euler_sample(model, noise, labels, steps=1, guidance_scale=2)
    assert torch.equal(guided, torch.full_like(noise, 4))
    outside = euler_sample(
        model,
        noise,
        labels,
        steps=1,
        guidance_scale=2,
        cfg_interval=(0.1, 1.0),
    )
    assert torch.equal(outside, torch.full_like(noise, 2))
    reg = euler_sample(
        model,
        noise,
        labels,
        cls_noise=torch.zeros(1, 4),
        steps=1,
        guidance_scale=1,
    )
    assert isinstance(reg, EulerSample)
    assert torch.equal(reg.image, torch.full_like(noise, 2))
    assert torch.equal(reg.cls, torch.full((1, 4), 3.0))


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [(0, 2e-4), (25, 2e-4), (37.5, 1.1e-4), (50, 2e-5), (80, 2e-5)],
)
def test_learning_rate_recipe(epoch: float, expected: float) -> None:
    assert math.isclose(learning_rate_at_epoch(epoch), expected)
