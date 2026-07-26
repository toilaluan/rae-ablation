from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from jit_reg.config import (
    EncoderConfig,
    FeaturesConfig,
    ModelConfig,
    REGConfig,
    REGEarlyPredictionConfig,
    REGHeadConfig,
    REPAConfig,
)
from jit_reg.model import sit
from jit_reg.model.sit import SiT, SiTOutput


def tiny_config(
    *,
    reg: bool = False,
    repa: bool = False,
    head: str = "standard",
    early: int | None = None,
    reinjection_mode: str = "replace",
    scope: str = "image",
) -> ModelConfig:
    return ModelConfig(
        image_size=16,
        patch_size=8,
        hidden_size=16,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        num_classes=3,
        features=FeaturesConfig(
            encoder=EncoderConfig(feature_dim=8, register_tokens=4),
            reg=REGConfig(
                enabled=reg,
                pool_size=2,
                head=REGHeadConfig(
                    kind=head,
                    hidden_size=32,
                    depth=2,
                    mlp_ratio=2,
                ),
                early_prediction=REGEarlyPredictionConfig(
                    block=early,
                    projection_hidden_size=32,
                    reinjection_mode=reinjection_mode,
                ),
            ),
            repa=REPAConfig(
                enabled=repa,
                block=1,
                projection_hidden_size=16,
                token_scope=scope,
            ),
        ),
    )


def inputs(config: ModelConfig) -> tuple[torch.Tensor, ...]:
    batch = 2
    image = torch.randn(batch, 3, 16, 16)
    timestep = torch.tensor([0.2, 0.7])
    labels = torch.tensor([0, 2])
    reg = config.features.reg
    noisy_reg = (
        torch.randn(batch, reg.num_tokens, config.features.encoder.feature_dim)
        if reg.enabled
        else None
    )
    return image, timestep, labels, noisy_reg


def run(config: ModelConfig, *, return_repa: bool = False) -> SiTOutput:
    model = SiT(config)
    image, timestep, labels, noisy_reg = inputs(config)
    return model(
        image,
        timestep,
        labels,
        noisy_reg=noisy_reg,
        reg_timestep=timestep if noisy_reg is not None else None,
        return_repa=return_repa,
    )


def test_feature_disabled_model_is_baseline_only() -> None:
    model = SiT(tiny_config())
    assert model.reg_input is None
    assert model.reg_head is None
    assert model.reg_reinjection is None
    assert model.repa_projection is None
    assert not any(
        key.startswith(("reg_", "repa_")) for key in model.state_dict()
    )
    image, timestep, labels, _ = inputs(model.config)
    output = model(image, timestep, labels)
    assert output.image.shape == image.shape
    assert output.reg is None and output.repa is None


@pytest.mark.parametrize("head", ["standard", "wide_mlp"])
@pytest.mark.parametrize("early", [None, 1])
def test_reg_head_token_and_location_matrix(head: str, early: int | None) -> None:
    config = tiny_config(
        reg=True,
        head=head,
        early=early,
    )
    output = run(config)
    assert output.reg is not None
    assert output.reg.shape == (
        2,
        config.features.reg.num_tokens,
        config.features.encoder.feature_dim,
    )


@pytest.mark.parametrize(
    ("reg", "scope", "expected_tokens"),
    [(False, "image", 4), (True, "image", 4), (True, "image_and_reg", 8)],
)
def test_repa_scope_shapes(reg: bool, scope: str, expected_tokens: int) -> None:
    output = run(
        tiny_config(reg=reg, repa=True, scope=scope),
        return_repa=True,
    )
    assert output.repa is not None
    assert output.repa.shape == (2, expected_tokens, 8)


def test_reg_requires_explicit_reg_timestep() -> None:
    config = tiny_config(reg=True)
    model = SiT(config)
    image, timestep, labels, noisy_reg = inputs(config)
    with pytest.raises(ValueError, match="reg_timestep"):
        model(image, timestep, labels, noisy_reg=noisy_reg)


def test_early_prediction_reinjects_exact_fc_gelu_fc_prefix() -> None:
    config = tiny_config(
        reg=True,
        head="standard",
        early=1,
    )
    model = SiT(config)
    captured: list[torch.Tensor] = []

    def capture(_module: nn.Module, args: tuple[object, ...]) -> None:
        captured.append(args[0].detach().clone())  # type: ignore[union-attr]

    handle = model.blocks[1].register_forward_pre_hook(capture)
    image, timestep, labels, noisy_reg = inputs(config)
    output = model(
        image,
        timestep,
        labels,
        noisy_reg=noisy_reg,
        reg_timestep=timestep * 0.5,
    )
    handle.remove()
    assert output.reg is not None and model.reg_reinjection is not None
    expected = model._add_reg_positions(model.reg_reinjection(output.reg))
    assert len(captured) == 1
    assert torch.allclose(
        captured[0][:, : model.num_reg_tokens],
        expected,
        atol=1e-6,
    )
    assert captured[0].shape[1] == model.num_reg_tokens + model.num_patches
    assert [type(layer) for layer in model.reg_reinjection] == [
        nn.Linear,
        nn.GELU,
        nn.Linear,
    ]


def test_additive_early_prediction_reinjects_into_existing_prefix() -> None:
    config = tiny_config(
        reg=True,
        head="standard",
        early=1,
        reinjection_mode="add",
    )
    model = SiT(config)
    previous: list[torch.Tensor] = []
    captured: list[torch.Tensor] = []

    def capture_previous(
        _module: nn.Module, _args: tuple[object, ...], output: torch.Tensor
    ) -> None:
        previous.append(output.detach().clone())

    first = model.blocks[0].register_forward_hook(capture_previous)
    second = model.blocks[1].register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    image, timestep, labels, noisy_reg = inputs(config)
    output = model(
        image,
        timestep,
        labels,
        noisy_reg=noisy_reg,
        reg_timestep=timestep,
    )
    first.remove()
    second.remove()
    assert output.reg is not None and model.reg_reinjection is not None
    expected = (
        previous[0][:, : model.num_reg_tokens]
        + model.reg_reinjection(output.reg)
    )
    assert torch.allclose(
        captured[0][:, : model.num_reg_tokens], expected, atol=1e-6
    )


def test_every_block_receives_separate_image_and_reg_conditioning() -> None:
    config = tiny_config(reg=True)
    model = SiT(config)
    conditionings: list[tuple[torch.Tensor, torch.Tensor]] = []
    handles = []
    for block in model.blocks:
        handles.append(
            block.register_forward_pre_hook(
                lambda _module, args: conditionings.append(
                    (args[1].detach().clone(), args[5].detach().clone())
                )
            )
        )
    image, timestep, labels, noisy_reg = inputs(config)
    reg_timestep = torch.tensor([0.8, 0.9])
    model(
        image,
        timestep,
        labels,
        noisy_reg=noisy_reg,
        reg_timestep=reg_timestep,
    )
    for handle in handles:
        handle.remove()
    assert len(conditionings) == config.depth
    for image_conditioning, reg_conditioning in conditionings:
        assert image_conditioning.shape == reg_conditioning.shape == (
            2,
            config.hidden_size,
        )
        assert not torch.allclose(image_conditioning, reg_conditioning)


def test_removed_experiment_parameters_and_classes_are_absent() -> None:
    source = inspect.getsource(sit)
    for retired in (
        "continuation_tokens",
        "x0_conditioning",
        "PatchReprojection",
        "wide_intermediate",
        "terminal_wide",
    ):
        assert retired not in source
    model = SiT(tiny_config(reg=True, head="wide_mlp", early=1))
    for retired in ("continuation", "reprojection", "reg_x0_conditioning"):
        assert not any(retired in key for key in model.state_dict())
