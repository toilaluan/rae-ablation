from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from jit_reg.config import (
    CONFIG_SCHEMA_VERSION,
    DDTConfig,
    ExperimentConfig,
    RepaConfig,
    load_config,
)
from jit_reg.model import DDT


def tiny_config() -> DDTConfig:
    return DDTConfig(
        input_size=2,
        patch_size=[1, 1],
        in_channels=4,
        hidden_size=[16, 32],
        depth=[3, 1],
        num_heads=[4, 4],
        mlp_ratio=2,
        base_model_depth=1,
        num_classes=3,
        num_timestep_tokens=2,
        num_stage_tokens=2,
        num_class_tokens=2,
        variant="DDT-Tiny",
    )


def _paths(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        result: set[str] = set()
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result |= _paths(child, path)
        return result
    return {prefix}


def test_schema_v16_defaults_and_single_complete_preset() -> None:
    config = ExperimentConfig()
    config.validate()
    assert config.schema_version == CONFIG_SCHEMA_VERSION == 16
    assert config.vision.token_shape == (768, 8, 8)
    assert config.model.input_shape == config.vision.token_shape
    assert config.model.base_model_depth == 4
    assert config.model.num_stage_tokens == 4
    assert config.cascade.pool_size == 2
    assert config.sampling.semantic_steps == 10
    assert config.sampling.target_steps == 40
    assert config.sampling.ig_scale == 1.78
    assert config.repa == RepaConfig(
        enabled=True, layer_depth=8, projector_width=2048, coefficient=0.5
    )

    preset = load_config("configs/dino-cascade.yaml")
    assert preset.repa.enabled is False
    raw = OmegaConf.to_container(
        OmegaConf.load("configs/dino-cascade.yaml"), resolve=True
    )
    assert isinstance(raw, dict)
    assert _paths(raw) == _paths(preset.as_dict())


def test_old_and_unknown_configuration_are_rejected(tmp_path) -> None:
    old = tmp_path / "old.yaml"
    old.write_text("schema_version: 15\n")
    with pytest.raises(ValueError, match="schema_version: 16"):
        load_config(old)
    current = tmp_path / "current.yaml"
    current.write_text("schema_version: 16\nunknown: true\n")
    with pytest.raises(ValueError, match="schema-v16"):
        load_config(current)


def test_cascade_and_base_depth_validation() -> None:
    config = ExperimentConfig()
    config.cascade.pool_size = 3
    with pytest.raises(ValueError, match="divide"):
        config.validate()
    config = tiny_config()
    config.base_model_depth = 4
    with pytest.raises(ValueError, match="encoder depth"):
        config.validate()


def test_ddt_dual_prediction_repa_shapes_and_zero_heads() -> None:
    model = DDT(tiny_config(), RepaConfig(layer_depth=2, projector_width=24))
    x = torch.randn(2, 4, 2, 2)
    timestep = torch.tensor([0.2, 0.7])
    stage = torch.tensor([0, 1])
    labels = torch.tensor([1, 2])
    output = model(x, timestep, stage, labels)
    assert output.full_prediction.shape == x.shape
    assert output.base_prediction.shape == x.shape
    assert output.repa_projection.shape == (2, 4, 4)
    assert torch.count_nonzero(output.full_prediction) == 0
    assert torch.count_nonzero(output.base_prediction) == 0
    assert torch.equal(
        output.full_prediction,
        model.forward_full_prediction(x, timestep, stage, labels),
    )
    assert torch.equal(
        output.base_prediction,
        model.forward_base_prediction(x, timestep, stage, labels),
    )
    predictions = model.forward_predictions(x, timestep, stage, labels)
    assert torch.equal(output.full_prediction, predictions.full_prediction)
    assert torch.equal(output.base_prediction, predictions.base_prediction)


def test_disabled_repa_removes_projector_and_projection() -> None:
    model = DDT(
        tiny_config(),
        RepaConfig(enabled=False, layer_depth=2, projector_width=24),
    )
    output = model(
        torch.randn(2, 4, 2, 2),
        torch.rand(2),
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
    )
    assert model.repa_projector is None
    assert output.repa_projection is None
    assert not any(name.startswith("repa_projector") for name, _ in model.named_parameters())


def test_base_prediction_isolated_after_configured_depth() -> None:
    model = DDT(tiny_config(), RepaConfig(layer_depth=2, projector_width=24))
    with torch.no_grad():
        model.base_head.projection.weight.normal_()
    args = (
        torch.randn(2, 4, 2, 2),
        torch.rand(2),
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
    )
    before = model.forward_base_prediction(*args)
    with torch.no_grad():
        for block in model.encoder_blocks[model.config.base_model_depth :]:
            for parameter in block.parameters():
                parameter.add_(torch.randn_like(parameter))
        for parameter in model.decoder_blocks.parameters():
            parameter.add_(torch.randn_like(parameter))
    assert torch.equal(before, model.forward_base_prediction(*args))


def test_repa_loss_backpropagates_to_selected_encoder_depth() -> None:
    model = DDT(tiny_config(), RepaConfig(layer_depth=2, projector_width=24))
    projection = model(
        torch.randn(2, 4, 2, 2),
        torch.rand(2),
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
    ).repa_projection
    projection.square().mean().backward()
    assert model.spatial_projection.weight.grad is not None
    assert all(next(block.parameters()).grad is not None for block in model.encoder_blocks[:2])
    assert all(
        parameter.grad is None
        for block in model.encoder_blocks[2:]
        for parameter in block.parameters()
    )


def test_gradient_checkpointing_matches_uncheckpointed_execution() -> None:
    first = DDT(tiny_config(), RepaConfig(layer_depth=2, projector_width=24))
    with torch.no_grad():
        first.full_head.projection.weight.normal_()
        first.base_head.projection.weight.normal_()
    second = DDT(deepcopy(tiny_config()), RepaConfig(layer_depth=2, projector_width=24))
    second.load_state_dict(first.state_dict())
    second.set_gradient_checkpointing(True)
    args = (
        torch.randn(2, 4, 2, 2),
        torch.rand(2),
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
    )
    first_loss = sum(value.sum() for value in first(*args))
    second_loss = sum(value.sum() for value in second(*args))
    first_loss.backward()
    second_loss.backward()
    assert torch.allclose(first_loss, second_loss)
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            assert torch.allclose(left.grad, right.grad)


def test_ddt_supports_fullgraph_compilation() -> None:
    model = DDT(tiny_config(), RepaConfig(layer_depth=2, projector_width=24)).eval()
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    output = compiled(
        torch.randn(1, 4, 2, 2),
        torch.rand(1),
        torch.tensor([0]),
        torch.tensor([1]),
    )
    assert output.full_prediction.shape == (1, 4, 2, 2)
    assert output.base_prediction.shape == (1, 4, 2, 2)
    assert output.repa_projection.shape == (1, 4, 4)


def test_repa_disabled_ddt_supports_fullgraph_compilation() -> None:
    model = DDT(
        tiny_config(),
        RepaConfig(enabled=False, layer_depth=2, projector_width=24),
    ).eval()
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    output = compiled(
        torch.randn(1, 4, 2, 2),
        torch.rand(1),
        torch.tensor([0]),
        torch.tensor([1]),
    )
    assert output.full_prediction.shape == (1, 4, 2, 2)
    assert output.base_prediction.shape == (1, 4, 2, 2)
    assert output.repa_projection is None
