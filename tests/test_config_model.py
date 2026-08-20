from __future__ import annotations

from copy import deepcopy

import pytest
import torch

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
        num_classes=3,
        num_timestep_tokens=2,
        num_mix_tokens=2,
        num_class_tokens=2,
        variant="DDT-Tiny",
    )


def test_schema_v14_defaults_and_matched_presets() -> None:
    config = ExperimentConfig()
    config.validate()
    assert config.schema_version == CONFIG_SCHEMA_VERSION == 14
    assert config.vision.token_shape == (768, 16, 16)
    assert config.model.input_shape == config.vision.token_shape
    assert config.vision.target_layer == 7
    assert config.vision.semantic_layer == 12
    assert config.repa == RepaConfig(layer_depth=8, projector_width=2048, coefficient=0.5)
    assert load_config("configs/dino-tokenmix.yaml").mixing.mode == "token_mix"
    assert load_config("configs/dino-r1-baseline.yaml").mixing.mode == "target_baseline"


def test_old_and_unknown_configuration_are_rejected(tmp_path) -> None:
    old = tmp_path / "old.yaml"
    old.write_text("schema_version: 13\n")
    with pytest.raises(ValueError, match="schema_version: 14"):
        load_config(old)
    current = tmp_path / "current.yaml"
    current.write_text("schema_version: 14\nunknown: true\n")
    with pytest.raises(ValueError, match="schema-v14"):
        load_config(current)


def test_ddt_direct_velocity_repa_shapes_and_zero_head() -> None:
    model = DDT(tiny_config(), RepaConfig(layer_depth=2, projector_width=24))
    x = torch.randn(2, 4, 2, 2)
    timestep = torch.tensor([0.2, 0.7])
    ratio = torch.tensor([0.4, 1.0])
    labels = torch.tensor([1, 2])
    output = model(x, timestep, ratio, labels)
    assert output.velocity.shape == x.shape
    assert output.repa_projection.shape == (2, 4, 4)
    assert torch.count_nonzero(output.velocity) == 0
    assert torch.equal(output.velocity, model.forward_velocity(x, timestep, ratio, labels))
    assert all(
        torch.count_nonzero(block.modulation[-1].weight) == 0
        for block in model.decoder_blocks
    )


def test_repa_loss_backpropagates_to_selected_encoder_depth() -> None:
    model = DDT(tiny_config(), RepaConfig(layer_depth=2, projector_width=24))
    projection = model(
        torch.randn(2, 4, 2, 2),
        torch.rand(2),
        torch.rand(2),
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
    second = DDT(deepcopy(tiny_config()), RepaConfig(layer_depth=2, projector_width=24))
    second.load_state_dict(first.state_dict())
    second.set_gradient_checkpointing(True)
    args = (
        torch.randn(2, 4, 2, 2),
        torch.rand(2),
        torch.rand(2),
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
    model = DDT(
        tiny_config(), RepaConfig(layer_depth=2, projector_width=24)
    ).eval()
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    output = compiled(
        torch.randn(1, 4, 2, 2),
        torch.rand(1),
        torch.rand(1),
        torch.tensor([1]),
    )
    assert output.velocity.shape == (1, 4, 2, 2)
    assert output.repa_projection.shape == (1, 4, 4)
