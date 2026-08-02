from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from jit_reg.config import (
    CONFIG_SCHEMA_VERSION,
    RAE_BASE_REPOSITORY,
    RAE_BASE_REVISION,
    RAE_COMPRESSOR_REPOSITORY,
    RAE_COMPRESSOR_REVISION,
    DDTConfig,
    ExperimentConfig,
    load_config,
)
from jit_reg.model import DDT


def tiny_config() -> DDTConfig:
    return DDTConfig(
        input_size=2,
        in_channels=4,
        hidden_size=[16, 32],
        depth=[3, 1],
        num_heads=[4, 4],
        mlp_ratio=2,
        base_model_depth=2,
        num_classes=3,
        num_timestep_tokens=2,
        num_class_tokens=2,
        variant="DDT-Tiny",
    )


def test_exact_schema_v13_defaults_and_presets() -> None:
    config = ExperimentConfig()
    config.validate()
    assert config.schema_version == CONFIG_SCHEMA_VERSION == 13
    assert config.model.input_shape == (1024, 8, 8)
    assert config.model.patch_size == [1, 1]
    assert config.model.hidden_size == [768, 2048]
    assert config.model.depth == [12, 2]
    assert config.model.num_heads == [12, 16]
    assert config.model.base_model_depth == 4
    assert (config.model.num_timestep_tokens, config.model.num_class_tokens) == (4, 8)
    assert config.train.base_loss_weight == 1.0
    assert (
        config.rae.base_repository,
        config.rae.base_revision,
        config.rae.compressor_repository,
        config.rae.compressor_revision,
    ) == (
        RAE_BASE_REPOSITORY,
        RAE_BASE_REVISION,
        RAE_COMPRESSOR_REPOSITORY,
        RAE_COMPRESSOR_REVISION,
    )
    expected = {
        "ddt-baseline.yaml": (1, 1),
        "ddt-k-early.yaml": (4, 1),
        "ddt-k-final.yaml": (1, 4),
        "ddt-k-hybrid.yaml": (4, 2),
    }
    for name, values in expected.items():
        preset = load_config(f"configs/{name}")
        assert (preset.exploration.k_early, preset.exploration.k_full) == values


def test_old_and_unknown_configuration_are_rejected(tmp_path) -> None:
    old = tmp_path / "old.yaml"
    old.write_text("schema_version: 12\n")
    with pytest.raises(ValueError, match="schema_version: 13"):
        load_config(old)
    current = tmp_path / "current.yaml"
    current.write_text("schema_version: 13\nrepa: {}\n")
    with pytest.raises(ValueError, match="schema-v13"):
        load_config(current)


def test_ddt_validation_checks_pairs_grid_heads_and_base_depth() -> None:
    config = tiny_config()
    config.hidden_size = [16]
    with pytest.raises(ValueError, match="two values"):
        config.validate()
    config = tiny_config()
    config.patch_size = [3, 1]
    with pytest.raises(ValueError, match="divisible"):
        config.validate()
    config = tiny_config()
    config.num_heads = [3, 4]
    with pytest.raises(ValueError, match="divisible"):
        config.validate()
    config = tiny_config()
    config.base_model_depth = 4
    with pytest.raises(ValueError, match="encoder depth"):
        config.validate()


def test_ddt_paths_shapes_zero_initialization_and_base_isolation() -> None:
    model = DDT(tiny_config())
    x = torch.randn(2, 4, 2, 2)
    timestep = torch.tensor([0.2, 0.7])
    labels = torch.tensor([1, 2])
    output = model(x, timestep, labels)
    assert output.full_prediction.shape == x.shape
    assert output.base_prediction.shape == x.shape
    assert torch.equal(output.base_prediction, model.forward_base_prediction(x, timestep, labels))
    assert torch.equal(output.full_prediction, model.forward_full_prediction(x, timestep, labels))
    assert torch.count_nonzero(output.full_prediction) == 0
    assert torch.count_nonzero(output.base_prediction) == 0
    assert all(torch.count_nonzero(block.modulation[-1].weight) == 0 for block in model.decoder_blocks)

    with torch.no_grad():
        model.base_head.projection.weight.normal_()
    before = model.forward_base_prediction(x, timestep, labels)
    with torch.no_grad():
        for block in model.encoder_blocks[model.config.base_model_depth :]:
            for parameter in block.parameters():
                parameter.add_(torch.randn_like(parameter))
        for parameter in model.decoder_blocks.parameters():
            parameter.add_(torch.randn_like(parameter))
    after = model.forward_base_prediction(x, timestep, labels)
    assert torch.equal(before, after)


def test_base_loss_backpropagates_through_first_four_equivalent_blocks() -> None:
    model = DDT(tiny_config())
    with torch.no_grad():
        model.base_head.projection.weight.normal_()
    prediction = model.forward_base_prediction(
        torch.randn(2, 4, 2, 2), torch.rand(2), torch.tensor([1, 2])
    )
    prediction.square().mean().backward()
    assert model.spatial_projection.weight.grad is not None
    assert all(next(block.parameters()).grad is not None for block in model.encoder_blocks[:2])
    assert all(parameter.grad is None for block in model.encoder_blocks[2:] for parameter in block.parameters())
    assert all(parameter.grad is None for parameter in model.decoder_blocks.parameters())


def test_gradient_checkpointing_matches_uncheckpointed_execution() -> None:
    first = DDT(tiny_config())
    with torch.no_grad():
        first.full_head.projection.weight.normal_()
        first.base_head.projection.weight.normal_()
    second = DDT(deepcopy(tiny_config()))
    second.load_state_dict(first.state_dict())
    second.set_gradient_checkpointing(True)
    args = (torch.randn(2, 4, 2, 2), torch.rand(2), torch.tensor([1, 2]))
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
    model = DDT(tiny_config()).eval()
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    output = compiled(torch.randn(1, 4, 2, 2), torch.rand(1), torch.tensor([1]))
    assert output.full_prediction.shape == (1, 4, 2, 2)
    assert output.base_prediction.shape == (1, 4, 2, 2)
