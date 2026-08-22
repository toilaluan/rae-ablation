from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from jit_reg.config import CONFIG_SCHEMA_VERSION, DiTConfig, ExperimentConfig, load_config
from jit_reg.model import SemanticDiT


def tiny_config() -> DiTConfig:
    return DiTConfig(
        input_size=2,
        patch_size=1,
        in_channels=4,
        hidden_size=16,
        depth=3,
        num_heads=4,
        mlp_ratio=2,
        gradient_checkpointing=False,
        variant="SemanticDiT-Tiny",
    )


def _paths(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        result: set[str] = set()
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result |= _paths(child, path)
        return result
    return {prefix}


def _enable_conditioned_output(model: SemanticDiT) -> None:
    with torch.no_grad():
        model.final_layer.projection.weight.normal_()
        model.final_layer.modulation[-1].weight.normal_(std=0.1)
        for block in model.blocks:
            block.modulation[-1].weight.normal_(std=0.1)


def test_schema_v18_defaults_and_single_complete_preset() -> None:
    config = ExperimentConfig()
    config.validate()
    assert config.schema_version == CONFIG_SCHEMA_VERSION == 18
    assert config.vision.token_shape == (768, 8, 8)
    assert config.model.input_shape == config.vision.token_shape
    assert config.model == DiTConfig()
    assert config.decoder.train.epochs == 10
    assert config.flow.train.epochs == 20
    assert config.sampling.steps == 40
    assert config.sampling.noise_seed == 0
    assert config.flow.endpoint == "target"
    assert sum(parameter.numel() for parameter in SemanticDiT(config.model).parameters()) == 131_312_640

    preset = load_config("configs/dino-cascade.yaml")
    raw = OmegaConf.to_container(
        OmegaConf.load("configs/dino-cascade.yaml"), resolve=True
    )
    assert isinstance(raw, dict)
    assert _paths(raw) == _paths(preset.as_dict())


def test_old_and_unknown_configuration_are_rejected(tmp_path) -> None:
    old = tmp_path / "old.yaml"
    old.write_text("schema_version: 17\n")
    with pytest.raises(ValueError, match="schema_version: 18"):
        load_config(old)
    current = tmp_path / "current.yaml"
    current.write_text("schema_version: 18\nunknown: true\n")
    with pytest.raises(ValueError, match="schema-v18"):
        load_config(current)


def test_matched_endpoint_routes_do_not_change_model_configuration() -> None:
    target = ExperimentConfig()
    pixel = deepcopy(target)
    pixel.flow.endpoint = "pixel"
    target.validate()
    pixel.validate()
    assert target.model == pixel.model
    assert sum(parameter.numel() for parameter in SemanticDiT(target.model).parameters()) == sum(
        parameter.numel() for parameter in SemanticDiT(pixel.model).parameters()
    )


def test_semantic_dit_shape_and_zero_initialization() -> None:
    model = SemanticDiT(tiny_config()).eval()
    noisy = torch.randn(2, 4, 2, 2)
    semantic = torch.randn_like(noisy)
    output = model(noisy, torch.tensor([0.2, 0.7]), semantic)
    assert output.shape == noisy.shape
    assert torch.count_nonzero(output) == 0


def test_semantic_adaln_and_timestep_receive_gradients() -> None:
    model = SemanticDiT(tiny_config())
    _enable_conditioned_output(model)
    noisy = torch.randn(2, 4, 2, 2)
    semantic = torch.randn_like(noisy, requires_grad=True)
    model(noisy, torch.tensor([0.2, 0.7]), semantic).square().mean().backward()
    assert semantic.grad is not None
    assert model.semantic_projection.weight.grad is not None
    assert model.timestep_embedder.mlp[0].weight.grad is not None
    assert all(block.modulation[-1].weight.grad is not None for block in model.blocks)


def test_gradient_checkpointing_matches_uncheckpointed_execution() -> None:
    first = SemanticDiT(tiny_config())
    _enable_conditioned_output(first)
    second = SemanticDiT(deepcopy(tiny_config()))
    second.load_state_dict(first.state_dict())
    second.set_gradient_checkpointing(True)
    noisy = torch.randn(2, 4, 2, 2)
    timestep = torch.rand(2)
    semantic = torch.randn_like(noisy)
    first_loss = first(noisy, timestep, semantic).sum()
    second_loss = second(noisy, timestep, semantic).sum()
    first_loss.backward()
    second_loss.backward()
    assert torch.allclose(first_loss, second_loss)
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            assert torch.allclose(left.grad, right.grad)


def test_semantic_dit_supports_fullgraph_compilation() -> None:
    model = SemanticDiT(tiny_config()).eval()
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    noisy = torch.randn(1, 4, 2, 2)
    output = compiled(noisy, torch.rand(1), torch.randn_like(noisy))
    assert output.shape == noisy.shape
