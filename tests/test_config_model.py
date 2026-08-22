from __future__ import annotations

import pytest
import torch

from cascade_dit.config import CONFIG_SCHEMA_VERSION, ExperimentConfig, ModelConfig, load_config
from cascade_dit.model import DecoupledCascadeDiT, LatentDiT, SharedCascadeDiT, build_model


def tiny_config(architecture: str) -> ModelConfig:
    return ModelConfig(
        architecture=architecture,
        hidden_size=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        base_model_depth=1,
        num_classes=3,
        num_timestep_tokens=2,
        num_class_tokens=2,
    )


def inputs(batch: int = 2) -> tuple[torch.Tensor, ...]:
    semantic = torch.randn(batch, 768, 8, 8)
    image_latents = torch.randn(batch, 128, 16, 16)
    timestep = torch.rand(batch)
    labels = torch.arange(batch).remainder(3)
    mask = torch.arange(batch).remainder(2).bool()
    return semantic, image_latents, timestep, labels, mask


def test_presets_and_clean_schema_break(tmp_path) -> None:
    assert CONFIG_SCHEMA_VERSION == 1
    assert load_config("configs/latent-baseline.yaml").model.architecture == "latent"
    assert load_config("configs/cascade-shared.yaml").model.architecture == "shared_cascade"
    assert load_config("configs/cascade-decoupled.yaml").model.architecture == "decoupled_cascade"
    old = tmp_path / "old.yaml"
    old.write_text("schema_version: 16\nvision: {}\n")
    with pytest.raises(ValueError, match="schema_version: 1"):
        load_config(old)
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("schema_version: 1\ntarget_steps: 40\n")
    with pytest.raises(ValueError, match="schema-v1"):
        load_config(unknown)


def test_three_architectures_have_expected_shapes_and_ownership() -> None:
    semantic, image_latents, timestep, labels, mask = inputs()
    baseline = build_model(tiny_config("latent"))
    assert isinstance(baseline, LatentDiT)
    assert baseline(image_latents, timestep, labels).shape == image_latents.shape
    assert not any("semantic" in name for name, _module in baseline.named_modules())

    shared = build_model(tiny_config("shared_cascade"))
    assert isinstance(shared, SharedCascadeDiT)
    shared_output = shared(
        semantic, timestep, labels, image_latents, timestep, labels, semantic, mask
    )
    assert shared_output.semantic.full.shape == semantic.shape
    assert shared_output.semantic.base.shape == semantic.shape
    assert shared_output.image_latent.shape == image_latents.shape

    decoupled = build_model(tiny_config("decoupled_cascade"))
    assert isinstance(decoupled, DecoupledCascadeDiT)
    semantic_parameters = {parameter.data_ptr() for parameter in decoupled.semantic_dit.parameters()}
    image_parameters = {parameter.data_ptr() for parameter in decoupled.image_latent_dit.parameters()}
    assert semantic_parameters.isdisjoint(image_parameters)


def test_conditional_adaln_is_zero_initialized_and_receives_gradients() -> None:
    semantic, image_latents, timestep, labels, _mask = inputs()
    model = build_model(tiny_config("decoupled_cascade"))
    conditional = model.image_latent_dit
    assert all(
        torch.count_nonzero(block.semantic_adaln.weight) == 0
        for block in conditional.blocks
    )
    conditional(image_latents, timestep, labels, semantic).square().mean().backward()
    assert all(
        block.semantic_adaln.weight.grad is not None
        and block.semantic_adaln.weight.grad.abs().sum() > 0
        for block in conditional.blocks
    )
    assert all(block.semantic_adaln is None for block in model.semantic_dit.blocks)


@pytest.mark.parametrize("architecture", ["latent", "shared_cascade", "decoupled_cascade"])
def test_training_forward_supports_fullgraph_compilation(architecture: str) -> None:
    semantic, image_latents, timestep, labels, mask = inputs(batch=1)
    compiled = torch.compile(
        build_model(tiny_config(architecture)).eval(), backend="eager", fullgraph=True
    )
    output = (
        compiled(image_latents, timestep, labels)
        if architecture == "latent"
        else compiled(
            semantic,
            timestep,
            labels,
            image_latents,
            timestep,
            labels,
            semantic,
            mask,
        )
    )
    assert output is not None


def test_default_config_is_valid() -> None:
    ExperimentConfig().validate()
