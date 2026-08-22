from __future__ import annotations

import pytest
import torch

from cascade_dit.config import (
    CONFIG_SCHEMA_VERSION,
    ExperimentConfig,
    ModelConfig,
    load_config,
)
from cascade_dit.model import (
    DecoupledCascadeDDT,
    LatentDDT,
    SharedCascadeDDT,
    build_model,
)


def tiny_config(architecture: str, image_target: str = "mage_latents") -> ModelConfig:
    return ModelConfig(
        architecture=architecture,
        image_target=image_target,
        hidden_size=(32, 48),
        depth=(2, 1),
        num_heads=(4, 3),
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
    assert CONFIG_SCHEMA_VERSION == 2
    latent = load_config("configs/latent-baseline.yaml")
    shared = load_config("configs/cascade-shared.yaml")
    decoupled = load_config("configs/cascade-decoupled.yaml")
    assert latent.model.architecture == "latent"
    assert shared.model.architecture == "shared_cascade"
    assert decoupled.model.architecture == "decoupled_cascade"
    assert shared.model.hidden_size == [1152, 2048]
    assert shared.model.depth == [28, 2]
    assert shared.model.num_heads == [16, 16]
    assert shared.model.image_target == "mage_latents"
    assert shared.train.semantic_timestep_shift == 4.0
    assert shared.train.image_timestep_shift == 4.0
    assert shared.sampling.semantic_timestep_shift == 1.0
    assert shared.sampling.image_timestep_shift == 1.0
    assert latent.image_latents.cache.enabled is True
    assert latent.semantic_tokens.cache.enabled is False
    assert shared.image_latents.cache.enabled is True
    assert shared.semantic_tokens.cache.enabled is True
    old = tmp_path / "old.yaml"
    old.write_text("schema_version: 1\nmodel: {}\n")
    with pytest.raises(ValueError, match="schema_version: 2"):
        load_config(old)
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("schema_version: 2\ntarget_steps: 40\n")
    with pytest.raises(ValueError, match="schema-v2"):
        load_config(unknown)


def test_three_architectures_have_expected_shapes_and_ownership() -> None:
    semantic, image_latents, timestep, labels, mask = inputs()
    baseline = build_model(tiny_config("latent"))
    assert isinstance(baseline, LatentDDT)
    assert baseline(image_latents, timestep, labels).shape == image_latents.shape
    assert not any("semantic" in name for name, _module in baseline.named_modules())

    shared = build_model(tiny_config("shared_cascade"))
    assert isinstance(shared, SharedCascadeDDT)
    shared_output = shared(
        semantic, timestep, labels, image_latents, timestep, labels, semantic, mask
    )
    assert shared_output.semantic.full.shape == semantic.shape
    assert shared_output.semantic.base.shape == semantic.shape
    assert shared_output.image_latent.shape == image_latents.shape

    decoupled = build_model(tiny_config("decoupled_cascade"))
    assert isinstance(decoupled, DecoupledCascadeDDT)
    semantic_parameters = {
        parameter.data_ptr() for parameter in decoupled.semantic_ddt.parameters()
    }
    image_parameters = {
        parameter.data_ptr() for parameter in decoupled.image_latent_ddt.parameters()
    }
    assert semantic_parameters.isdisjoint(image_parameters)


def test_conditional_route_fuses_aligned_inputs_for_both_ddt_paths() -> None:
    config = tiny_config("decoupled_cascade")
    model = build_model(config).image_latent_ddt
    image_latents = torch.full((1, 128, 16, 16), 2.0)
    semantic = torch.full((1, 768, 8, 8), 3.0)
    observed: list[torch.Tensor] = []

    def record(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        observed.append(args[0].detach().clone())

    source_hook = model.source_embedder.projection.register_forward_pre_hook(record)
    target_hook = model.target_embedder.projection.register_forward_pre_hook(record)
    model(image_latents, torch.tensor([0.5]), torch.tensor([1]), semantic)
    source_hook.remove()
    target_hook.remove()
    assert len(observed) == 2
    for fused in observed:
        assert fused.shape == (1, 64, 1280)
        assert torch.all(fused[..., :512] == 2)
        assert torch.all(fused[..., 512:] == 3)


def test_decoder_adaln_and_prediction_heads_are_zero_initialized() -> None:
    model = build_model(tiny_config("decoupled_cascade"))
    for denoiser in (model.semantic_ddt, model.image_latent_ddt):
        assert all(
            torch.count_nonzero(block.adaln_modulation[-1].weight) == 0
            and torch.count_nonzero(block.adaln_modulation[-1].bias) == 0
            for block in denoiser.decoder_blocks
        )
    for head in (
        model.semantic_ddt.semantic_full_head,
        model.semantic_ddt.semantic_base_head,
        model.image_latent_ddt.image_latent_head,
    ):
        assert torch.count_nonzero(head.projection.weight) == 0
        assert torch.count_nonzero(head.projection.bias) == 0


def test_fused_source_and_target_paths_receive_gradients_after_warmup() -> None:
    semantic, image_latents, timestep, labels, _mask = inputs(batch=1)
    model = build_model(tiny_config("decoupled_cascade")).image_latent_ddt
    with torch.no_grad():
        for block in model.decoder_blocks:
            torch.nn.init.normal_(block.adaln_modulation[-1].weight, std=0.01)
        torch.nn.init.normal_(model.image_latent_head.modulation[-1].weight, std=0.01)
        torch.nn.init.normal_(model.image_latent_head.projection.weight, std=0.01)
    model(image_latents, timestep, labels, semantic).square().mean().backward()
    for embedder in (model.source_embedder, model.target_embedder):
        gradient = embedder.projection.weight.grad
        assert gradient is not None
        assert gradient.abs().sum() > 0


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


def test_legacy_shared_timestep_fields_migrate_per_route(tmp_path) -> None:
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        """schema_version: 2
train:
  timestep_loc: -0.5
  timestep_scale: 1.5
  timestep_shift: 3.0
sampling:
  timestep_shift: 2.0
"""
    )
    config = load_config(legacy)
    assert config.train.semantic_timestep_loc == -0.5
    assert config.train.image_timestep_loc == -0.5
    assert config.train.semantic_timestep_scale == 1.5
    assert config.train.image_timestep_scale == 1.5
    assert config.train.semantic_timestep_shift == 3.0
    assert config.train.image_timestep_shift == 3.0
    assert config.sampling.semantic_timestep_shift == 2.0
    assert config.sampling.image_timestep_shift == 2.0
    assert "timestep_shift" not in config.as_dict()["sampling"]


def test_legacy_timestep_override_sets_both_routes_and_mixed_source_fails(tmp_path) -> None:
    config = load_config(
        "configs/cascade-shared.yaml",
        ["train.timestep_shift=6", "sampling.timestep_shift=2.5"],
    )
    assert config.train.semantic_timestep_shift == 6
    assert config.train.image_timestep_shift == 6
    assert config.sampling.semantic_timestep_shift == 2.5
    assert config.sampling.image_timestep_shift == 2.5

    mixed = tmp_path / "mixed.yaml"
    mixed.write_text(
        """schema_version: 2
sampling:
  timestep_shift: 2.0
  semantic_timestep_shift: 3.0
"""
    )
    with pytest.raises(ValueError, match="mixes legacy"):
        load_config(mixed)


@pytest.mark.parametrize("architecture", ["latent", "shared_cascade", "decoupled_cascade"])
def test_all_architectures_support_direct_pixel_targets(architecture: str) -> None:
    config = tiny_config(architecture, image_target="pixels")
    model = build_model(config)
    pixels = torch.randn(1, 3, 256, 256)
    semantic = torch.randn(1, 768, 8, 8)
    timestep = torch.tensor([0.5])
    labels = torch.tensor([1])
    mask = torch.tensor([False])
    output = (
        model(pixels, timestep, labels)
        if architecture == "latent"
        else model(
            semantic,
            timestep,
            labels,
            pixels,
            timestep,
            labels,
            semantic,
            mask,
        ).image_latent
    )
    assert output.shape == pixels.shape
    assert config.image_target_patch_size == 32


@pytest.mark.parametrize("architecture", ["shared_cascade", "decoupled_cascade"])
def test_null_image_class_zeros_semantic_condition(architecture: str) -> None:
    config = tiny_config(architecture)
    model = build_model(config)
    image_model = getattr(model, "image_latent_ddt", model)
    embedder = getattr(image_model, "source_embedder", None)
    if embedder is None:
        embedder = image_model.image_source_embedder
    observed: list[torch.Tensor] = []

    def record(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        observed.append(args[0].detach().clone())

    hook = embedder.projection.register_forward_pre_hook(record)
    semantic = torch.ones(2, 768, 8, 8)
    image_latents = torch.zeros(2, 128, 16, 16)
    timestep = torch.full((2,), 0.5)
    model(
        semantic,
        timestep,
        torch.tensor([0, 1]),
        image_latents,
        timestep,
        torch.tensor([0, config.num_classes]),
        semantic,
        torch.tensor([False, False]),
    )
    hook.remove()
    fused = observed[-1]
    assert torch.all(fused[0, :, 512:] == 1)
    assert torch.all(fused[1, :, 512:] == 0)
