from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from jit_reg.config import (
    EarlyPredictionConfig,
    ExperimentConfig,
    ModelConfig,
    REPAConfig,
    load_config,
)
from jit_reg.model.sit import FusedSwiGLU, SiT, SiTAttention


def tiny_model() -> ModelConfig:
    return ModelConfig(
        image_size=8,
        patch_size=2,
        hidden_size=32,
        depth=3,
        num_heads=4,
        mlp_ratio=2,
        num_classes=10,
        variant="SiT-Tiny",
    )


def tiny_repa() -> REPAConfig:
    config = REPAConfig(enabled=True, block=2, projection_hidden_size=16)
    config.encoder.feature_dim = 8
    return config


def tiny_early() -> EarlyPredictionConfig:
    return EarlyPredictionConfig(enabled=True, block=2, mlp_hidden_size=16)


def test_official_presets_are_valid_and_cover_core_modes() -> None:
    expected = {
        "sit-baseline.yaml": (False, False, 1, 1, "none"),
        "sit-repa.yaml": (True, False, 1, 1, "none"),
        "sit-k-final.yaml": (False, False, 1, 4, "none"),
        "sit-k-final-repa.yaml": (True, False, 1, 4, "none"),
        "sit-k-repa.yaml": (True, False, 4, 1, "repa"),
        "sit-k-hybrid-early.yaml": (False, True, 4, 2, "early_prediction"),
        "sit-k-hybrid-repa.yaml": (True, False, 4, 2, "repa"),
    }
    for name, values in expected.items():
        config = load_config(f"configs/{name}")
        assert (
            config.repa.enabled,
            config.early_prediction.enabled,
            config.exploration.k_early,
            config.exploration.k_full,
            config.exploration.early_selector,
        ) == values
        assert config.model.variant == "SiT-B"


@pytest.mark.parametrize(
    ("k_early", "selector", "repa", "early", "message"),
    [
        (1, "repa", True, False, "k_early=1"),
        (2, "none", False, False, "k_early>1"),
        (2, "repa", False, False, "REPA selection"),
        (2, "early_prediction", False, False, "early prediction selection"),
    ],
)
def test_exploration_validation(
    k_early: int,
    selector: str,
    repa: bool,
    early: bool,
    message: str,
) -> None:
    config = ExperimentConfig()
    config.exploration.k_early = k_early
    config.exploration.early_selector = selector
    config.repa.enabled = repa
    config.early_prediction.enabled = early
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_candidate_chunk_must_divide_k() -> None:
    config = ExperimentConfig()
    config.repa.enabled = True
    config.exploration.k_early = 4
    config.exploration.k_full = 2
    config.exploration.early_selector = "repa"
    config.exploration.early_candidate_chunk_size = 3
    with pytest.raises(ValueError, match="divide"):
        config.validate()
    config.exploration.early_candidate_chunk_size = None
    config.exploration.full_candidate_chunk_size = 3
    with pytest.raises(ValueError, match="divide"):
        config.validate()


def test_pixel_and_flux2_geometry() -> None:
    pixel = tiny_model()
    assert pixel.input_shape == (3, 8, 8)
    latent = ModelConfig(
        input_space="flux2_latents",
        image_size=256,
        in_channels=128,
        patch_size=2,
        hidden_size=32,
        depth=2,
        num_heads=4,
    )
    latent.validate()
    assert latent.input_shape == (128, 16, 16)
    assert latent.grid_size == 8


def test_sit_x_prediction_and_prefixes_match_full_forward() -> None:
    model = SiT(tiny_model(), tiny_repa(), tiny_early())
    noisy = torch.randn(2, 3, 8, 8)
    timestep = torch.tensor([0.2, 0.7])
    labels = torch.tensor([1, 2])
    output = model(
        noisy,
        timestep,
        labels,
        return_repa=True,
        return_early_prediction=True,
    )
    repa_prefix = model.forward_repa(noisy, timestep, labels)
    early_prefix = model.forward_early_prediction(noisy, timestep, labels)
    assert output.x_prediction.shape == noisy.shape
    assert output.repa_prediction is not None
    assert output.repa_prediction.shape == (2, 16, 8)
    assert torch.equal(output.repa_prediction, repa_prefix)
    assert output.early_x_prediction is not None
    assert output.early_x_prediction.shape == noisy.shape
    assert torch.equal(output.early_x_prediction, early_prefix)
    assert model(noisy, timestep, labels).early_x_prediction is None


def test_early_prediction_head_detaches_transformer_states() -> None:
    model = SiT(tiny_model(), early_prediction_config=tiny_early())
    output = model(
        torch.randn(2, 3, 8, 8),
        torch.rand(2),
        torch.tensor([1, 2]),
        return_early_prediction=True,
    )
    assert output.early_x_prediction is not None
    output.early_x_prediction.sum().backward()
    assert model.early_prediction_head is not None
    assert model.early_prediction_head.projection.weight.grad is not None
    assert model.patch_projection.weight.grad is None
    assert all(parameter.grad is None for block in model.blocks for parameter in block.parameters())


def test_modern_block_components_are_always_enabled() -> None:
    model = SiT(tiny_model())
    block = model.blocks[0]
    assert isinstance(block.attention, SiTAttention)
    assert isinstance(block.feed_forward, FusedSwiGLU)
    assert block.attention.qkv.out_features == 3 * tiny_model().hidden_size
    assert hasattr(block.attention, "rope_cos_bf16")
    assert block.attention.q_norm.normalized_shape == (8,)


def test_zero_initialized_x_head_and_adaln() -> None:
    model = SiT(tiny_model())
    output = model(
        torch.randn(2, 3, 8, 8),
        torch.rand(2),
        torch.tensor([1, 2]),
    )
    assert torch.count_nonzero(output.x_prediction) == 0
    assert all(torch.count_nonzero(block.modulation[-1].weight) == 0 for block in model.blocks)


def test_gradient_checkpointing_preserves_gradients() -> None:
    config = tiny_model()
    first = SiT(config)
    second = SiT(deepcopy(config))
    second.load_state_dict(first.state_dict())
    second.set_gradient_checkpointing(True)
    args = (torch.randn(2, 3, 8, 8), torch.rand(2), torch.tensor([1, 2]))
    first(*args).x_prediction.sum().backward()
    second(*args).x_prediction.sum().backward()
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        assert torch.allclose(left.grad, right.grad)
