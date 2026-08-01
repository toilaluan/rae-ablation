from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from jit_reg.config import ExperimentConfig, ModelConfig, REPAConfig, load_config
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


def test_official_presets_are_valid_and_cover_core_modes() -> None:
    expected = {
        "sit-baseline.yaml": (False, 1, "none"),
        "sit-repa.yaml": (True, 1, "none"),
        "sit-k-final.yaml": (False, 4, "final_velocity"),
        "sit-k-final-repa.yaml": (True, 4, "final_velocity"),
        "sit-k-repa.yaml": (True, 4, "repa"),
    }
    for name, values in expected.items():
        config = load_config(f"configs/{name}")
        assert (config.repa.enabled, config.exploration.k, config.exploration.selector) == values
        assert config.model.variant == "SiT-B"


@pytest.mark.parametrize(
    ("k", "selector", "repa", "message"),
    [
        (1, "final_velocity", False, "K=1"),
        (2, "none", False, "K>1"),
        (2, "repa", False, "REPA selection"),
    ],
)
def test_exploration_validation(k: int, selector: str, repa: bool, message: str) -> None:
    config = ExperimentConfig()
    config.exploration.k = k
    config.exploration.selector = selector
    config.repa.enabled = repa
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_candidate_chunk_must_divide_k() -> None:
    config = ExperimentConfig()
    config.exploration.k = 4
    config.exploration.selector = "final_velocity"
    config.exploration.candidate_chunk_size = 3
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


def test_sit_x_prediction_and_repa_prefix_match_full_forward() -> None:
    model = SiT(tiny_model(), tiny_repa())
    noisy = torch.randn(2, 3, 8, 8)
    timestep = torch.tensor([0.2, 0.7])
    labels = torch.tensor([1, 2])
    output = model(noisy, timestep, labels, return_repa=True)
    prefix = model.forward_repa(noisy, timestep, labels)
    assert output.x_prediction.shape == noisy.shape
    assert output.repa_prediction is not None
    assert output.repa_prediction.shape == (2, 16, 8)
    assert torch.equal(output.repa_prediction, prefix)


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
