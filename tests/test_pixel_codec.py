from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from jit_reg.config import PixelCodecConfig
from jit_reg.pixel_codec import (
    PixelCodec,
    PixelTokenStatistics,
    adaptive_alignment_weight,
    load_pixel_codec_artifact,
    load_pixel_codec_checkpoint,
    save_pixel_codec_artifact,
    vision_foundation_losses,
)


def test_vf_loss_matches_tokens_and_penalizes_misalignment() -> None:
    tokens = torch.randn(2, 8, 3, 3)
    matched = vision_foundation_losses(
        tokens, tokens, cosine_margin=0.0, distance_margin=0.0
    )
    shuffled = vision_foundation_losses(
        tokens, tokens.flip(-1), cosine_margin=0.0, distance_margin=0.0
    )
    assert matched.total.item() == pytest.approx(0.0, abs=1e-6)
    assert shuffled.total > matched.total


def test_adaptive_alignment_sets_requested_gradient_ratio() -> None:
    representation = torch.tensor([2.0], requires_grad=True)
    reconstruction = (3 * representation).square().sum()
    alignment = (2 * representation).square().sum()
    weight = adaptive_alignment_weight(
        reconstruction, alignment, representation, gradient_ratio=0.1
    )
    reconstruction_gradient = torch.autograd.grad(
        reconstruction, representation, retain_graph=True
    )[0]
    weighted_alignment_gradient = torch.autograd.grad(
        weight * alignment, representation
    )[0]
    assert weighted_alignment_gradient.norm().item() == pytest.approx(
        0.1 * reconstruction_gradient.norm().item(), rel=1e-5
    )


def test_pixel_statistics_roundtrip_and_correlated_mixture_variance() -> None:
    statistics = PixelTokenStatistics(
        mean=torch.tensor([1.0, -2.0]),
        std=torch.tensor([2.0, 4.0]),
        rae_correlation=torch.tensor([0.5, -0.5]),
    )
    raw = torch.randn(3, 2, 2, 2)
    assert torch.allclose(statistics.denormalize(statistics.normalize(raw)), raw)
    variance = statistics.expected_mixture_variance(torch.tensor([0.0, 0.5, 1.0]))
    assert torch.equal(variance[0], torch.ones(2))
    assert torch.allclose(variance[1], torch.tensor([0.75, 0.25]))
    assert torch.equal(variance[2], torch.ones(2))


def test_compiled_training_checkpoint_can_be_finalized_as_artifact(tmp_path) -> None:
    config = PixelCodecConfig(widths=[32] * 6)
    original = PixelCodec(config)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file(
        {f"_orig_mod.{name}": value for name, value in original.state_dict().items()},
        str(checkpoint / "model.safetensors"),
    )

    loaded_checkpoint = load_pixel_codec_checkpoint(checkpoint, config, "cpu")
    assert not loaded_checkpoint.model.training
    assert all(not parameter.requires_grad for parameter in loaded_checkpoint.model.parameters())
    assert loaded_checkpoint.weights_sha256

    statistics = PixelTokenStatistics(
        mean=torch.zeros(config.latent_channels),
        std=torch.ones(config.latent_channels),
        rae_correlation=torch.zeros(config.latent_channels),
    )
    output = tmp_path / "artifact"
    save_pixel_codec_artifact(output, loaded_checkpoint.model, statistics, metadata={})
    artifact = load_pixel_codec_artifact(output, "cpu", expected_config=config)
    assert artifact.weights_sha256
    assert torch.equal(artifact.statistics.mean, statistics.mean)
