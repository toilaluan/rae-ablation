from __future__ import annotations

import pytest
import torch

from jit_reg.pixel_codec import (
    PixelTokenStatistics,
    adaptive_alignment_weight,
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
