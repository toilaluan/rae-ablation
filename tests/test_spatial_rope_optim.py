from __future__ import annotations

import pytest
import torch
from torch import nn

from jit_reg.model.rope import (
    apply_2d_rope,
    apply_2d_rope_positions,
    apply_aligned_spatial_rope,
)
from jit_reg.model.sit import Attention
from jit_reg.optim import _muon_eligible


def test_spatial_rope_uses_image_aligned_cell_centers() -> None:
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn_like(q)
    actual = apply_aligned_spatial_rope(q, k, 2, 4, 4)
    coordinates = torch.tensor([0.5, 0.5, 2.5, 2.5])
    x_coordinates = torch.tensor([0.5, 2.5, 0.5, 2.5])
    expected = apply_2d_rope_positions(q, k, coordinates, x_coordinates)
    assert torch.allclose(actual[0], expected[0])
    assert torch.allclose(actual[1], expected[1])


def test_attention_rotates_entire_spatial_reg_prefix_and_image_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = Attention(
        8,
        1,
        spatial_pool_size=2,
        spatial_rope=True,
    )
    with torch.no_grad():
        attention.qkv.weight.zero_()
        attention.qkv.bias.zero_()
        identity = torch.eye(8)
        attention.qkv.weight[:8].copy_(identity)
        attention.qkv.weight[8:16].copy_(identity)
        attention.qkv.weight[16:].copy_(identity)
        attention.projection.weight.copy_(identity)
        attention.projection.bias.zero_()
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        assert not is_causal
        captured.append((q.detach().clone(), k.detach().clone()))
        return v

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", capture)
    tokens = torch.randn(1, 8, 8)
    attention(tokens, height=2, width=2, patch_start=4)
    raw = attention.q_norm(tokens.reshape(1, 1, 8, 8))
    expected_spatial, _ = apply_aligned_spatial_rope(
        raw[:, :, :4], raw[:, :, :4], 2, 2, 2
    )
    expected_patches, _ = apply_2d_rope(raw[:, :, 4:], raw[:, :, 4:], 2, 2)
    rotated = captured[0][0]
    assert torch.allclose(rotated[:, :, :4], expected_spatial)
    assert torch.allclose(rotated[:, :, 4:], expected_patches)


@pytest.mark.parametrize(
    ("name", "shape", "eligible"),
    [
        ("reg_head.modulation.1.weight", (16, 16), True),
        ("reg_head.layers.0.feed_forward.gate.weight", (32, 16), True),
        ("reg_head.layers.0.norm.weight", (16,), False),
        ("reg_head.layers.0.feed_forward.gate.bias", (32,), False),
        ("reg_head.output.projection.weight", (8, 16), False),
        ("reg_head.token_projection.weight", (32, 16), False),
    ],
)
def test_reg_head_muon_routing(name: str, shape: tuple[int, ...], eligible: bool) -> None:
    parameter = nn.Parameter(torch.empty(shape))
    assert _muon_eligible(name, parameter) is eligible
