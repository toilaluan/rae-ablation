from __future__ import annotations

import pytest
import torch
from torch import nn

from jit_reg.model.layers import DDTAttention
from jit_reg.model.rope import (
    apply_2d_rope,
    apply_2d_rope_positions,
    apply_aligned_spatial_rope,
    apply_rope_cos_sin,
    precompute_ddt_rope_tables,
    precompute_rope_tables,
)
from jit_reg.model.tokens import DDTTokenLayout
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
    attention = DDTAttention(
        8,
        1,
        grid_size=2,
        reg_pool_size=2,
        layout=DDTTokenLayout(4, 4),
        spatial_reg_rope=True,
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
    attention(tokens)
    raw = attention.q_norm(tokens.reshape(1, 1, 8, 8))
    expected_spatial, _ = apply_aligned_spatial_rope(
        raw[:, :, :4], raw[:, :, :4], 2, 2, 2
    )
    expected_patches, _ = apply_2d_rope(raw[:, :, 4:], raw[:, :, 4:], 2, 2)
    rotated = captured[0][0]
    assert torch.allclose(rotated[:, :, :4], expected_spatial)
    assert torch.allclose(rotated[:, :, 4:], expected_patches)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_precomputed_image_rope_matches_functional_reference(
    dtype: torch.dtype,
) -> None:
    q = torch.randn(2, 3, 6, 8).to(dtype)
    k = torch.randn_like(q)
    cos, sin = precompute_rope_tables(2, 3, 8, dtype=dtype)
    actual = apply_rope_cos_sin(q, k, cos, sin)
    expected = apply_2d_rope(q, k, 2, 3)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


@pytest.mark.parametrize("spatial_rope", [False, True])
def test_precomputed_prefix_rope_matches_identity_or_spatial_reference(
    spatial_rope: bool,
) -> None:
    q = torch.randn(1, 2, 8, 8)
    k = torch.randn_like(q)
    cos, sin = precompute_rope_tables(
        2,
        2,
        8,
        prefix_tokens=4,
        spatial_pool_size=2 if spatial_rope else None,
    )
    actual_q, actual_k = apply_rope_cos_sin(q, k, cos, sin)
    if spatial_rope:
        prefix_q, prefix_k = apply_aligned_spatial_rope(
            q[:, :, :4], k[:, :, :4], 2, 2, 2
        )
    else:
        prefix_q, prefix_k = q[:, :, :4], k[:, :, :4]
    image_q, image_k = apply_2d_rope(
        q[:, :, 4:], k[:, :, 4:], 2, 2
    )
    assert torch.equal(actual_q, torch.cat((prefix_q, image_q), dim=2))
    assert torch.equal(actual_k, torch.cat((prefix_k, image_k), dim=2))


def test_spatial_rope_can_be_embedded_between_conditioning_prefixes() -> None:
    q = torch.randn(1, 2, 12, 8)
    k = torch.randn_like(q)
    cos, sin = precompute_rope_tables(
        2,
        2,
        8,
        prefix_tokens=8,
        spatial_pool_size=2,
        spatial_prefix_start=2,
    )
    actual_q, actual_k = apply_rope_cos_sin(q, k, cos, sin)
    reg_q, reg_k = apply_aligned_spatial_rope(
        q[:, :, 2:6], k[:, :, 2:6], 2, 2, 2
    )
    image_q, image_k = apply_2d_rope(
        q[:, :, 8:], k[:, :, 8:], 2, 2
    )
    assert torch.equal(
        actual_q,
        torch.cat((q[:, :, :2], reg_q, q[:, :, 6:8], image_q), dim=2),
    )
    assert torch.equal(
        actual_k,
        torch.cat((k[:, :, :2], reg_k, k[:, :, 6:8], image_k), dim=2),
    )


def test_ddt_rope_orders_reg_image_repa_then_identity_conditions() -> None:
    q = torch.randn(1, 2, 15, 8)
    k = torch.randn_like(q)
    cos, sin = precompute_ddt_rope_tables(
        2,
        2,
        8,
        reg_pool_size=2,
        repa_pool_size=2,
        condition_tokens=3,
        spatial_reg_rope=True,
    )
    actual_q, actual_k = apply_rope_cos_sin(q, k, cos, sin)
    reg_q, reg_k = apply_aligned_spatial_rope(
        q[:, :, :4],
        k[:, :, :4],
        2,
        2,
        2,
    )
    image_q, image_k = apply_2d_rope(
        q[:, :, 4:8],
        k[:, :, 4:8],
        2,
        2,
    )
    repa_q, repa_k = apply_aligned_spatial_rope(
        q[:, :, 8:12],
        k[:, :, 8:12],
        2,
        2,
        2,
    )
    assert torch.equal(
        actual_q,
        torch.cat((reg_q, image_q, repa_q, q[:, :, 12:]), dim=2),
    )
    assert torch.equal(
        actual_k,
        torch.cat((reg_k, image_k, repa_k, k[:, :, 12:]), dim=2),
    )


def test_attention_rope_tables_are_nonpersistent_and_forward_is_static(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = DDTAttention(
        8,
        1,
        grid_size=2,
        reg_pool_size=2,
        layout=DDTTokenLayout(4, 4),
        spatial_reg_rope=False,
    )
    assert {
        "rope_cos_fp32",
        "rope_sin_fp32",
        "rope_cos_bf16",
        "rope_sin_bf16",
    } <= dict(attention.named_buffers()).keys()
    assert not any("rope_" in name for name in attention.state_dict())

    def unexpected(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("coordinate or trigonometric work ran in forward")

    monkeypatch.setattr(torch, "arange", unexpected)
    monkeypatch.setattr(torch, "outer", unexpected)
    monkeypatch.setattr(torch.Tensor, "cos", unexpected)
    monkeypatch.setattr(torch.Tensor, "sin", unexpected)
    assert attention(torch.randn(1, 8, 8)).shape == (1, 8, 8)


@pytest.mark.parametrize(
    ("name", "shape", "eligible"),
    [
        ("decoder_blocks.0.modulation.1.weight", (96, 16), True),
        ("decoder_blocks.0.feed_forward.gate.weight", (32, 16), True),
        ("decoder_blocks.0.norm1.weight", (16,), False),
        ("decoder_blocks.0.feed_forward.gate.bias", (32,), False),
        ("final_layer.reg_projection.weight", (8, 16), False),
        ("decoder_reg_projection.weight", (32, 8), False),
        ("context_projection.weight", (32, 16), False),
    ],
)
def test_ddt_parameter_muon_routing(
    name: str,
    shape: tuple[int, ...],
    eligible: bool,
) -> None:
    parameter = nn.Parameter(torch.empty(shape))
    assert _muon_eligible(name, parameter) is eligible
