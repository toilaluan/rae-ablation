from __future__ import annotations

import torch
from torch import Tensor


def _rotate_pairs(x: Tensor) -> Tensor:
    shape = x.shape
    x = x.reshape(*shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _axis_angles(positions: Tensor, half_dim: int, dtype: torch.dtype) -> Tensor:
    if half_dim % 2:
        raise ValueError("RoPE half dimension must be even")
    inv_freq = 1.0 / (
        10_000
        ** (
            torch.arange(
                0, half_dim, 2, device=positions.device, dtype=torch.float32
            )
            / half_dim
        )
    )
    angles = torch.outer(positions.float(), inv_freq).repeat_interleave(2, dim=-1)
    return angles.to(dtype=dtype)


def rope_cos_sin(
    y_positions: Tensor,
    x_positions: Tensor,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Build factorized 2D RoPE tables for explicit coordinates."""
    if (
        y_positions.ndim != 1
        or x_positions.ndim != 1
        or y_positions.shape != x_positions.shape
    ):
        raise ValueError("RoPE coordinates must be equal one-dimensional tensors")
    if head_dim % 4:
        raise ValueError("head dimension must be divisible by four")
    half = head_dim // 2
    y = _axis_angles(y_positions, half, dtype)
    x = _axis_angles(x_positions, half, dtype)
    angles = torch.cat((y, x), dim=-1)
    return angles.cos(), angles.sin()


def apply_rope_cos_sin(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply a precomputed RoPE table to complete Q/K token sequences."""
    if q.shape != k.shape or q.ndim != 4:
        raise ValueError("q and k must have equal [B, heads, tokens, dim] shapes")
    if cos.shape != sin.shape or cos.shape != q.shape[-2:]:
        raise ValueError("RoPE tables must have shape [tokens, head_dim]")
    cos = cos.reshape(1, 1, *cos.shape)
    sin = sin.reshape(1, 1, *sin.shape)
    return (
        q * cos + _rotate_pairs(q) * sin,
        k * cos + _rotate_pairs(k) * sin,
    )


def precompute_rope_tables(
    image_height: int,
    image_width: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Build a 2D RoPE table for a static image-token layout."""
    if image_height < 1 or image_width < 1:
        raise ValueError("RoPE grid dimensions must be positive")
    image_y = torch.arange(image_height, dtype=torch.float32)
    image_x = torch.arange(image_width, dtype=torch.float32)
    image_y = (
        image_y[:, None].expand(image_height, image_width).reshape(-1)
    )
    image_x = (
        image_x[None, :].expand(image_height, image_width).reshape(-1)
    )
    return rope_cos_sin(image_y, image_x, head_dim, dtype)


def apply_2d_rope_positions(
    q: Tensor,
    k: Tensor,
    y_positions: Tensor,
    x_positions: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply factorized 2D RoPE at explicit coordinates."""
    if q.shape != k.shape or q.ndim != 4:
        raise ValueError("q and k must have equal [B, heads, tokens, dim] shapes")
    if (
        y_positions.ndim != 1
        or x_positions.ndim != 1
        or y_positions.shape != x_positions.shape
        or q.shape[-2] != y_positions.shape[0]
    ):
        raise ValueError("RoPE coordinates must match the token count")
    cos, sin = rope_cos_sin(
        y_positions.to(q.device),
        x_positions.to(q.device),
        q.shape[-1],
        q.dtype,
    )
    return apply_rope_cos_sin(q, k, cos, sin)


def apply_2d_rope(q: Tensor, k: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
    """Apply factorized 2D rotary embeddings to ``[B, heads, HW, dim]`` tensors."""
    if height < 1 or width < 1:
        raise ValueError("RoPE grid dimensions must be positive")
    y = torch.arange(height, device=q.device, dtype=torch.float32)
    x = torch.arange(width, device=q.device, dtype=torch.float32)
    y = y[:, None].expand(height, width).reshape(-1)
    x = x[None, :].expand(height, width).reshape(-1)
    return apply_2d_rope_positions(q, k, y, x)
