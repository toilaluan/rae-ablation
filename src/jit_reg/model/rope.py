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
    prefix_tokens: int = 0,
    spatial_pool_size: int | None = None,
    spatial_prefix_start: int = 0,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Build one image-plus-prefix RoPE table for a static attention layout."""
    if image_height < 1 or image_width < 1:
        raise ValueError("RoPE grid dimensions must be positive")
    if prefix_tokens < 0:
        raise ValueError("RoPE prefix token count cannot be negative")
    if spatial_prefix_start < 0:
        raise ValueError("spatial RoPE prefix start cannot be negative")
    image_y = torch.arange(image_height, dtype=torch.float32)
    image_x = torch.arange(image_width, dtype=torch.float32)
    image_y = (
        image_y[:, None].expand(image_height, image_width).reshape(-1)
    )
    image_x = (
        image_x[None, :].expand(image_height, image_width).reshape(-1)
    )
    image_cos, image_sin = rope_cos_sin(
        image_y, image_x, head_dim, dtype
    )
    if prefix_tokens == 0:
        return image_cos, image_sin
    prefix_cos = torch.ones(prefix_tokens, head_dim, dtype=dtype)
    prefix_sin = torch.zeros(prefix_tokens, head_dim, dtype=dtype)
    if spatial_pool_size is not None:
        spatial_tokens = spatial_pool_size**2
        if spatial_prefix_start + spatial_tokens > prefix_tokens:
            raise ValueError(
                "spatial RoPE tokens must fit within the prefix"
            )
        cell = torch.arange(spatial_pool_size, dtype=torch.float32) + 0.5
        prefix_y = cell * (image_height / spatial_pool_size) - 0.5
        prefix_x = cell * (image_width / spatial_pool_size) - 0.5
        prefix_y = (
            prefix_y[:, None]
            .expand(spatial_pool_size, spatial_pool_size)
            .reshape(-1)
        )
        prefix_x = (
            prefix_x[None, :]
            .expand(spatial_pool_size, spatial_pool_size)
            .reshape(-1)
        )
        spatial_cos, spatial_sin = rope_cos_sin(
            prefix_y, prefix_x, head_dim, dtype
        )
        prefix_cos[spatial_prefix_start : spatial_prefix_start + spatial_tokens] = (
            spatial_cos
        )
        prefix_sin[spatial_prefix_start : spatial_prefix_start + spatial_tokens] = (
            spatial_sin
        )
    return (
        torch.cat((prefix_cos, image_cos), dim=0),
        torch.cat((prefix_sin, image_sin), dim=0),
    )


def precompute_ddt_rope_tables(
    image_height: int,
    image_width: int,
    head_dim: int,
    *,
    reg_pool_size: int | None = None,
    condition_tokens: int = 0,
    spatial_reg_rope: bool = False,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Build static RoPE for ``[REG, image, condition]`` DDT sequences."""
    if condition_tokens < 0:
        raise ValueError("DDT condition token count cannot be negative")
    reg_tokens = reg_pool_size**2 if reg_pool_size is not None else 0
    primary_cos, primary_sin = precompute_rope_tables(
        image_height,
        image_width,
        head_dim,
        prefix_tokens=reg_tokens,
        spatial_pool_size=(
            reg_pool_size if spatial_reg_rope else None
        ),
        dtype=dtype,
    )
    if condition_tokens == 0:
        return primary_cos, primary_sin
    condition_cos = torch.ones(
        condition_tokens,
        head_dim,
        dtype=dtype,
    )
    condition_sin = torch.zeros(
        condition_tokens,
        head_dim,
        dtype=dtype,
    )
    return (
        torch.cat((primary_cos, condition_cos), dim=0),
        torch.cat((primary_sin, condition_sin), dim=0),
    )


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


def apply_aligned_spatial_rope(
    q: Tensor,
    k: Tensor,
    pool_size: int,
    image_height: int,
    image_width: int,
) -> tuple[Tensor, Tensor]:
    """Rotate pooled cells at their centers in image-patch coordinates."""
    if pool_size < 1 or image_height < 1 or image_width < 1:
        raise ValueError("spatial and image grids must be positive")
    cell = torch.arange(pool_size, device=q.device, dtype=torch.float32) + 0.5
    y = cell * (image_height / pool_size) - 0.5
    x = cell * (image_width / pool_size) - 0.5
    y = y[:, None].expand(pool_size, pool_size).reshape(-1)
    x = x[None, :].expand(pool_size, pool_size).reshape(-1)
    return apply_2d_rope_positions(q, k, y, x)
