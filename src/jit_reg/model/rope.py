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
    head_dim = q.shape[-1]
    if head_dim % 4:
        raise ValueError("head dimension must be divisible by four")
    half = head_dim // 2
    y = _axis_angles(y_positions.to(q.device), half, q.dtype).reshape(
        1, 1, -1, half
    )
    x = _axis_angles(x_positions.to(q.device), half, q.dtype).reshape(
        1, 1, -1, half
    )

    def rotate(tensor: Tensor) -> Tensor:
        ty, tx = tensor.split(half, dim=-1)
        ty = ty * y.cos() + _rotate_pairs(ty) * y.sin()
        tx = tx * x.cos() + _rotate_pairs(tx) * x.sin()
        return torch.cat((ty, tx), dim=-1)

    return rotate(q), rotate(k)


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
