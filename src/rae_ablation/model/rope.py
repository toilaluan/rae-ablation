from __future__ import annotations

import torch
from torch import Tensor


def _rotate_pairs(x: Tensor) -> Tensor:
    shape = x.shape
    x = x.reshape(*shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _axis_angles(length: int, half_dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if half_dim % 2:
        raise ValueError("RoPE half dimension must be even")
    positions = torch.arange(length, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        10_000 ** (torch.arange(0, half_dim, 2, device=device, dtype=torch.float32) / half_dim)
    )
    angles = torch.outer(positions, inv_freq).repeat_interleave(2, dim=-1)
    return angles.to(dtype=dtype)


def apply_2d_rope(q: Tensor, k: Tensor, height: int, width: int) -> tuple[Tensor, Tensor]:
    """Apply factorized 2D rotary embeddings to ``[B, heads, HW, dim]`` tensors."""
    if q.shape != k.shape or q.ndim != 4:
        raise ValueError("q and k must have equal [B, heads, tokens, dim] shapes")
    if q.shape[-2] != height * width:
        raise ValueError("token count does not match height * width")
    head_dim = q.shape[-1]
    if head_dim % 4:
        raise ValueError("head dimension must be divisible by four")
    half = head_dim // 2
    y = _axis_angles(height, half, q.device, q.dtype)[:, None, :]
    x = _axis_angles(width, half, q.device, q.dtype)[None, :, :]
    y = y.expand(height, width, half).reshape(1, 1, height * width, half)
    x = x.expand(height, width, half).reshape(1, 1, height * width, half)

    def rotate(tensor: Tensor) -> Tensor:
        ty, tx = tensor.split(half, dim=-1)
        ty = ty * y.cos() + _rotate_pairs(ty) * y.sin()
        tx = tx * x.cos() + _rotate_pairs(tx) * x.sin()
        return torch.cat((ty, tx), dim=-1)

    return rotate(q), rotate(k)
