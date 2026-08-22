from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rope import apply_rope_cos_sin, precompute_rope_tables


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype) * self.weight


class MLP(nn.Module):
    """The unconditioned pixel-decoder MLP retained from schema v17."""

    def __init__(self, width: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = max(1, int(width * mlp_ratio))
        self.w1 = nn.Linear(width, hidden)
        self.w2 = nn.Linear(hidden, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)))


class NormalizedAttention(nn.Module):
    """The q/k-normalized attention retained by the stage-1 pixel decoder."""

    def __init__(self, width: int, num_heads: int, spatial_size: int) -> None:
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.proj = nn.Linear(width, width)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            cos, sin = precompute_rope_tables(
                spatial_size, spatial_size, self.head_dim, dtype=dtype
            )
            self.register_buffer(f"rope_cos_{name}", cos, persistent=False)
            self.register_buffer(f"rope_sin_{name}", sin, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        shape = (batch, tokens, self.num_heads, self.head_dim)
        q = self.q(x).reshape(shape).transpose(1, 2)
        k = self.k(x).reshape(shape).transpose(1, 2)
        v = self.v(x).reshape(shape).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        suffix = "bf16" if q.dtype == torch.bfloat16 else "fp32"
        cos = getattr(self, f"rope_cos_{suffix}")
        sin = getattr(self, f"rope_sin_{suffix}")
        if cos.dtype != q.dtype:
            cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q, k = apply_rope_cos_sin(q, k, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.proj(attended.transpose(1, 2).reshape(batch, tokens, self.width))


class UnconditionedTransformerBlock(nn.Module):
    """Transformer block used only by the frozen-target pixel decoder stage."""

    def __init__(
        self, width: int, num_heads: int, mlp_ratio: float, spatial_size: int
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attention = NormalizedAttention(width, num_heads, spatial_size)
        self.norm2 = RMSNorm(width)
        self.feed_forward = MLP(width, mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))
