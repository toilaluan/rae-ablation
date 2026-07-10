from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .depth_attention import DepthAttention
from .rope import apply_2d_rope


class DiTOutput(NamedTuple):
    full: Tensor
    early: Tensor


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timestep: Tensor) -> Tensor:
        half = self.frequency_size // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / half
        )
        angles = timestep.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.mlp(embedding.to(dtype=self.mlp[0].weight.dtype))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.intermediate_size = intermediate_size
        self.up = nn.Linear(hidden_size, 2 * intermediate_size)
        self.down = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.up(x).chunk(2, dim=-1)
        return self.dropout(self.down(value * F.silu(gate)))


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class SpatialAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        layer: int,
        depth: int,
        depth_stride: int,
        depth_enabled: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout
        self.depth_attention = DepthAttention(layer, depth, depth_stride, depth_enabled)

    def forward(
        self,
        x: Tensor,
        height: int,
        width: int,
        key_bank: Tensor,
        value_bank: Tensor,
        bank_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, tokens, hidden = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        mixed_v, key_bank, value_bank, bank_mask = self.depth_attention(
            q, k, v, key_bank, value_bank, bank_mask
        )
        spatial_q, spatial_k = apply_2d_rope(q, k, height, width)
        attended = F.scaled_dot_product_attention(
            spatial_q,
            spatial_k,
            mixed_v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, hidden)
        return self.proj_dropout(self.proj(attended)), key_bank, value_bank, bank_mask


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, elementwise_affine=False)
        self.attn = SpatialAttention(
            config.hidden_size,
            config.num_heads,
            layer,
            config.depth,
            config.depth_attention.stride,
            config.depth_attention.enabled,
            config.dropout,
        )
        self.norm2 = nn.RMSNorm(config.hidden_size, elementwise_affine=False)
        self.mlp = SwiGLU(config.hidden_size, config.mlp_size, config.dropout)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_size, 6 * config.hidden_size)
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        x: Tensor,
        conditioning: Tensor,
        height: int,
        width: int,
        key_bank: Tensor,
        value_bank: Tensor,
        bank_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(conditioning).chunk(
            6, dim=-1
        )
        attended, key_bank, value_bank, bank_mask = self.attn(
            _modulate(self.norm1(x), shift_a, scale_a),
            height,
            width,
            key_bank,
            value_bank,
            bank_mask,
        )
        x = x + gate_a[:, None] * attended
        x = x + gate_m[:, None] * self.mlp(_modulate(self.norm2(x), shift_m, scale_m))
        return x, key_bank, value_bank, bank_mask


class PredictionHead(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.projection = nn.Linear(hidden_size, out_channels)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale = self.modulation(conditioning).chunk(2, dim=-1)
        return self.projection(_modulate(self.norm(x), shift, scale))


class DiffusionTransformer(nn.Module):
    """Class-conditional latent DiT with early prediction and causal depth mixing."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.input_projection = nn.Linear(c.in_channels, c.hidden_size)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, c.base_grid_size * c.base_grid_size, c.hidden_size)
        )
        self.time_embedding = TimestepEmbedding(c.hidden_size)
        self.class_embedding = nn.Embedding(c.num_classes, c.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(c, layer) for layer in range(c.depth)])
        self.early_head = PredictionHead(c.hidden_size, c.in_channels)
        self.full_head = PredictionHead(c.hidden_size, c.in_channels)
        self.gradient_checkpointing = c.gradient_checkpointing
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.class_embedding.weight, std=0.02)

    def interpolated_position_embedding(self, height: int, width: int) -> Tensor:
        base = self.config.base_grid_size
        pos = self.position_embedding.reshape(1, base, base, self.config.hidden_size).permute(
            0, 3, 1, 2
        )
        if (height, width) != (base, base):
            pos = F.interpolate(
                pos.float(), (height, width), mode="bicubic", align_corners=False
            ).to(dtype=self.position_embedding.dtype)
        return pos.permute(0, 2, 3, 1).reshape(1, height * width, self.config.hidden_size)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def forward(self, latent: Tensor, timestep: Tensor, labels: Tensor) -> DiTOutput:
        if latent.ndim != 4 or latent.shape[1] != self.config.in_channels:
            raise ValueError("latent must have shape [B, configured in_channels, H, W]")
        batch, _, height, width = latent.shape
        tokens = latent.flatten(2).transpose(1, 2)
        x = self.input_projection(tokens)
        x = x + self.interpolated_position_embedding(height, width).to(dtype=x.dtype)
        conditioning = self.time_embedding(timestep) + self.class_embedding(labels)

        sources = math.ceil(self.config.depth / self.config.depth_attention.stride)
        bank_shape = (
            batch,
            sources,
            self.config.num_heads,
            height * width,
            self.config.hidden_size // self.config.num_heads,
        )
        key_bank = x.new_zeros(bank_shape)
        value_bank = x.new_zeros(bank_shape)
        bank_mask = torch.zeros(sources, device=x.device, dtype=torch.bool)
        early_tokens: Tensor | None = None

        for index, block in enumerate(self.blocks, start=1):
            args = (x, conditioning, height, width, key_bank, value_bank, bank_mask)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x, key_bank, value_bank, bank_mask = checkpoint(
                    block,
                    *args,
                    use_reentrant=False,
                    preserve_rng_state=self.config.dropout > 0,
                )
            else:
                x, key_bank, value_bank, bank_mask = block(*args)
            if index == self.config.early_exit_block:
                early_tokens = x
        if early_tokens is None:
            raise RuntimeError("early prediction block was not reached")

        def unpatchify(tokens_out: Tensor) -> Tensor:
            return tokens_out.transpose(1, 2).reshape(batch, self.config.in_channels, height, width)

        return DiTOutput(
            full=unpatchify(self.full_head(x, conditioning)),
            early=unpatchify(self.early_head(early_tokens, conditioning)),
        )
