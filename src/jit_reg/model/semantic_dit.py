from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..config import DiTConfig
from .layers import RMSNorm
from .rope import apply_rope_cos_sin, precompute_rope_tables


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


def _axis_sincos(positions: Tensor, width: int) -> Tensor:
    if width % 2:
        raise ValueError("sin-cos embedding width must be even")
    frequencies = torch.exp(
        -math.log(10_000)
        * torch.arange(width // 2, dtype=torch.float32)
        / max(width // 2, 1)
    )
    angles = torch.outer(positions.float(), frequencies)
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


def _positional_embedding(side: int, width: int) -> Tensor:
    if width % 4:
        raise ValueError("2D positional embedding width must be divisible by four")
    y = torch.arange(side, dtype=torch.float32)[:, None].expand(side, side).reshape(-1)
    x = torch.arange(side, dtype=torch.float32)[None, :].expand(side, side).reshape(-1)
    return torch.cat((_axis_sincos(y, width // 2), _axis_sincos(x, width // 2)), dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, width: int, frequency_width: int = 256) -> None:
        super().__init__()
        self.frequency_width = frequency_width
        self.mlp = nn.Sequential(
            nn.Linear(frequency_width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def _frequency_embedding(self, timestep: Tensor) -> Tensor:
        half = self.frequency_width // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / max(half, 1)
        )
        angles = timestep.float()[:, None] * frequencies[None]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if self.frequency_width % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def forward(self, timestep: Tensor) -> Tensor:
        return self.mlp(self._frequency_embedding(timestep))


class Attention(nn.Module):
    def __init__(self, width: int, num_heads: int, spatial_size: int) -> None:
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv = nn.Linear(width, 3 * width, bias=True)
        self.projection = nn.Linear(width, width, bias=True)
        for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            cos, sin = precompute_rope_tables(
                spatial_size, spatial_size, self.head_dim, dtype=dtype
            )
            self.register_buffer(f"rope_cos_{name}", cos, persistent=False)
            self.register_buffer(f"rope_sin_{name}", sin, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, _width = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        suffix = "bf16" if q.dtype == torch.bfloat16 else "fp32"
        cos = getattr(self, f"rope_cos_{suffix}")
        sin = getattr(self, f"rope_sin_{suffix}")
        if cos.dtype != q.dtype:
            cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q, k = apply_rope_cos_sin(q, k, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.projection(
            attended.transpose(1, 2).reshape(batch, tokens, self.width)
        )


class SwiGLU(nn.Module):
    def __init__(self, width: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = max(1, int(2 * width * mlp_ratio / 3))
        self.input_projection = nn.Linear(width, 2 * hidden)
        self.output_projection = nn.Linear(hidden, width)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.input_projection(x).chunk(2, dim=-1)
        return self.output_projection(F.silu(gate) * value)


class SemanticDiTBlock(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        mlp_ratio: float,
        spatial_size: int,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attention = Attention(width, num_heads, spatial_size)
        self.norm2 = RMSNorm(width)
        self.feed_forward = SwiGLU(width, mlp_ratio)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(
            conditioning
        ).chunk(6, dim=-1)
        x = x + gate_a * self.attention(_modulate(self.norm1(x), shift_a, scale_a))
        return x + gate_m * self.feed_forward(
            _modulate(self.norm2(x), shift_m, scale_m)
        )


class FinalLayer(nn.Module):
    def __init__(self, width: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm(width)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.projection = nn.Linear(width, patch_size**2 * out_channels)

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale = self.modulation(conditioning).chunk(2, dim=-1)
        return self.projection(_modulate(self.norm(x), shift, scale))


class SemanticDiT(nn.Module):
    """Gaussian endpoint DiT conditioned by aligned semantic-token AdaLN."""

    def __init__(self, config: DiTConfig | None = None) -> None:
        super().__init__()
        self.config = config or DiTConfig()
        self.config.validate()
        self.in_channels = self.config.in_channels
        side = self.config.grid_size
        self.endpoint_projection = nn.Conv2d(
            self.in_channels,
            self.config.hidden_size,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        self.semantic_projection = nn.Conv2d(
            self.in_channels,
            self.config.hidden_size,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        self.register_buffer(
            "positional_embedding",
            _positional_embedding(side, self.config.hidden_size).unsqueeze(0),
            persistent=True,
        )
        self.timestep_embedder = TimestepEmbedder(self.config.hidden_size)
        self.blocks = nn.ModuleList(
            SemanticDiTBlock(
                self.config.hidden_size,
                self.config.num_heads,
                self.config.mlp_ratio,
                side,
            )
            for _ in range(self.config.depth)
        )
        self.final_layer = FinalLayer(
            self.config.hidden_size, self.config.patch_size, self.in_channels
        )
        self.gradient_checkpointing = self.config.gradient_checkpointing
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        for projection in (self.endpoint_projection, self.semantic_projection):
            nn.init.xavier_uniform_(projection.weight.flatten(1))
            nn.init.zeros_(projection.bias)
        for layer in (self.timestep_embedder.mlp[0], self.timestep_embedder.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.modulation[-1].weight)
        nn.init.zeros_(self.final_layer.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.projection.weight)
        nn.init.zeros_(self.final_layer.projection.bias)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def _autocast(self, x: Tensor):
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if x.device.type == "cuda"
            else nullcontext()
        )

    @staticmethod
    def _tokens(projection: nn.Conv2d, x: Tensor) -> Tensor:
        return projection(x).flatten(2).transpose(1, 2)

    def _unpatchify(self, tokens: Tensor) -> Tensor:
        side = self.config.grid_size
        patch = self.config.patch_size
        batch = tokens.shape[0]
        expected = (batch, side * side, patch**2 * self.in_channels)
        if tuple(tokens.shape) != expected:
            raise ValueError(f"DiT output tokens must have shape {expected}")
        return (
            tokens.reshape(batch, side, side, patch, patch, self.in_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, self.in_channels, side * patch, side * patch)
        )

    def _validate_inputs(
        self, noisy: Tensor, timestep: Tensor, semantic: Tensor
    ) -> None:
        if tuple(noisy.shape[1:]) != self.config.input_shape:
            raise ValueError(
                f"DiT noisy input must have shape [B, {self.config.input_shape}]"
            )
        if semantic.shape != noisy.shape:
            raise ValueError("semantic and noisy endpoint tensors must have matching shapes")
        if timestep.shape != (noisy.shape[0],):
            raise ValueError("timestep must have shape [B]")

    def _block(
        self, block: SemanticDiTBlock, states: Tensor, conditioning: Tensor
    ) -> Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, states, conditioning, use_reentrant=False)
        return block(states, conditioning)

    def forward(self, noisy: Tensor, timestep: Tensor, semantic: Tensor) -> Tensor:
        self._validate_inputs(noisy, timestep, semantic)
        with self._autocast(noisy):
            states = self._tokens(self.endpoint_projection, noisy)
            states = states + self.positional_embedding.to(states.dtype)
            conditioning = self._tokens(self.semantic_projection, semantic)
            conditioning = conditioning + self.timestep_embedder(timestep)[:, None]
            for block in self.blocks:
                states = self._block(block, states, conditioning)
            return self._unpatchify(self.final_layer(states, conditioning))
