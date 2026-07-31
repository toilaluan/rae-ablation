from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig


def classifier_free_labels(
    labels: Tensor,
    num_classes: int,
    dropout_probability: float,
    generator: torch.Generator,
) -> Tensor:
    if labels.ndim != 1:
        raise ValueError("labels must have shape [B]")
    if dropout_probability == 0:
        return labels
    dropped = (
        torch.rand(labels.shape, device=labels.device, generator=generator) < dropout_probability
    )
    return torch.where(dropped, torch.full_like(labels, num_classes), labels)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


def _rotate_pairs(x: Tensor) -> Tensor:
    paired = x.float().reshape(*x.shape[:-1], -1, 2)
    rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def _rope_tables(grid_size: int, head_dim: int, theta: float = 10_000.0) -> tuple[Tensor, Tensor]:
    if head_dim % 4:
        raise ValueError("head dimension must be divisible by four for factorized 2D RoPE")
    axis_dim = head_dim // 2
    frequencies = theta ** (-torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim)
    rows = torch.arange(grid_size, dtype=torch.float32).repeat_interleave(grid_size)
    columns = torch.arange(grid_size, dtype=torch.float32).repeat(grid_size)

    def angles(position: Tensor) -> Tensor:
        return torch.outer(position, frequencies).repeat_interleave(2, dim=-1)

    values = torch.cat((angles(rows), angles(columns)), dim=-1)
    return values.cos(), values.sin()


class TimestepEmbedder(nn.Module):
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
            -math.log(10_000.0)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / half
        )
        angles = (1_000 * timestep.float()).unsqueeze(-1) * frequencies
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.mlp(embedding)


class FusedSwiGLU(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        intermediate = int(2 * hidden_size * mlp_ratio / 3)
        self.intermediate_size = intermediate
        self.gate_value = nn.Linear(hidden_size, 2 * intermediate)
        self.projection = nn.Linear(intermediate, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.gate_value(x).chunk(2, dim=-1)
        return self.projection(F.silu(gate) * value)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, grid_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.projection = nn.Linear(hidden_size, hidden_size)
        cos, sin = _rope_tables(grid_size, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, hidden = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = self.q_norm(query)
        key = self.k_norm(key)
        cos = self.rope_cos[:tokens].to(device=x.device, dtype=query.dtype)[None, :, None, :]
        sin = self.rope_sin[:tokens].to(device=x.device, dtype=query.dtype)[None, :, None, :]
        query = query * cos + _rotate_pairs(query) * sin
        key = key * cos + _rotate_pairs(key) * sin
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        return self.projection(attended.transpose(1, 2).reshape(batch, tokens, hidden))


class SiTBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6, elementwise_affine=False)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=1e-6, elementwise_affine=False)
        self.attention = Attention(config.hidden_size, config.num_heads, config.grid_size)
        self.mlp = FusedSwiGLU(config.hidden_size, config.mlp_ratio)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size),
        )

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.modulation(
            condition
        ).chunk(6, dim=-1)
        x = x + gate_attn * self.attention(
            _modulate(self.attention_norm(x), shift_attn, scale_attn)
        )
        return x + gate_mlp * self.mlp(_modulate(self.mlp_norm(x), shift_mlp, scale_mlp))


class FinalLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 2 * config.hidden_size),
        )
        self.projection = nn.Linear(
            config.hidden_size,
            config.patch_size**2 * config.in_channels,
        )

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.projection(_modulate(self.norm(x), shift, scale))


@dataclass(frozen=True)
class SiTOutput:
    x0: Tensor | None
    features: tuple[Tensor, ...]


class SiT(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        align_layers: list[int] | None = None,
        projector_hidden_size: int = 1536,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.time_embed = TimestepEmbedder(config.hidden_size)
        self.class_embed = nn.Embedding(config.num_classes + 1, config.hidden_size)
        self.blocks = nn.ModuleList(SiTBlock(config) for _ in range(config.depth))
        self.final = FinalLayer(config)
        self.alignment_projectors = AlignmentProjectors(
            align_layers or [], config.hidden_size, projector_hidden_size
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        patch_weight = self.patch_embed.weight.data
        nn.init.xavier_uniform_(patch_weight.reshape(patch_weight.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final.modulation[-1].weight)
        nn.init.zeros_(self.final.modulation[-1].bias)
        nn.init.zeros_(self.final.projection.weight)
        nn.init.zeros_(self.final.projection.bias)

    def _condition(self, timestep: Tensor, labels: Tensor, tokens: int) -> Tensor:
        if timestep.ndim == 1:
            timestep = timestep[:, None].expand(-1, tokens)
        if timestep.shape != (labels.shape[0], tokens):
            raise ValueError(f"timestep must have shape [B] or [B, {tokens}]")
        return self.time_embed(timestep) + self.class_embed(labels)[:, None, :]

    def _unpatchify(self, tokens: Tensor) -> Tensor:
        batch = tokens.shape[0]
        grid = self.config.grid_size
        patch = self.config.patch_size
        channels = self.config.in_channels
        return (
            tokens.reshape(batch, grid, grid, patch, patch, channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, channels, grid * patch, grid * patch)
        )

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        labels: Tensor,
        *,
        capture_layers: tuple[int, ...] = (),
        stop_at_layer: int | None = None,
        project_features: bool = False,
    ) -> SiTOutput:
        expected = (x.shape[0], *self.config.input_shape)
        if x.ndim != 4 or tuple(x.shape) != expected:
            raise ValueError(f"input must have shape {expected}")
        if labels.shape != (x.shape[0],):
            raise ValueError("labels must have shape [B]")
        if capture_layers != tuple(sorted(set(capture_layers))):
            raise ValueError("capture_layers must be sorted and unique")
        final_block = self.config.depth if stop_at_layer is None else stop_at_layer
        if not 1 <= final_block <= self.config.depth:
            raise ValueError("stop_at_layer must be within model depth")
        if any(layer < 1 or layer > final_block for layer in capture_layers):
            raise ValueError("captured layers must not exceed the executed depth")
        if project_features and capture_layers != self.alignment_projectors.layers:
            raise ValueError("projected captures must match configured alignment layers")
        state = self.patch_embed(x).flatten(2).transpose(1, 2)
        condition = self._condition(timestep, labels, state.shape[1])
        captured: list[Tensor] = []
        capture_set = set(capture_layers)
        for layer, block in enumerate(self.blocks[:final_block], start=1):
            if self.config.gradient_checkpointing and self.training:
                state = checkpoint(block, state, condition, use_reentrant=False)
            else:
                state = block(state, condition)
            if layer in capture_set:
                captured.append(state)
        prediction = None
        if stop_at_layer is None:
            prediction = self._unpatchify(self.final(state, condition))
        features = tuple(captured)
        if project_features:
            features = self.alignment_projectors(features)
        return SiTOutput(prediction, features)


class AlignmentProjectors(nn.Module):
    def __init__(self, layers: list[int], hidden_size: int, projector_hidden_size: int) -> None:
        super().__init__()
        self.layers = tuple(layers)
        self.projectors = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, projector_hidden_size),
                nn.SiLU(),
                nn.Linear(projector_hidden_size, hidden_size),
            )
            for _ in layers
        )

    def forward(self, features: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        if len(features) != len(self.projectors):
            raise ValueError("feature count does not match alignment projectors")
        return tuple(
            projector(feature) for projector, feature in zip(self.projectors, features, strict=True)
        )
