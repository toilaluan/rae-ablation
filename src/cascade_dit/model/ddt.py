from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..config import GRID_SIZE, TOKEN_COUNT, ModelConfig
from .rope import apply_rope_cos_sin, precompute_rope_tables


def _modulate(values: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return values * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, values: Tensor) -> Tensor:
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(values.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width: int, ratio: float) -> None:
        super().__init__()
        hidden = max(1, int(2 / 3 * width * ratio))
        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.output = nn.Linear(hidden, width)

    def forward(self, values: Tensor) -> Tensor:
        return self.output(F.silu(self.gate(values)) * self.value(values))


class Attention(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        *,
        token_count: int,
        spatial_tokens: int,
        qk_norm: bool,
        rope: bool,
        sdpa: bool,
    ) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.sdpa = sdpa
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.query_norm = RMSNorm(self.head_width) if qk_norm else nn.Identity()
        self.key_norm = RMSNorm(self.head_width) if qk_norm else nn.Identity()
        if rope:
            cosine, sine = precompute_rope_tables(
                GRID_SIZE, GRID_SIZE, self.head_width, dtype=torch.float32
            )
            suffix = token_count - spatial_tokens
            if suffix:
                cosine = torch.cat((cosine, torch.ones(suffix, self.head_width)))
                sine = torch.cat((sine, torch.zeros(suffix, self.head_width)))
        else:
            cosine = sine = torch.empty(0)
        self.register_buffer("rope_cosine", cosine, persistent=False)
        self.register_buffer("rope_sine", sine, persistent=False)

    def forward(self, values: Tensor) -> Tensor:
        batch, token_count, _ = values.shape
        shape = (batch, token_count, self.heads, self.head_width)
        query = self.query(values).reshape(shape).transpose(1, 2)
        key = self.key(values).reshape(shape).transpose(1, 2)
        value = self.value(values).reshape(shape).transpose(1, 2)
        query, key = self.query_norm(query), self.key_norm(key)
        if self.rope_cosine.numel():
            query, key = apply_rope_cos_sin(
                query,
                key,
                self.rope_cosine.to(query),
                self.rope_sine.to(query),
            )
        if self.sdpa:
            attended = F.scaled_dot_product_attention(query, key, value)
        else:
            weights = (query * self.head_width**-0.5) @ key.transpose(-2, -1)
            attended = weights.softmax(dim=-1) @ value
        return self.output(
            attended.transpose(1, 2).reshape(batch, token_count, self.width)
        )


class DDTEncoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, global_tokens: int) -> None:
        super().__init__()
        path = config.encoder
        self.norm_attention = RMSNorm(path.width)
        self.attention = Attention(
            path.width,
            path.heads,
            token_count=TOKEN_COUNT + global_tokens,
            spatial_tokens=TOKEN_COUNT,
            qk_norm=config.attention.qk_norm,
            rope=config.attention.rope,
            sdpa=config.attention.sdpa,
        )
        self.norm_mlp = RMSNorm(path.width)
        self.mlp = SwiGLU(path.width, config.feedforward.mlp_ratio)

    def forward(self, values: Tensor) -> Tensor:
        values = values + self.attention(self.norm_attention(values))
        return values + self.mlp(self.norm_mlp(values))


class DDTDecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        path = config.decoder
        self.norm_attention = RMSNorm(path.width)
        self.attention = Attention(
            path.width,
            path.heads,
            token_count=TOKEN_COUNT,
            spatial_tokens=TOKEN_COUNT,
            qk_norm=config.attention.qk_norm,
            rope=config.attention.rope,
            sdpa=config.attention.sdpa,
        )
        self.norm_mlp = RMSNorm(path.width)
        self.mlp = SwiGLU(path.width, config.feedforward.mlp_ratio)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(path.width, 6 * path.width))

    def forward(self, values: Tensor, conditioning: Tensor) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaln(
            conditioning
        ).chunk(6, dim=-1)
        values = values + gate_a * self.attention(
            _modulate(self.norm_attention(values), shift_a, scale_a)
        )
        return values + gate_m * self.mlp(
            _modulate(self.norm_mlp(values), shift_m, scale_m)
        )


class DDTFinalLayer(nn.Module):
    def __init__(self, width: int, output_width: int) -> None:
        super().__init__()
        self.norm = RMSNorm(width)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.projection = nn.Linear(width, output_width)

    def forward(self, values: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale = self.adaln(conditioning).chunk(2, dim=-1)
        return self.projection(_modulate(self.norm(values), shift, scale))


class TimestepEmbedder(nn.Module):
    def __init__(self, width: int, token_count: int, frequency_width: int = 256) -> None:
        super().__init__()
        self.register_buffer(
            "frequencies", torch.randn(frequency_width), persistent=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * frequency_width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.tokens = nn.Parameter(torch.randn(token_count, width) / width**0.5)

    def forward(self, timestep: Tensor) -> tuple[Tensor, Tensor]:
        angles = timestep.float()[:, None] * self.frequencies[None] * (2 * torch.pi)
        base = self.mlp(torch.cat((angles.sin(), angles.cos()), dim=-1))[:, None]
        return base, base + self.tokens


class LabelEmbedder(nn.Module):
    def __init__(
        self,
        width: int,
        num_classes: int,
        token_count: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_probability = dropout_probability
        self.embedding = nn.Embedding(num_classes + 1, width)
        self.tokens = nn.Parameter(torch.randn(token_count, width) / width**0.5)

    def forward(self, labels: Tensor, force_drop: Tensor | None = None) -> Tensor:
        if labels.ndim != 1 or torch.any(labels < 0) or torch.any(labels > self.num_classes):
            raise ValueError("class labels are outside the configured range")
        if force_drop is None and self.training and self.dropout_probability:
            force_drop = torch.rand(labels.shape, device=labels.device) < self.dropout_probability
        if force_drop is not None:
            if force_drop.shape != labels.shape:
                raise ValueError("force_class_drop must have shape [B]")
            labels = torch.where(force_drop.bool(), self.num_classes, labels)
        return self.embedding(labels)[:, None] + self.tokens


class RepresentationEmbedder(nn.Module):
    def __init__(self, levels: int, width: int) -> None:
        super().__init__()
        self.levels = levels
        self.embedding = nn.Parameter(torch.randn(levels, width) / width**0.5)

    def forward(self, weights: Tensor) -> Tensor:
        if weights.ndim != 2 or weights.shape[1] != self.levels:
            raise ValueError(f"R must have shape [B,{self.levels}]")
        if not torch.compiler.is_compiling():
            if not torch.isfinite(weights).all() or torch.any(weights < 0):
                raise ValueError("R must be finite and non-negative")
            sums = weights.float().sum(dim=-1)
            if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5):
                raise ValueError("every R row must sum to one")
        return weights.to(self.embedding) @ self.embedding


class DDT(nn.Module):
    """Universal class-conditional DDT over an 8x8x768 representation grid."""

    def __init__(self, config: ModelConfig, representation_levels: int) -> None:
        super().__init__()
        if representation_levels < 2:
            raise ValueError("DDT requires at least two representation levels")
        self.config = config
        self.representation_levels = representation_levels
        encoder_width = config.encoder.width
        decoder_width = config.decoder.width
        conditioning = config.conditioning
        global_tokens = conditioning.timestep_tokens + conditioning.class_tokens + 1

        self.source_projection = nn.Linear(config.token_dim, encoder_width)
        self.target_projection = nn.Linear(config.token_dim, decoder_width)
        self.timestep_embedder = TimestepEmbedder(
            encoder_width, conditioning.timestep_tokens
        )
        self.class_embedder = LabelEmbedder(
            encoder_width,
            conditioning.num_classes,
            conditioning.class_tokens,
            conditioning.class_dropout_probability,
        )
        self.representation_embedder = RepresentationEmbedder(
            representation_levels, encoder_width
        )
        self.encoder_blocks = nn.ModuleList(
            DDTEncoderBlock(config, global_tokens) for _ in range(config.encoder.depth)
        )
        self.encoder_to_decoder = (
            nn.Linear(encoder_width, decoder_width)
            if encoder_width != decoder_width
            else nn.Identity()
        )
        self.decoder_blocks = nn.ModuleList(
            DDTDecoderBlock(config) for _ in range(config.decoder.depth)
        )
        self.final_layer = DDTFinalLayer(decoder_width, config.token_dim)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        nn.init.normal_(self.class_embedder.embedding.weight, std=0.02)
        for layer in (self.timestep_embedder.mlp[0], self.timestep_embedder.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
        for block in self.decoder_blocks:
            nn.init.zeros_(block.adaln[-1].weight)
            nn.init.zeros_(block.adaln[-1].bias)
        nn.init.zeros_(self.final_layer.adaln[-1].weight)
        nn.init.zeros_(self.final_layer.adaln[-1].bias)
        nn.init.zeros_(self.final_layer.projection.weight)
        nn.init.zeros_(self.final_layer.projection.bias)

    def _encoder_block(self, block: DDTEncoderBlock, values: Tensor) -> Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, values, use_reentrant=False)
        return block(values)

    def _decoder_block(
        self, block: DDTDecoderBlock, values: Tensor, conditioning: Tensor
    ) -> Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, values, conditioning, use_reentrant=False)
        return block(values, conditioning)

    @staticmethod
    def _autocast(values: Tensor):
        if values.device.type == "cuda" and not torch.is_autocast_enabled():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def forward(
        self,
        noisy_tokens: Tensor,
        timestep: Tensor,
        labels: Tensor,
        representation_weights: Tensor,
        *,
        force_class_drop: Tensor | None = None,
    ) -> Tensor:
        if noisy_tokens.ndim != 3 or noisy_tokens.shape[1:] != (
            TOKEN_COUNT,
            self.config.token_dim,
        ):
            raise ValueError("DDT input must have shape [B,64,768]")
        batch = noisy_tokens.shape[0]
        if timestep.shape != (batch,) or labels.shape != (batch,):
            raise ValueError("timestep and labels must have shape [B]")
        if representation_weights.shape[0] != batch:
            raise ValueError("R batch dimension must match noisy tokens")

        with self._autocast(noisy_tokens):
            representation = self.representation_embedder(representation_weights)[:, None]
            timestep_base, timestep_tokens = self.timestep_embedder(timestep)
            class_tokens = self.class_embedder(labels, force_class_drop)
            source = self.source_projection(noisy_tokens) + representation
            states = torch.cat(
                (source, timestep_tokens, class_tokens, representation), dim=1
            )
            for block in self.encoder_blocks:
                states = self._encoder_block(block, states)
            spatial = states[:, :TOKEN_COUNT]
            conditioning = self.encoder_to_decoder(
                F.silu(spatial + timestep_base + representation)
            )
            decoded = self.target_projection(noisy_tokens)
            for block in self.decoder_blocks:
                decoded = self._decoder_block(block, decoded, conditioning)
            return self.final_layer(decoded, conditioning)


def build_model(config: ModelConfig, representation_levels: int) -> DDT:
    return DDT(config, representation_levels)
