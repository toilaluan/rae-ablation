from __future__ import annotations

from contextlib import nullcontext
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..config import (
    IMAGE_LATENT_CHANNELS,
    IMAGE_LATENT_SHAPE,
    SEMANTIC_CHANNELS,
    SEMANTIC_TOKEN_SHAPE,
    ModelConfig,
)
from .rope import apply_rope_cos_sin, precompute_rope_tables

SPATIAL_TOKENS = 64
TOKEN_GRID_SIZE = 8


class SemanticPredictions(NamedTuple):
    full: Tensor
    base: Tensor


class CascadePredictions(NamedTuple):
    semantic: SemanticPredictions
    image_latent: Tensor


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
    def __init__(self, width: int, heads: int, global_tokens: int = 0) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.query_norm = RMSNorm(self.head_width)
        self.key_norm = RMSNorm(self.head_width)
        for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            cosine, sine = precompute_rope_tables(
                TOKEN_GRID_SIZE,
                TOKEN_GRID_SIZE,
                self.head_width,
                dtype=dtype,
            )
            if global_tokens:
                cosine = torch.cat(
                    (cosine, torch.ones(global_tokens, self.head_width, dtype=dtype))
                )
                sine = torch.cat(
                    (sine, torch.zeros(global_tokens, self.head_width, dtype=dtype))
                )
            self.register_buffer(f"rope_cosine_{name}", cosine, persistent=False)
            self.register_buffer(f"rope_sine_{name}", sine, persistent=False)

    def forward(self, values: Tensor) -> Tensor:
        batch, token_count, _width = values.shape
        shape = (batch, token_count, self.heads, self.head_width)
        query = self.query(values).reshape(shape).transpose(1, 2)
        key = self.key(values).reshape(shape).transpose(1, 2)
        value = self.value(values).reshape(shape).transpose(1, 2)
        query, key = self.query_norm(query), self.key_norm(key)
        suffix = "bf16" if query.dtype == torch.bfloat16 else "fp32"
        cosine = getattr(self, f"rope_cosine_{suffix}")
        sine = getattr(self, f"rope_sine_{suffix}")
        if cosine.dtype != query.dtype:
            cosine, sine = cosine.to(query.dtype), sine.to(query.dtype)
        query, key = apply_rope_cos_sin(query, key, cosine, sine)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=False)
        return self.output(
            attended.transpose(1, 2).reshape(batch, token_count, self.width)
        )


class DDTEncoderBlock(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        mlp_ratio: float,
        global_tokens: int,
    ) -> None:
        super().__init__()
        self.norm_attention = RMSNorm(width)
        self.attention = Attention(width, heads, global_tokens)
        self.norm_mlp = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_ratio)

    def forward(self, values: Tensor) -> Tensor:
        values = values + self.attention(self.norm_attention(values))
        return values + self.mlp(self.norm_mlp(values))


class DDTDecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm_attention = RMSNorm(width)
        self.attention = Attention(width, heads)
        self.norm_mlp = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_ratio)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(width, 6 * width)
        )
        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)

    def forward(self, values: Tensor, conditioning: Tensor) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaln_modulation(
            conditioning
        ).chunk(6, dim=-1)
        values = values + gate_a * self.attention(
            _modulate(self.norm_attention(values), shift_a, scale_a)
        )
        return values + gate_m * self.mlp(
            _modulate(self.norm_mlp(values), shift_m, scale_m)
        )


class DDTFinalLayer(nn.Module):
    def __init__(self, width: int, channels: int, patch_size: int) -> None:
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.norm = RMSNorm(width)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.projection = nn.Linear(width, channels * patch_size**2)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, values: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale = self.modulation(conditioning).chunk(2, dim=-1)
        patches = self.projection(_modulate(self.norm(values), shift, scale))
        batch = patches.shape[0]
        patch = self.patch_size
        return (
            patches.reshape(
                batch,
                TOKEN_GRID_SIZE,
                TOKEN_GRID_SIZE,
                patch,
                patch,
                self.channels,
            )
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(
                batch,
                self.channels,
                TOKEN_GRID_SIZE * patch,
                TOKEN_GRID_SIZE * patch,
            )
        )


def _patchify(values: Tensor, patch_size: int) -> Tensor:
    batch, channels, _height, _width = values.shape
    return (
        values.reshape(
            batch,
            channels,
            TOKEN_GRID_SIZE,
            patch_size,
            TOKEN_GRID_SIZE,
            patch_size,
        )
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, SPATIAL_TOKENS, channels * patch_size**2)
    )


class RoutePatchEmbedder(nn.Module):
    """Patch, align, concatenate, and project one route's representations."""

    def __init__(
        self,
        inputs: tuple[tuple[int, int], ...],
        width: int,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        input_width = sum(channels * patch**2 for channels, patch in inputs)
        self.projection = nn.Linear(input_width, width)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, *values: Tensor) -> Tensor:
        if len(values) != len(self.inputs):
            raise ValueError("route embedder received the wrong number of representations")
        patches = [
            _patchify(value, patch_size)
            for value, (_channels, patch_size) in zip(
                values, self.inputs, strict=True
            )
        ]
        return self.projection(torch.cat(patches, dim=-1))


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
        for layer in (self.mlp[0], self.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)

    def forward(self, timestep: Tensor) -> tuple[Tensor, Tensor]:
        angles = timestep.float()[:, None] * self.frequencies[None] * (2 * torch.pi)
        base = self.mlp(torch.cat((angles.sin(), angles.cos()), dim=-1))[:, None]
        return base, base + self.tokens


class LabelEmbedder(nn.Module):
    def __init__(self, width: int, num_classes: int, token_count: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, width)
        self.tokens = nn.Parameter(torch.randn(token_count, width) / width**0.5)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, labels: Tensor) -> Tensor:
        return self.embedding(labels)[:, None] + self.tokens


def _validate_common(
    values: Tensor,
    timestep: Tensor,
    labels: Tensor,
    shape: tuple[int, ...],
) -> None:
    if tuple(values.shape[1:]) != shape:
        raise ValueError(f"input must have shape [B,{','.join(map(str, shape))}]")
    if timestep.shape != (values.shape[0],) or labels.shape != (values.shape[0],):
        raise ValueError("timestep and labels must have shape [B]")


def _validate_semantic_condition(image_latents: Tensor, semantic_tokens: Tensor) -> None:
    if tuple(semantic_tokens.shape) != (
        image_latents.shape[0],
        *SEMANTIC_TOKEN_SHAPE,
    ):
        raise ValueError("semantic_tokens must have shape [B,768,8,8]")


class _DDTCore(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        encoder_width, decoder_width = config.hidden_size
        encoder_depth, decoder_depth = config.depth
        encoder_heads, decoder_heads = config.num_heads
        global_tokens = config.num_timestep_tokens + config.num_class_tokens
        self.timestep_embedder = TimestepEmbedder(
            encoder_width, config.num_timestep_tokens
        )
        self.class_embedder = LabelEmbedder(
            encoder_width, config.num_classes, config.num_class_tokens
        )
        self.encoder_blocks = nn.ModuleList(
            DDTEncoderBlock(
                encoder_width,
                encoder_heads,
                config.mlp_ratio,
                global_tokens,
            )
            for _ in range(encoder_depth)
        )
        self.encoder_to_decoder = (
            nn.Linear(encoder_width, decoder_width)
            if encoder_width != decoder_width
            else nn.Identity()
        )
        self.decoder_blocks = nn.ModuleList(
            DDTDecoderBlock(decoder_width, decoder_heads, config.mlp_ratio)
            for _ in range(decoder_depth)
        )
        self.gradient_checkpointing = config.gradient_checkpointing

    def _global_tokens(
        self, timestep: Tensor, labels: Tensor
    ) -> tuple[Tensor, Tensor]:
        timestep_base, timestep_tokens = self.timestep_embedder(timestep)
        return timestep_base, torch.cat(
            (timestep_tokens, self.class_embedder(labels)), dim=1
        )

    def _encoder_block(self, block: DDTEncoderBlock, values: Tensor) -> Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, values, use_reentrant=False)
        return block(values)

    def _decoder_block(
        self,
        block: DDTDecoderBlock,
        values: Tensor,
        conditioning: Tensor,
    ) -> Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, values, conditioning, use_reentrant=False)
        return block(values, conditioning)

    def _encode(
        self,
        patches: Tensor,
        timestep: Tensor,
        labels: Tensor,
        *,
        capture_depth: int | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        timestep_base, global_tokens = self._global_tokens(timestep, labels)
        states = torch.cat((patches, global_tokens), dim=1)
        captured: Tensor | None = None
        for index, block in enumerate(self.encoder_blocks, start=1):
            states = self._encoder_block(block, states)
            if index == capture_depth:
                captured = states[:, :SPATIAL_TOKENS]
        spatial = states[:, :SPATIAL_TOKENS]
        conditioning = self.encoder_to_decoder(F.silu(timestep_base + spatial))
        return conditioning, captured, timestep_base

    def _decode(self, patches: Tensor, conditioning: Tensor) -> Tensor:
        states = patches
        for block in self.decoder_blocks:
            states = self._decoder_block(block, states, conditioning)
        return states

    @staticmethod
    def _autocast(values: Tensor):
        if values.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()


class LatentDDT(_DDTCore):
    """Pure native Mage image-latent DDT baseline."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        config = config or ModelConfig(architecture="latent")
        if config.architecture != "latent":
            raise ValueError("LatentDDT requires model.architecture=latent")
        config.validate()
        super().__init__(config)
        encoder_width, decoder_width = config.hidden_size
        inputs = ((IMAGE_LATENT_CHANNELS, config.image_latent_patch_size),)
        self.source_embedder = RoutePatchEmbedder(inputs, encoder_width)
        self.target_embedder = RoutePatchEmbedder(inputs, decoder_width)
        self.image_latent_head = DDTFinalLayer(
            decoder_width,
            IMAGE_LATENT_CHANNELS,
            config.image_latent_patch_size,
        )

    def forward(self, image_latents: Tensor, timestep: Tensor, labels: Tensor) -> Tensor:
        _validate_common(image_latents, timestep, labels, IMAGE_LATENT_SHAPE)
        with self._autocast(image_latents):
            conditioning, _captured, _timestep_base = self._encode(
                self.source_embedder(image_latents), timestep, labels
            )
            decoded = self._decode(self.target_embedder(image_latents), conditioning)
            return self.image_latent_head(decoded, conditioning)

    forward_image_prediction = forward


class SemanticDDT(_DDTCore):
    def __init__(self, config: ModelConfig | None = None) -> None:
        config = config or ModelConfig(architecture="decoupled_cascade")
        config.validate()
        super().__init__(config)
        encoder_width, decoder_width = config.hidden_size
        inputs = ((SEMANTIC_CHANNELS, config.semantic_patch_size),)
        self.source_embedder = RoutePatchEmbedder(inputs, encoder_width)
        self.target_embedder = RoutePatchEmbedder(inputs, decoder_width)
        self.semantic_full_head = DDTFinalLayer(
            decoder_width, SEMANTIC_CHANNELS, config.semantic_patch_size
        )
        self.semantic_base_head = DDTFinalLayer(
            encoder_width, SEMANTIC_CHANNELS, config.semantic_patch_size
        )

    def forward(
        self, semantic_tokens: Tensor, timestep: Tensor, labels: Tensor
    ) -> SemanticPredictions:
        _validate_common(semantic_tokens, timestep, labels, SEMANTIC_TOKEN_SHAPE)
        with self._autocast(semantic_tokens):
            conditioning, base_states, timestep_base = self._encode(
                self.source_embedder(semantic_tokens),
                timestep,
                labels,
                capture_depth=self.config.base_model_depth,
            )
            assert base_states is not None
            decoded = self._decode(self.target_embedder(semantic_tokens), conditioning)
            base_conditioning = F.silu(timestep_base + base_states)
            return SemanticPredictions(
                self.semantic_full_head(decoded, conditioning),
                self.semantic_base_head(base_conditioning, base_conditioning),
            )

    forward_semantic_predictions = forward


class ConditionalLatentDDT(_DDTCore):
    def __init__(self, config: ModelConfig | None = None) -> None:
        config = config or ModelConfig(architecture="decoupled_cascade")
        config.validate()
        super().__init__(config)
        encoder_width, decoder_width = config.hidden_size
        inputs = (
            (IMAGE_LATENT_CHANNELS, config.image_latent_patch_size),
            (SEMANTIC_CHANNELS, config.semantic_patch_size),
        )
        self.source_embedder = RoutePatchEmbedder(inputs, encoder_width)
        self.target_embedder = RoutePatchEmbedder(inputs, decoder_width)
        self.image_latent_head = DDTFinalLayer(
            decoder_width,
            IMAGE_LATENT_CHANNELS,
            config.image_latent_patch_size,
        )

    def forward(
        self,
        image_latents: Tensor,
        timestep: Tensor,
        labels: Tensor,
        semantic_tokens: Tensor,
    ) -> Tensor:
        _validate_common(image_latents, timestep, labels, IMAGE_LATENT_SHAPE)
        _validate_semantic_condition(image_latents, semantic_tokens)
        with self._autocast(image_latents):
            conditioning, _captured, _timestep_base = self._encode(
                self.source_embedder(image_latents, semantic_tokens), timestep, labels
            )
            decoded = self._decode(
                self.target_embedder(image_latents, semantic_tokens), conditioning
            )
            return self.image_latent_head(decoded, conditioning)

    forward_image_prediction = forward


class SharedCascadeDDT(_DDTCore):
    """Semantic and fused image-latent routes around one shared DDT core."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        config = config or ModelConfig(architecture="shared_cascade")
        if config.architecture != "shared_cascade":
            raise ValueError(
                "SharedCascadeDDT requires model.architecture=shared_cascade"
            )
        config.validate()
        super().__init__(config)
        encoder_width, decoder_width = config.hidden_size
        semantic_inputs = ((SEMANTIC_CHANNELS, config.semantic_patch_size),)
        image_inputs = (
            (IMAGE_LATENT_CHANNELS, config.image_latent_patch_size),
            (SEMANTIC_CHANNELS, config.semantic_patch_size),
        )
        self.semantic_source_embedder = RoutePatchEmbedder(
            semantic_inputs, encoder_width
        )
        self.semantic_target_embedder = RoutePatchEmbedder(
            semantic_inputs, decoder_width
        )
        self.image_source_embedder = RoutePatchEmbedder(image_inputs, encoder_width)
        self.image_target_embedder = RoutePatchEmbedder(image_inputs, decoder_width)
        self.semantic_full_head = DDTFinalLayer(
            decoder_width, SEMANTIC_CHANNELS, config.semantic_patch_size
        )
        self.semantic_base_head = DDTFinalLayer(
            encoder_width, SEMANTIC_CHANNELS, config.semantic_patch_size
        )
        self.image_latent_head = DDTFinalLayer(
            decoder_width,
            IMAGE_LATENT_CHANNELS,
            config.image_latent_patch_size,
        )

    def forward_semantic_predictions(
        self, semantic_tokens: Tensor, timestep: Tensor, labels: Tensor
    ) -> SemanticPredictions:
        _validate_common(semantic_tokens, timestep, labels, SEMANTIC_TOKEN_SHAPE)
        with self._autocast(semantic_tokens):
            conditioning, base_states, timestep_base = self._encode(
                self.semantic_source_embedder(semantic_tokens),
                timestep,
                labels,
                capture_depth=self.config.base_model_depth,
            )
            assert base_states is not None
            decoded = self._decode(
                self.semantic_target_embedder(semantic_tokens), conditioning
            )
            base_conditioning = F.silu(timestep_base + base_states)
            return SemanticPredictions(
                self.semantic_full_head(decoded, conditioning),
                self.semantic_base_head(base_conditioning, base_conditioning),
            )

    def forward_image_prediction(
        self,
        image_latents: Tensor,
        timestep: Tensor,
        labels: Tensor,
        semantic_tokens: Tensor,
    ) -> Tensor:
        _validate_common(image_latents, timestep, labels, IMAGE_LATENT_SHAPE)
        _validate_semantic_condition(image_latents, semantic_tokens)
        with self._autocast(image_latents):
            conditioning, _captured, _timestep_base = self._encode(
                self.image_source_embedder(image_latents, semantic_tokens),
                timestep,
                labels,
            )
            decoded = self._decode(
                self.image_target_embedder(image_latents, semantic_tokens), conditioning
            )
            return self.image_latent_head(decoded, conditioning)

    def forward(
        self,
        noisy_semantic_tokens: Tensor,
        semantic_timestep: Tensor,
        semantic_labels: Tensor,
        noisy_image_latents: Tensor,
        image_timestep: Tensor,
        image_labels: Tensor,
        clean_semantic_tokens: Tensor,
        predicted_semantic_mask: Tensor,
    ) -> CascadePredictions:
        semantic = self.forward_semantic_predictions(
            noisy_semantic_tokens, semantic_timestep, semantic_labels
        )
        if predicted_semantic_mask.shape != (noisy_semantic_tokens.shape[0],):
            raise ValueError("predicted_semantic_mask must have shape [B]")
        semantic_condition = torch.where(
            predicted_semantic_mask[:, None, None, None],
            semantic.full.detach(),
            clean_semantic_tokens.detach(),
        )
        image_latent = self.forward_image_prediction(
            noisy_image_latents, image_timestep, image_labels, semantic_condition
        )
        return CascadePredictions(semantic, image_latent)


class DecoupledCascadeDDT(nn.Module):
    """Two independent DDT denoisers joined only by a semantic tensor."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig(architecture="decoupled_cascade")
        if config.architecture != "decoupled_cascade":
            raise ValueError(
                "DecoupledCascadeDDT requires model.architecture=decoupled_cascade"
            )
        config.validate()
        self.config = config
        self.semantic_ddt = SemanticDDT(config)
        self.image_latent_ddt = ConditionalLatentDDT(config)

    def forward_semantic_predictions(
        self, semantic_tokens: Tensor, timestep: Tensor, labels: Tensor
    ) -> SemanticPredictions:
        return self.semantic_ddt(semantic_tokens, timestep, labels)

    def forward_image_prediction(
        self,
        image_latents: Tensor,
        timestep: Tensor,
        labels: Tensor,
        semantic_tokens: Tensor,
    ) -> Tensor:
        return self.image_latent_ddt(
            image_latents, timestep, labels, semantic_tokens
        )

    def forward(
        self,
        noisy_semantic_tokens: Tensor,
        semantic_timestep: Tensor,
        semantic_labels: Tensor,
        noisy_image_latents: Tensor,
        image_timestep: Tensor,
        image_labels: Tensor,
        clean_semantic_tokens: Tensor,
        predicted_semantic_mask: Tensor,
    ) -> CascadePredictions:
        semantic = self.forward_semantic_predictions(
            noisy_semantic_tokens, semantic_timestep, semantic_labels
        )
        if predicted_semantic_mask.shape != (noisy_semantic_tokens.shape[0],):
            raise ValueError("predicted_semantic_mask must have shape [B]")
        semantic_condition = torch.where(
            predicted_semantic_mask[:, None, None, None],
            semantic.full.detach(),
            clean_semantic_tokens.detach(),
        )
        image_latent = self.forward_image_prediction(
            noisy_image_latents, image_timestep, image_labels, semantic_condition
        )
        return CascadePredictions(semantic, image_latent)


def build_model(
    config: ModelConfig,
) -> LatentDDT | SharedCascadeDDT | DecoupledCascadeDDT:
    if config.architecture == "latent":
        return LatentDDT(config)
    if config.architecture == "shared_cascade":
        return SharedCascadeDDT(config)
    if config.architecture == "decoupled_cascade":
        return DecoupledCascadeDDT(config)
    raise ValueError(f"unknown architecture: {config.architecture}")
