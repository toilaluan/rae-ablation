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


class MLP(nn.Module):
    def __init__(self, width: int, ratio: float) -> None:
        super().__init__()
        hidden = max(1, int(width * ratio))
        self.input = nn.Linear(width, hidden)
        self.output = nn.Linear(hidden, width)

    def forward(self, values: Tensor) -> Tensor:
        return self.output(F.silu(self.input(values)))


class Attention(nn.Module):
    def __init__(self, width: int, heads: int, global_tokens: int) -> None:
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
            cosine, sine = precompute_rope_tables(8, 8, self.head_width, dtype=dtype)
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


class TransformerBlock(nn.Module):
    """Transformer block with optional patchwise semantic AdaLN."""

    def __init__(
        self,
        width: int,
        heads: int,
        mlp_ratio: float,
        global_tokens: int,
        *,
        conditional: bool,
    ) -> None:
        super().__init__()
        self.norm_attention = RMSNorm(width)
        self.attention = Attention(width, heads, global_tokens)
        self.norm_mlp = RMSNorm(width)
        self.mlp = MLP(width, mlp_ratio)
        self.semantic_adaln = nn.Linear(width, 6 * width) if conditional else None
        if self.semantic_adaln is not None:
            nn.init.zeros_(self.semantic_adaln.weight)
            nn.init.zeros_(self.semantic_adaln.bias)

    @staticmethod
    def _merge(spatial: Tensor, global_values: Tensor) -> Tensor:
        return torch.cat((spatial, global_values), dim=1)

    def forward(
        self, values: Tensor, semantic_patches: Tensor | None = None
    ) -> Tensor:
        if semantic_patches is None:
            values = values + self.attention(self.norm_attention(values))
            return values + self.mlp(self.norm_mlp(values))
        if self.semantic_adaln is None:
            raise ValueError("semantic conditioning was supplied to an unconditional block")
        if semantic_patches.shape != values[:, :64].shape:
            raise ValueError("semantic conditioning must contain 64 patchwise embeddings")
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.semantic_adaln(
            semantic_patches
        ).chunk(6, dim=-1)
        spatial, global_values = values[:, :64], values[:, 64:]
        normalized = self.norm_attention(values)
        attention_input = self._merge(
            _modulate(normalized[:, :64], shift_a, scale_a), normalized[:, 64:]
        )
        attention = self.attention(attention_input)
        values = self._merge(
            spatial + gate_a * attention[:, :64],
            global_values + attention[:, 64:],
        )
        spatial, global_values = values[:, :64], values[:, 64:]
        normalized = self.norm_mlp(values)
        mlp_input = self._merge(
            _modulate(normalized[:, :64], shift_m, scale_m), normalized[:, 64:]
        )
        mlp = self.mlp(mlp_input)
        return self._merge(
            spatial + gate_m * mlp[:, :64],
            global_values + mlp[:, 64:],
        )


class TimestepEmbedder(nn.Module):
    def __init__(self, width: int, token_count: int, frequency_width: int = 256) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.randn(frequency_width), persistent=True)
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
    def __init__(self, width: int, num_classes: int, token_count: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, width)
        self.tokens = nn.Parameter(torch.randn(token_count, width) / width**0.5)

    def forward(self, labels: Tensor) -> Tensor:
        return self.embedding(labels)[:, None] + self.tokens


class PredictionHead(nn.Module):
    def __init__(self, width: int, channels: int, patch_size: int) -> None:
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.norm = RMSNorm(width)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.projection = nn.Linear(width, channels * patch_size**2)

    def forward(self, values: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale = self.modulation(conditioning).chunk(2, dim=-1)
        patches = self.projection(_modulate(self.norm(values), shift, scale))
        side = 8
        batch = patches.shape[0]
        patch = self.patch_size
        return (
            patches.reshape(batch, side, side, patch, patch, self.channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, self.channels, side * patch, side * patch)
        )


class _DiTCore(nn.Module):
    def __init__(self, config: ModelConfig, *, conditional: bool) -> None:
        super().__init__()
        self.config = config
        global_tokens = config.num_timestep_tokens + config.num_class_tokens
        self.timestep_embedder = TimestepEmbedder(
            config.hidden_size, config.num_timestep_tokens
        )
        self.class_embedder = LabelEmbedder(
            config.hidden_size, config.num_classes, config.num_class_tokens
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config.hidden_size,
                config.num_heads,
                config.mlp_ratio,
                global_tokens,
                conditional=conditional,
            )
            for _ in range(config.depth)
        )
        self.gradient_checkpointing = config.gradient_checkpointing
        nn.init.normal_(self.class_embedder.embedding.weight, std=0.02)
        for layer in (self.timestep_embedder.mlp[0], self.timestep_embedder.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)

    def _global_tokens(
        self, timestep: Tensor, labels: Tensor
    ) -> tuple[Tensor, Tensor]:
        timestep_base, timestep_tokens = self.timestep_embedder(timestep)
        return timestep_base, torch.cat(
            (timestep_tokens, self.class_embedder(labels)), dim=1
        )

    def _block(
        self,
        block: TransformerBlock,
        values: Tensor,
        semantic_patches: Tensor | None,
    ) -> Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, values, semantic_patches, use_reentrant=False)
        return block(values, semantic_patches)

    @staticmethod
    def _autocast(values: Tensor):
        if values.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()


def _validate_common(values: Tensor, timestep: Tensor, labels: Tensor, shape: tuple[int, ...]) -> None:
    if tuple(values.shape[1:]) != shape:
        raise ValueError(f"input must have shape [B,{','.join(map(str, shape))}]")
    if timestep.shape != (values.shape[0],) or labels.shape != (values.shape[0],):
        raise ValueError("timestep and labels must have shape [B]")


class LatentDiT(_DiTCore):
    """Pure native Mage image-latent baseline with no semantic modules."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        config = config or ModelConfig(architecture="latent")
        if config.architecture != "latent":
            raise ValueError("LatentDiT requires model.architecture=latent")
        config.validate()
        super().__init__(config, conditional=False)
        self.image_latent_projection = nn.Conv2d(
            IMAGE_LATENT_CHANNELS,
            config.hidden_size,
            kernel_size=config.image_latent_patch_size,
            stride=config.image_latent_patch_size,
        )
        self.image_latent_head = PredictionHead(
            config.hidden_size, IMAGE_LATENT_CHANNELS, config.image_latent_patch_size
        )

    def forward(self, image_latents: Tensor, timestep: Tensor, labels: Tensor) -> Tensor:
        _validate_common(image_latents, timestep, labels, IMAGE_LATENT_SHAPE)
        with self._autocast(image_latents):
            patches = self.image_latent_projection(image_latents).flatten(2).transpose(1, 2)
            timestep_base, global_tokens = self._global_tokens(timestep, labels)
            states = torch.cat((patches, global_tokens), dim=1)
            for block in self.blocks:
                states = self._block(block, states, None)
            conditioning = F.silu(timestep_base + states[:, :64])
            return self.image_latent_head(states[:, :64], conditioning)

    forward_image_prediction = forward


class SemanticDiT(_DiTCore):
    def __init__(self, config: ModelConfig | None = None) -> None:
        config = config or ModelConfig(architecture="decoupled_cascade")
        config.validate()
        super().__init__(config, conditional=False)
        self.semantic_projection = nn.Conv2d(
            SEMANTIC_CHANNELS,
            config.hidden_size,
            kernel_size=config.semantic_patch_size,
            stride=config.semantic_patch_size,
        )
        self.semantic_full_head = PredictionHead(
            config.hidden_size, SEMANTIC_CHANNELS, config.semantic_patch_size
        )
        self.semantic_base_head = PredictionHead(
            config.hidden_size, SEMANTIC_CHANNELS, config.semantic_patch_size
        )

    def forward(
        self, semantic_tokens: Tensor, timestep: Tensor, labels: Tensor
    ) -> SemanticPredictions:
        _validate_common(semantic_tokens, timestep, labels, SEMANTIC_TOKEN_SHAPE)
        with self._autocast(semantic_tokens):
            patches = self.semantic_projection(semantic_tokens).flatten(2).transpose(1, 2)
            timestep_base, global_tokens = self._global_tokens(timestep, labels)
            states = torch.cat((patches, global_tokens), dim=1)
            base_states: Tensor | None = None
            for index, block in enumerate(self.blocks, start=1):
                states = self._block(block, states, None)
                if index == self.config.base_model_depth:
                    base_states = states[:, :64]
            assert base_states is not None
            full_states = states[:, :64]
            full_conditioning = F.silu(timestep_base + full_states)
            base_conditioning = F.silu(timestep_base + base_states)
            return SemanticPredictions(
                self.semantic_full_head(full_states, full_conditioning),
                self.semantic_base_head(base_states, base_conditioning),
            )

    forward_semantic_predictions = forward


class ConditionalLatentDiT(_DiTCore):
    def __init__(self, config: ModelConfig | None = None) -> None:
        config = config or ModelConfig(architecture="decoupled_cascade")
        config.validate()
        super().__init__(config, conditional=True)
        self.image_latent_projection = nn.Conv2d(
            IMAGE_LATENT_CHANNELS,
            config.hidden_size,
            kernel_size=config.image_latent_patch_size,
            stride=config.image_latent_patch_size,
        )
        self.semantic_condition_projection = nn.Conv2d(
            SEMANTIC_CHANNELS,
            config.hidden_size,
            kernel_size=config.semantic_patch_size,
            stride=config.semantic_patch_size,
        )
        self.image_latent_head = PredictionHead(
            config.hidden_size, IMAGE_LATENT_CHANNELS, config.image_latent_patch_size
        )

    def forward(
        self,
        image_latents: Tensor,
        timestep: Tensor,
        labels: Tensor,
        semantic_tokens: Tensor,
    ) -> Tensor:
        _validate_common(image_latents, timestep, labels, IMAGE_LATENT_SHAPE)
        if tuple(semantic_tokens.shape) != (
            image_latents.shape[0],
            *SEMANTIC_TOKEN_SHAPE,
        ):
            raise ValueError("semantic_tokens must have shape [B,768,8,8]")
        with self._autocast(image_latents):
            patches = self.image_latent_projection(image_latents).flatten(2).transpose(1, 2)
            semantic_patches = (
                self.semantic_condition_projection(semantic_tokens)
                .flatten(2)
                .transpose(1, 2)
            )
            timestep_base, global_tokens = self._global_tokens(timestep, labels)
            states = torch.cat((patches, global_tokens), dim=1)
            for block in self.blocks:
                states = self._block(block, states, semantic_patches)
            spatial = states[:, :64]
            conditioning = F.silu(timestep_base + spatial)
            return self.image_latent_head(spatial, conditioning)

    forward_image_prediction = forward


class SharedCascadeDiT(_DiTCore):
    """Semantic and image-latent routes around one shared transformer core."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        config = config or ModelConfig(architecture="shared_cascade")
        if config.architecture != "shared_cascade":
            raise ValueError("SharedCascadeDiT requires model.architecture=shared_cascade")
        config.validate()
        super().__init__(config, conditional=True)
        self.semantic_projection = nn.Conv2d(
            SEMANTIC_CHANNELS,
            config.hidden_size,
            kernel_size=config.semantic_patch_size,
        )
        self.image_latent_projection = nn.Conv2d(
            IMAGE_LATENT_CHANNELS,
            config.hidden_size,
            kernel_size=config.image_latent_patch_size,
            stride=config.image_latent_patch_size,
        )
        self.semantic_condition_projection = nn.Conv2d(
            SEMANTIC_CHANNELS,
            config.hidden_size,
            kernel_size=config.semantic_patch_size,
        )
        self.semantic_full_head = PredictionHead(
            config.hidden_size, SEMANTIC_CHANNELS, config.semantic_patch_size
        )
        self.semantic_base_head = PredictionHead(
            config.hidden_size, SEMANTIC_CHANNELS, config.semantic_patch_size
        )
        self.image_latent_head = PredictionHead(
            config.hidden_size, IMAGE_LATENT_CHANNELS, config.image_latent_patch_size
        )

    def forward_semantic_predictions(
        self, semantic_tokens: Tensor, timestep: Tensor, labels: Tensor
    ) -> SemanticPredictions:
        _validate_common(semantic_tokens, timestep, labels, SEMANTIC_TOKEN_SHAPE)
        with self._autocast(semantic_tokens):
            patches = self.semantic_projection(semantic_tokens).flatten(2).transpose(1, 2)
            timestep_base, global_tokens = self._global_tokens(timestep, labels)
            states = torch.cat((patches, global_tokens), dim=1)
            base_states: Tensor | None = None
            for index, block in enumerate(self.blocks, start=1):
                states = self._block(block, states, None)
                if index == self.config.base_model_depth:
                    base_states = states[:, :64]
            assert base_states is not None
            full_states = states[:, :64]
            return SemanticPredictions(
                self.semantic_full_head(
                    full_states, F.silu(timestep_base + full_states)
                ),
                self.semantic_base_head(
                    base_states, F.silu(timestep_base + base_states)
                ),
            )

    def forward_image_prediction(
        self,
        image_latents: Tensor,
        timestep: Tensor,
        labels: Tensor,
        semantic_tokens: Tensor,
    ) -> Tensor:
        _validate_common(image_latents, timestep, labels, IMAGE_LATENT_SHAPE)
        if tuple(semantic_tokens.shape) != (
            image_latents.shape[0],
            *SEMANTIC_TOKEN_SHAPE,
        ):
            raise ValueError("semantic_tokens must have shape [B,768,8,8]")
        with self._autocast(image_latents):
            patches = self.image_latent_projection(image_latents).flatten(2).transpose(1, 2)
            semantic_patches = (
                self.semantic_condition_projection(semantic_tokens)
                .flatten(2)
                .transpose(1, 2)
            )
            timestep_base, global_tokens = self._global_tokens(timestep, labels)
            states = torch.cat((patches, global_tokens), dim=1)
            for block in self.blocks:
                states = self._block(block, states, semantic_patches)
            spatial = states[:, :64]
            return self.image_latent_head(
                spatial, F.silu(timestep_base + spatial)
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


class DecoupledCascadeDiT(nn.Module):
    """Two independent denoisers joined only by a semantic tensor."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig(architecture="decoupled_cascade")
        if config.architecture != "decoupled_cascade":
            raise ValueError(
                "DecoupledCascadeDiT requires model.architecture=decoupled_cascade"
            )
        config.validate()
        self.config = config
        self.semantic_dit = SemanticDiT(config)
        self.image_latent_dit = ConditionalLatentDiT(config)

    @property
    def blocks(self) -> tuple[nn.Module, ...]:
        return (*self.semantic_dit.blocks, *self.image_latent_dit.blocks)

    def forward_semantic_predictions(
        self, semantic_tokens: Tensor, timestep: Tensor, labels: Tensor
    ) -> SemanticPredictions:
        return self.semantic_dit(semantic_tokens, timestep, labels)

    def forward_image_prediction(
        self,
        image_latents: Tensor,
        timestep: Tensor,
        labels: Tensor,
        semantic_tokens: Tensor,
    ) -> Tensor:
        return self.image_latent_dit(image_latents, timestep, labels, semantic_tokens)

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


def build_model(config: ModelConfig) -> LatentDiT | SharedCascadeDiT | DecoupledCascadeDiT:
    if config.architecture == "latent":
        return LatentDiT(config)
    if config.architecture == "shared_cascade":
        return SharedCascadeDiT(config)
    if config.architecture == "decoupled_cascade":
        return DecoupledCascadeDiT(config)
    raise ValueError(f"unknown architecture: {config.architecture}")
