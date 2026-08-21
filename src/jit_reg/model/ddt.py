from __future__ import annotations

from contextlib import nullcontext
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..config import DDTConfig, RepaConfig
from .rope import apply_rope_cos_sin, precompute_rope_tables


class DDTOutput(NamedTuple):
    full_prediction: Tensor
    base_prediction: Tensor
    repa_projection: Tensor | None

    @property
    def x_prediction(self) -> Tensor:
        return self.full_prediction


class DDTPredictions(NamedTuple):
    full_prediction: Tensor
    base_prediction: Tensor


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = max(1, int(2 * width * mlp_ratio / 3))
        self.w1 = nn.Linear(width, hidden)
        self.w2 = nn.Linear(width, hidden)
        self.w3 = nn.Linear(hidden, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class MLP(nn.Module):
    def __init__(self, width: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = max(1, int(width * mlp_ratio))
        self.w1 = nn.Linear(width, hidden)
        self.w2 = nn.Linear(hidden, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)))

class NormalizedAttention(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        spatial_size: int,
        condition_tokens: int = 0,
    ) -> None:
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
                spatial_size,
                spatial_size,
                self.head_dim,
                dtype=dtype,
            )
            if condition_tokens:
                cos = torch.cat((cos, torch.ones(condition_tokens, self.head_dim, dtype=dtype)))
                sin = torch.cat((sin, torch.zeros(condition_tokens, self.head_dim, dtype=dtype)))
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


class DDTEncoderBlock(nn.Module):
    def __init__(
        self,
        width: int,
        num_heads: int,
        mlp_ratio: float,
        spatial_size: int,
        condition_tokens: int,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attention = NormalizedAttention(width, num_heads, spatial_size, condition_tokens)
        self.norm2 = RMSNorm(width)
        self.feed_forward = MLP(width, mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class DDTDecoderBlock(nn.Module):
    def __init__(self, width: int, num_heads: int, mlp_ratio: float, spatial_size: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(width)
        self.attention = NormalizedAttention(width, num_heads, spatial_size)
        self.norm2 = RMSNorm(width)
        self.feed_forward = MLP(width, mlp_ratio)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(
            conditioning
        ).chunk(6, dim=-1)
        x = x + gate_a * self.attention(_modulate(self.norm1(x), shift_a, scale_a))
        return x + gate_m * self.feed_forward(_modulate(self.norm2(x), shift_m, scale_m))


class DDTFinalLayer(nn.Module):
    def __init__(self, width: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm(width)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.projection = nn.Linear(width, patch_size**2 * out_channels)

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale = self.modulation(conditioning).chunk(2, dim=-1)
        return self.projection(_modulate(self.norm(x), shift, scale))


class GaussianFourierEmbedding(nn.Module):
    def __init__(self, width: int, token_count: int, embedding_size: int = 256) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.randn(embedding_size), persistent=True)
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_size, width),
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


class StageEmbedder(nn.Module):
    def __init__(self, width: int, token_count: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(2, width)
        self.tokens = nn.Parameter(torch.randn(token_count, width) / width**0.5)

    def forward(self, stage: Tensor) -> tuple[Tensor, Tensor]:
        base = self.embedding(stage)[:, None]
        return base, base + self.tokens


class DDT(nn.Module):
    """Stage-conditioned semantic-cascade DDT with optional training-only REPA."""

    def __init__(
        self,
        config: DDTConfig | None = None,
        repa_config: RepaConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DDTConfig()
        self.repa_config = repa_config or RepaConfig()
        self.config.validate()
        self.repa_config.validate(self.config.depth[0])
        enc_width, dec_width = self.config.hidden_size
        enc_depth, dec_depth = self.config.depth
        enc_heads, dec_heads = self.config.num_heads
        s_patch, x_patch = self.config.patch_size
        self.in_channels = self.config.in_channels
        self.spatial_tokens = (self.config.input_size // s_patch) ** 2
        self.spatial_projection = nn.Conv2d(
            self.in_channels, enc_width, kernel_size=s_patch, stride=s_patch
        )
        self.full_input_projection = nn.Conv2d(
            self.in_channels, dec_width, kernel_size=x_patch, stride=x_patch
        )
        self.timestep_embedder = GaussianFourierEmbedding(
            enc_width, self.config.num_timestep_tokens
        )
        self.stage_embedder = StageEmbedder(enc_width, self.config.num_stage_tokens)
        self.class_embedder = LabelEmbedder(
            enc_width, self.config.num_classes, self.config.num_class_tokens
        )
        condition_tokens = (
            self.config.num_timestep_tokens
            + self.config.num_stage_tokens
            + self.config.num_class_tokens
        )
        self.encoder_blocks = nn.ModuleList(
            DDTEncoderBlock(
                enc_width,
                enc_heads,
                self.config.mlp_ratio,
                self.config.input_size // s_patch,
                condition_tokens,
            )
            for _ in range(enc_depth)
        )
        self.encoder_to_decoder = (
            nn.Linear(enc_width, dec_width) if enc_width != dec_width else nn.Identity()
        )
        self.decoder_blocks = nn.ModuleList(
            DDTDecoderBlock(
                dec_width,
                dec_heads,
                self.config.mlp_ratio,
                self.config.input_size // x_patch,
            )
            for _ in range(dec_depth)
        )
        self.full_head = DDTFinalLayer(dec_width, x_patch, self.in_channels)
        self.base_head = DDTFinalLayer(enc_width, s_patch, self.in_channels)
        self.repa_projector = (
            nn.Sequential(
                nn.Linear(enc_width, self.repa_config.projector_width),
                nn.SiLU(),
                nn.Linear(
                    self.repa_config.projector_width,
                    self.repa_config.projector_width,
                ),
                nn.SiLU(),
                nn.Linear(self.repa_config.projector_width, self.config.in_channels),
            )
            if self.repa_config.enabled
            else None
        )
        self.gradient_checkpointing = self.config.gradient_checkpointing
        self._initialize_weights()

    @property
    def blocks(self) -> tuple[nn.Module, ...]:
        return (*self.encoder_blocks, *self.decoder_blocks)

    @property
    def s_embedder(self) -> nn.Conv2d:
        return self.spatial_projection

    @property
    def x_embedder(self) -> nn.Conv2d:
        return self.full_input_projection

    @property
    def s_projector(self) -> nn.Module:
        return self.encoder_to_decoder

    @property
    def final_layer(self) -> DDTFinalLayer:
        return self.full_head

    def _initialize_weights(self) -> None:
        for projection in (self.spatial_projection, self.full_input_projection):
            nn.init.xavier_uniform_(projection.weight.flatten(1))
            nn.init.zeros_(projection.bias)
        nn.init.normal_(self.class_embedder.embedding.weight, std=0.02)
        nn.init.normal_(self.stage_embedder.embedding.weight, std=0.02)
        for layer in (
            self.timestep_embedder.mlp[0],
            self.timestep_embedder.mlp[2],
        ):
            nn.init.normal_(layer.weight, std=0.02)
        for block in self.decoder_blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        for head in (self.full_head, self.base_head):
            nn.init.zeros_(head.modulation[-1].weight)
            nn.init.zeros_(head.modulation[-1].bias)
            nn.init.zeros_(head.projection.weight)
            nn.init.zeros_(head.projection.bias)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def _autocast(self, x: Tensor):
        if x.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def _tokens(projection: nn.Conv2d, x: Tensor) -> Tensor:
        return projection(x).flatten(2).transpose(1, 2)

    def _unpatchify(self, tokens: Tensor, patch_size: int) -> Tensor:
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise ValueError("DDT output token count must be square")
        batch = tokens.shape[0]
        return (
            tokens.reshape(batch, side, side, patch_size, patch_size, self.in_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, self.in_channels, side * patch_size, side * patch_size)
        )

    def _validate_inputs(
        self, x: Tensor, timestep: Tensor, stage: Tensor, labels: Tensor
    ) -> None:
        if tuple(x.shape[1:]) != self.config.input_shape:
            raise ValueError(f"DDT input must have shape [B, {self.config.input_shape}]")
        if any(value.shape != (x.shape[0],) for value in (timestep, stage, labels)):
            raise ValueError("timestep, stage, and labels must have shape [B]")

    def _sequence(
        self, x: Tensor, timestep: Tensor, stage: Tensor, labels: Tensor
    ) -> tuple[Tensor, Tensor]:
        spatial = self._tokens(self.spatial_projection, x)
        timestep_base, timestep_tokens = self.timestep_embedder(timestep)
        stage_base, stage_tokens = self.stage_embedder(stage)
        return (
            torch.cat(
                (spatial, timestep_tokens, stage_tokens, self.class_embedder(labels)), dim=1
            ),
            timestep_base + stage_base,
        )

    def _encoder_block(self, block: DDTEncoderBlock, states: Tensor) -> Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, states, use_reentrant=False)
        return block(states)

    def _decoder_block(
        self, block: DDTDecoderBlock, states: Tensor, conditioning: Tensor
    ) -> Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, states, conditioning, use_reentrant=False)
        return block(states, conditioning)

    def _full_from_states(self, x: Tensor, states: Tensor, timestep_base: Tensor) -> Tensor:
        conditioning = self.encoder_to_decoder(
            F.silu(timestep_base + states[:, : self.spatial_tokens])
        )
        decoded = self._tokens(self.full_input_projection, x)
        for block in self.decoder_blocks:
            decoded = self._decoder_block(block, decoded, conditioning)
        return self._unpatchify(
            self.full_head(decoded, conditioning), self.config.patch_size[1]
        )

    def _base_from_states(self, states: Tensor, timestep_base: Tensor) -> Tensor:
        conditioning = F.silu(timestep_base + states[:, : self.spatial_tokens])
        return self._unpatchify(
            self.base_head(conditioning, conditioning), self.config.patch_size[0]
        )

    def forward_base_prediction(
        self, x: Tensor, timestep: Tensor, stage: Tensor, labels: Tensor
    ) -> Tensor:
        self._validate_inputs(x, timestep, stage, labels)
        with self._autocast(x):
            states, timestep_base = self._sequence(x, timestep, stage, labels)
            for block in self.encoder_blocks[: self.config.base_model_depth]:
                states = self._encoder_block(block, states)
            return self._base_from_states(states, timestep_base)

    def forward_full_prediction(
        self, x: Tensor, timestep: Tensor, stage: Tensor, labels: Tensor
    ) -> Tensor:
        self._validate_inputs(x, timestep, stage, labels)
        with self._autocast(x):
            states, timestep_base = self._sequence(x, timestep, stage, labels)
            for block in self.encoder_blocks:
                states = self._encoder_block(block, states)
            return self._full_from_states(x, states, timestep_base)

    def forward_predictions(
        self, x: Tensor, timestep: Tensor, stage: Tensor, labels: Tensor
    ) -> DDTPredictions:
        """Return both x-prediction heads without evaluating the training-only REPA head."""
        self._validate_inputs(x, timestep, stage, labels)
        with self._autocast(x):
            states, timestep_base = self._sequence(x, timestep, stage, labels)
            base_states: Tensor | None = None
            for index, block in enumerate(self.encoder_blocks, start=1):
                states = self._encoder_block(block, states)
                if index == self.config.base_model_depth:
                    base_states = states
            assert base_states is not None
            return DDTPredictions(
                self._full_from_states(x, states, timestep_base),
                self._base_from_states(base_states, timestep_base),
            )

    def forward(
        self, x: Tensor, timestep: Tensor, stage: Tensor, labels: Tensor
    ) -> DDTOutput:
        self._validate_inputs(x, timestep, stage, labels)
        with self._autocast(x):
            states, timestep_base = self._sequence(x, timestep, stage, labels)
            base_states: Tensor | None = None
            repa_states: Tensor | None = None
            for index, block in enumerate(self.encoder_blocks, start=1):
                states = self._encoder_block(block, states)
                if index == self.config.base_model_depth:
                    base_states = states
                if self.repa_config.enabled and index == self.repa_config.layer_depth:
                    repa_states = states[:, : self.spatial_tokens]
            assert base_states is not None
            full = self._full_from_states(x, states, timestep_base)
            base = self._base_from_states(base_states, timestep_base)
            projection: Tensor | None = None
            if self.repa_config.enabled:
                assert repa_states is not None and self.repa_projector is not None
                projection = self.repa_projector(repa_states)
            return DDTOutput(full, base, projection)
