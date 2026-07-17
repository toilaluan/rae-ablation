from __future__ import annotations

import math
from contextlib import contextmanager
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .attnres import AttnResCapture, AttnResRouter
from .rope import apply_2d_rope


class DDTOutput(NamedTuple):
    full: Tensor
    early: Tensor


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class SpatialProjection(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, hidden_size, kernel_size=1)

    def forward(self, latent: Tensor) -> Tensor:
        return self.projection(latent).flatten(2).transpose(1, 2)


class GaussianFourierTokenEmbedding(nn.Module):
    def __init__(self, hidden_size: int, num_tokens: int, embedding_size: int = 256) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.randn(embedding_size))
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.hidden_size = hidden_size
        if num_tokens:
            self.learnable_tokens = nn.Parameter(
                torch.randn(num_tokens, hidden_size) / math.sqrt(hidden_size)
            )
        else:
            self.register_parameter("learnable_tokens", None)

    def embed(self, timestep: Tensor) -> Tensor:
        shape = timestep.shape
        angles = timestep.float().reshape(-1, 1) * self.frequencies.reshape(1, -1)
        angles = 2 * torch.pi * angles
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        embedded = self.mlp(embedding.to(dtype=self.mlp[0].weight.dtype))
        return embedded.reshape(*shape, self.hidden_size)

    def forward(self, timestep: Tensor) -> tuple[Tensor, Tensor]:
        if timestep.ndim != 1:
            raise ValueError("time-token timesteps must have shape [B]")
        if self.learnable_tokens is None:
            raise RuntimeError("time tokens are disabled for this model")
        base = self.embed(timestep).unsqueeze(1)
        return base, base + self.learnable_tokens.unsqueeze(0)


class LabelTokenEmbedding(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, num_tokens: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, hidden_size)
        self.learnable_tokens = nn.Parameter(
            torch.randn(num_tokens, hidden_size) / math.sqrt(hidden_size)
        )

    def forward(self, labels: Tensor) -> Tensor:
        return self.embedding(labels).unsqueeze(1) + self.learnable_tokens.unsqueeze(0)


class DDTAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.dropout = dropout

    def forward(
        self,
        x: Tensor,
        height: int,
        width: int,
        spatial_tokens: int,
    ) -> Tensor:
        batch, tokens, _ = x.shape

        def project(layer: nn.Linear) -> Tensor:
            return layer(x).reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(project(self.q))
        k = self.k_norm(project(self.k))
        v = project(self.v)
        spatial_q, spatial_k = apply_2d_rope(
            q[:, :, :spatial_tokens], k[:, :, :spatial_tokens], height, width
        )
        if spatial_tokens < tokens:
            q = torch.cat((spatial_q, q[:, :, spatial_tokens:]), dim=2)
            k = torch.cat((spatial_k, k[:, :, spatial_tokens:]), dim=2)
        else:
            q, k = spatial_q, spatial_k
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, self.hidden_size)
        return self.projection(attended)


class DDTFeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        intermediate_size = int(2 / 3 * hidden_size * mlp_ratio)
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(hidden_size, intermediate_size)
        self.w3 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class DDTEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        layer_index: int,
        attnres_block_size: int | None,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attention = DDTAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.feed_forward = DDTFeedForward(hidden_size, mlp_ratio, dropout)
        self.use_attnres = attnres_block_size is not None
        if attnres_block_size is None:
            self.attnres_is_attn_boundary = False
            self.attnres_is_mlp_boundary = False
            self.attn_router = None
            self.mlp_router = None
        else:
            self.attnres_is_attn_boundary = (2 * layer_index) % attnres_block_size == 0
            self.attnres_is_mlp_boundary = (2 * layer_index + 1) % attnres_block_size == 0
            self.attn_router = AttnResRouter(
                hidden_size, f"encoder.block_{layer_index + 1}.attention"
            )
            self.mlp_router = AttnResRouter(hidden_size, f"encoder.block_{layer_index + 1}.mlp")

    def forward(
        self,
        x: Tensor,
        height: int,
        width: int,
        spatial_tokens: int,
        *attnres_states: Tensor,
    ) -> tuple[Tensor, ...]:
        if not self.use_attnres:
            x = x + self.attention(self.norm1(x), height, width, spatial_tokens)
            return (x + self.feed_forward(self.norm2(x)),)

        if self.attn_router is None or self.mlp_router is None:
            raise RuntimeError("AttnRes encoder block is missing its routers")
        states = attnres_states
        prefix_sum: Tensor | None = x
        residuals = (*states, prefix_sum)
        if self.attnres_is_attn_boundary:
            states = residuals
            prefix_sum = None
        attention_input = self.attn_router(residuals, self.norm1.weight)
        attention_output = self.attention(attention_input, height, width, spatial_tokens)
        prefix_sum = attention_output if prefix_sum is None else prefix_sum + attention_output

        residuals = (*states, prefix_sum)
        if self.attnres_is_mlp_boundary:
            states = residuals
            prefix_sum = None
        mlp_input = self.mlp_router(residuals, self.norm2.weight)
        mlp_output = self.feed_forward(mlp_input)
        hidden_states = mlp_output if prefix_sum is None else prefix_sum + mlp_output
        return (hidden_states, *states)


class DDTDecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attention = DDTAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.feed_forward = DDTFeedForward(hidden_size, mlp_ratio, dropout)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x: Tensor, conditioning: Tensor, height: int, width: int) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(conditioning).chunk(
            6, dim=-1
        )
        x = x + gate_a * self.attention(
            _modulate(self.norm1(x), shift_a, scale_a),
            height,
            width,
            height * width,
        )
        return x + gate_m * self.feed_forward(_modulate(self.norm2(x), shift_m, scale_m))


class DDTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.projection = nn.Linear(hidden_size, out_channels)

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale = self.modulation(conditioning).chunk(2, dim=-1)
        return self.projection(_modulate(self.norm(x), shift, scale))


class RAEv2DDTIG(nn.Module):
    """Dynamic-grid, class-conditional adaptation of RAEv2 ``DiTwDDTHeadIG``."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.attnres_block_size is None:
            self.encoder_input = SpatialProjection(config.in_channels, config.encoder_hidden_size)
            self.input_residuals_proj = None
        else:
            self.encoder_input = None
            self.input_residuals_proj = nn.Linear(
                config.in_channels,
                config.num_input_residuals * config.encoder_hidden_size,
            )
        self.decoder_input = SpatialProjection(config.in_channels, config.decoder_hidden_size)
        self.time_embedding = GaussianFourierTokenEmbedding(
            config.encoder_hidden_size,
            0 if config.use_patch_forcing else config.num_time_tokens,
        )
        self.class_embedding = LabelTokenEmbedding(
            config.num_classes, config.encoder_hidden_size, config.num_class_tokens
        )
        self.encoder_blocks = nn.ModuleList(
            [
                DDTEncoderBlock(
                    config.encoder_hidden_size,
                    config.encoder_num_heads,
                    config.mlp_ratio,
                    config.dropout,
                    layer_index,
                    config.attnres_block_size,
                )
                for layer_index in range(config.encoder_depth)
            ]
        )
        if config.attnres_block_size is None:
            self.early_attnres = None
            self.final_attnres = None
        else:
            self.early_attnres = AttnResRouter(config.encoder_hidden_size, "encoder.early_output")
            self.final_attnres = AttnResRouter(config.encoder_hidden_size, "encoder.final_output")
        self.encoder_to_decoder = (
            nn.Linear(config.encoder_hidden_size, config.decoder_hidden_size)
            if config.encoder_hidden_size != config.decoder_hidden_size
            else nn.Identity()
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DDTDecoderBlock(
                    config.decoder_hidden_size,
                    config.decoder_num_heads,
                    config.mlp_ratio,
                    config.dropout,
                )
                for _ in range(config.decoder_depth)
            ]
        )
        self.full_head = DDTFinalLayer(config.decoder_hidden_size, config.in_channels)
        self.early_head = DDTFinalLayer(config.encoder_hidden_size, config.in_channels)
        self.gradient_checkpointing = config.gradient_checkpointing
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        embedders = [self.decoder_input]
        if self.encoder_input is not None:
            embedders.append(self.encoder_input)
        for embedder in embedders:
            nn.init.xavier_uniform_(embedder.projection.weight.flatten(1))
            nn.init.zeros_(embedder.projection.bias)
        if self.input_residuals_proj is not None:
            for weight in self.input_residuals_proj.weight.chunk(
                self.config.num_input_residuals, dim=0
            ):
                nn.init.xavier_uniform_(weight)
            nn.init.zeros_(self.input_residuals_proj.bias)
        nn.init.normal_(self.class_embedding.embedding.weight, std=0.02)
        for layer in (self.time_embedding.mlp[0], self.time_embedding.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
        for block in self.decoder_blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        for head in (self.full_head, self.early_head):
            nn.init.zeros_(head.modulation[-1].weight)
            nn.init.zeros_(head.modulation[-1].bias)
            nn.init.zeros_(head.projection.weight)
            nn.init.zeros_(head.projection.bias)
        for module in self.modules():
            if isinstance(module, AttnResRouter):
                nn.init.zeros_(module.query_projection.weight)

    @contextmanager
    def capture_attnres_weights(self, height: int, width: int) -> AttnResCapture:
        if self.config.attnres_block_size is None:
            raise RuntimeError("cannot capture AttnRes weights when AttnRes is disabled")
        capture = AttnResCapture(
            spatial_tokens=height * width,
            time_tokens=0 if self.config.use_patch_forcing else self.config.num_time_tokens,
            class_tokens=self.config.num_class_tokens,
            num_input_residuals=self.config.num_input_residuals,
        )
        routers = [module for module in self.modules() if isinstance(module, AttnResRouter)]
        if any(router.capture is not None for router in routers):
            raise RuntimeError("AttnRes weight capture is already active")
        for router in routers:
            router.capture = capture
        try:
            yield capture
        finally:
            for router in routers:
                router.capture = None

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def _checkpointing(self) -> bool:
        return self.gradient_checkpointing and self.training and torch.is_grad_enabled()

    @staticmethod
    def _unpatchify(tokens: Tensor, channels: int, height: int, width: int) -> Tensor:
        return tokens.transpose(1, 2).reshape(tokens.shape[0], channels, height, width)

    def _patch_time_embedding(
        self,
        timestep: Tensor,
        batch_size: int,
        height: int,
        width: int,
    ) -> Tensor:
        spatial_tokens = height * width
        if timestep.ndim == 1:
            if timestep.shape[0] != batch_size:
                raise ValueError("timestep batch dimension must match latent")
            return self.time_embedding.embed(timestep).unsqueeze(1).expand(-1, spatial_tokens, -1)
        if timestep.ndim == 4 and timestep.shape == (batch_size, 1, height, width):
            if not self.config.use_patch_forcing:
                raise ValueError("patchwise timesteps require model.use_patch_forcing")
            return self.time_embedding.embed(timestep[:, 0].flatten(1))
        raise ValueError("timestep must have shape [B] or [B, 1, H, W]")

    def forward(self, latent: Tensor, timestep: Tensor, labels: Tensor) -> DDTOutput:
        if latent.ndim != 4 or latent.shape[1] != self.config.in_channels:
            raise ValueError("latent must have shape [B, configured in_channels, H, W]")
        batch_size, _, height, width = latent.shape
        spatial_tokens = height * width
        patch_timestep = self._patch_time_embedding(timestep, batch_size, height, width)
        label_tokens = self.class_embedding(labels)
        if self.config.use_patch_forcing:
            condition_tokens = label_tokens
        else:
            _, time_tokens = self.time_embedding(timestep)
            condition_tokens = torch.cat((time_tokens, label_tokens), dim=1)
        if self.input_residuals_proj is None:
            if self.encoder_input is None:
                raise RuntimeError("legacy encoder input projection is missing")
            spatial = self.encoder_input(latent)
            sequence = torch.cat((spatial, condition_tokens), dim=1)
            attnres_states: tuple[Tensor, ...] = ()
        else:
            latent_tokens = latent.flatten(2).transpose(1, 2)
            projected = self.input_residuals_proj(latent_tokens).chunk(
                self.config.num_input_residuals, dim=-1
            )
            initial_residuals = tuple(
                torch.cat((spatial_source, condition_tokens), dim=1) for spatial_source in projected
            )
            sequence = initial_residuals[-1]
            attnres_states = initial_residuals[:-1]
        early_tokens: Tensor | None = None
        for index, block in enumerate(self.encoder_blocks, start=1):
            if self._checkpointing():
                block_output = checkpoint(
                    block,
                    sequence,
                    height,
                    width,
                    spatial_tokens,
                    *attnres_states,
                    use_reentrant=False,
                    preserve_rng_state=self.config.dropout > 0,
                )
            else:
                block_output = block(
                    sequence,
                    height,
                    width,
                    spatial_tokens,
                    *attnres_states,
                )
            sequence = block_output[0]
            attnres_states = block_output[1:]
            if index == self.config.base_model_depth:
                if self.early_attnres is None:
                    early_sequence = sequence
                else:
                    early_sequence = self.early_attnres((*attnres_states, sequence))
                early_tokens = early_sequence[:, :spatial_tokens]
        if early_tokens is None:
            raise RuntimeError("DDT base prediction block was not reached")

        if self.final_attnres is not None:
            sequence = self.final_attnres((*attnres_states, sequence))
        encoder_spatial = sequence[:, :spatial_tokens]
        conditioning = self.encoder_to_decoder(F.silu(patch_timestep + encoder_spatial))
        decoded = self.decoder_input(latent)
        for block in self.decoder_blocks:
            if self._checkpointing():
                decoded = checkpoint(
                    block,
                    decoded,
                    conditioning,
                    height,
                    width,
                    use_reentrant=False,
                    preserve_rng_state=self.config.dropout > 0,
                )
            else:
                decoded = block(decoded, conditioning, height, width)

        early_conditioning = F.silu(patch_timestep + early_tokens)
        full = self.full_head(decoded, conditioning)
        early = self.early_head(early_conditioning, early_conditioning)
        return DDTOutput(
            full=self._unpatchify(full, self.config.in_channels, height, width),
            early=self._unpatchify(early, self.config.in_channels, height, width),
        )
