from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .dit import DiTOutput
from .rope import apply_2d_rope


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class SpatialProjection(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, hidden_size, kernel_size=1)

    def forward(self, latent: Tensor) -> Tensor:
        return self.projection(latent).flatten(2).transpose(1, 2)


class GaussianFourierTokenEmbedding(nn.Module):
    def __init__(
        self, hidden_size: int, num_tokens: int, embedding_size: int = 256
    ) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.randn(embedding_size))
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.learnable_tokens = nn.Parameter(
            torch.randn(num_tokens, hidden_size) / math.sqrt(hidden_size)
        )

    def forward(self, timestep: Tensor) -> tuple[Tensor, Tensor]:
        angles = timestep.float().reshape(-1, 1) * self.frequencies.reshape(1, -1)
        angles = 2 * torch.pi * angles
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        base = self.mlp(embedding.to(dtype=self.mlp[0].weight.dtype)).unsqueeze(1)
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
        self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attention = DDTAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.feed_forward = DDTFeedForward(hidden_size, mlp_ratio, dropout)

    def forward(
        self, x: Tensor, height: int, width: int, spatial_tokens: int
    ) -> Tensor:
        x = x + self.attention(self.norm1(x), height, width, spatial_tokens)
        return x + self.feed_forward(self.norm2(x))


class DDTDecoderBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attention = DDTAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.feed_forward = DDTFeedForward(hidden_size, mlp_ratio, dropout)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(
        self, x: Tensor, conditioning: Tensor, height: int, width: int
    ) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(
            conditioning
        ).chunk(6, dim=-1)
        x = x + gate_a * self.attention(
            _modulate(self.norm1(x), shift_a, scale_a),
            height,
            width,
            height * width,
        )
        return x + gate_m * self.feed_forward(
            _modulate(self.norm2(x), shift_m, scale_m)
        )


class DDTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )
        self.projection = nn.Linear(hidden_size, out_channels)

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale = self.modulation(conditioning).chunk(2, dim=-1)
        return self.projection(_modulate(self.norm(x), shift, scale))


class RAEv2DDTIG(nn.Module):
    """Dynamic-grid, class-conditional adaptation of RAEv2 ``DiTwDDTHeadIG``."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        ddt = config.ddt_ig
        self.encoder_input = SpatialProjection(
            config.in_channels, ddt.encoder_hidden_size
        )
        self.decoder_input = SpatialProjection(
            config.in_channels, ddt.decoder_hidden_size
        )
        self.time_embedding = GaussianFourierTokenEmbedding(
            ddt.encoder_hidden_size, ddt.num_time_tokens
        )
        self.class_embedding = LabelTokenEmbedding(
            config.num_classes, ddt.encoder_hidden_size, ddt.num_class_tokens
        )
        self.encoder_blocks = nn.ModuleList(
            [
                DDTEncoderBlock(
                    ddt.encoder_hidden_size,
                    ddt.encoder_num_heads,
                    ddt.mlp_ratio,
                    config.dropout,
                )
                for _ in range(ddt.encoder_depth)
            ]
        )
        self.encoder_to_decoder = (
            nn.Linear(ddt.encoder_hidden_size, ddt.decoder_hidden_size)
            if ddt.encoder_hidden_size != ddt.decoder_hidden_size
            else nn.Identity()
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DDTDecoderBlock(
                    ddt.decoder_hidden_size,
                    ddt.decoder_num_heads,
                    ddt.mlp_ratio,
                    config.dropout,
                )
                for _ in range(ddt.decoder_depth)
            ]
        )
        self.full_head = DDTFinalLayer(ddt.decoder_hidden_size, config.in_channels)
        self.early_head = DDTFinalLayer(ddt.encoder_hidden_size, config.in_channels)
        self.gradient_checkpointing = config.gradient_checkpointing
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for embedder in (self.encoder_input, self.decoder_input):
            nn.init.xavier_uniform_(embedder.projection.weight.flatten(1))
            nn.init.zeros_(embedder.projection.bias)
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

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def _checkpointing(self) -> bool:
        return self.gradient_checkpointing and self.training and torch.is_grad_enabled()

    @staticmethod
    def _unpatchify(tokens: Tensor, channels: int, height: int, width: int) -> Tensor:
        return tokens.transpose(1, 2).reshape(tokens.shape[0], channels, height, width)

    def forward(self, latent: Tensor, timestep: Tensor, labels: Tensor) -> DiTOutput:
        if latent.ndim != 4 or latent.shape[1] != self.config.in_channels:
            raise ValueError("latent must have shape [B, configured in_channels, H, W]")
        _, _, height, width = latent.shape
        spatial_tokens = height * width
        base_timestep, time_tokens = self.time_embedding(timestep)
        sequence = torch.cat(
            (self.encoder_input(latent), time_tokens, self.class_embedding(labels)), dim=1
        )
        early_tokens: Tensor | None = None
        for index, block in enumerate(self.encoder_blocks, start=1):
            if self._checkpointing():
                sequence = checkpoint(
                    block,
                    sequence,
                    height,
                    width,
                    spatial_tokens,
                    use_reentrant=False,
                    preserve_rng_state=self.config.dropout > 0,
                )
            else:
                sequence = block(sequence, height, width, spatial_tokens)
            if index == self.config.ddt_ig.base_model_depth:
                early_tokens = sequence[:, :spatial_tokens]
        if early_tokens is None:
            raise RuntimeError("DDT base prediction block was not reached")

        encoder_spatial = sequence[:, :spatial_tokens]
        conditioning = self.encoder_to_decoder(F.silu(base_timestep + encoder_spatial))
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

        early_conditioning = F.silu(base_timestep + early_tokens)
        full = self.full_head(decoded, conditioning)
        early = self.early_head(early_conditioning, early_conditioning)
        return DiTOutput(
            full=self._unpatchify(full, self.config.in_channels, height, width),
            early=self._unpatchify(early, self.config.in_channels, height, width),
        )
