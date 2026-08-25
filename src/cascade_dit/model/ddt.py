from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..alignment import OrthogonalProjector
from ..config import GRID_SIZE, TOKEN_COUNT, TOKEN_DIM, DetailHeadsConfig, ModelConfig
from .detail import PixelPatchDetailer, TokenDetailHead
from .lora import attach_lora, has_lora, merge_lora
from .rope import apply_rope_cos_sin, precompute_rope_tables


def modulate(values: Tensor, shift: Tensor | None, scale: Tensor) -> Tensor:
    result = values * (1 + scale.unsqueeze(1))
    return result if shift is None else result + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def sinusoidal(timestep: Tensor, dimension: int, max_period: int = 10_000) -> Tensor:
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=timestep.device)
            / half
        )
        arguments = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat((arguments.cos(), arguments.sin()), dim=-1)
        if dimension % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
        return embedding

    def forward(self, timestep: Tensor) -> Tensor:
        return self.mlp(self.sinusoidal(timestep, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_probability: float) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_probability = dropout_probability
        self.embedding = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels: Tensor, force_drop: Tensor | None = None) -> Tensor:
        if labels.ndim != 1 or torch.any(labels < 0) or torch.any(labels > self.num_classes):
            raise ValueError("class labels are outside the configured range")
        if force_drop is None and self.training and self.dropout_probability > 0:
            force_drop = torch.rand(labels.shape, device=labels.device) < self.dropout_probability
        if force_drop is not None:
            labels = torch.where(force_drop.bool(), self.num_classes, labels)
        return self.embedding(labels)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        qk_norm: bool = True,
        sdpa: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.sdpa = sdpa
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, values: Tensor, rope: tuple[Tensor, Tensor] | None) -> Tensor:
        batch, tokens, width = values.shape
        qkv = self.qkv(values).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query, key = self.q_norm(query), self.k_norm(key)
        if rope is not None:
            query, key = apply_rope_cos_sin(query, key, rope[0], rope[1])
        if self.sdpa:
            attended = F.scaled_dot_product_attention(query, key, value)
        else:
            weights = (query * self.head_dim**-0.5) @ key.transpose(-2, -1)
            attended = weights.softmax(dim=-1) @ value
        return self.proj(attended.transpose(1, 2).reshape(batch, tokens, width))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, ratio: float) -> None:
        super().__init__()
        intermediate = int((2 / 3) * hidden_size * ratio)
        self.w1 = nn.Linear(hidden_size, intermediate)
        self.w3 = nn.Linear(hidden_size, intermediate)
        self.w2 = nn.Linear(intermediate, hidden_size)

    def forward(self, values: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(values)) * self.w3(values))


class LightningDiTBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size)
        self.norm2 = nn.RMSNorm(config.hidden_size)
        self.attn = Attention(
            config.hidden_size,
            config.num_heads,
            qk_norm=config.qk_norm,
            sdpa=config.sdpa,
        )
        self.mlp = SwiGLU(config.hidden_size, config.mlp_ratio)
        chunks = 6 if config.adaln_shift else 4
        self.adaln = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_size, chunks * config.hidden_size)
        )
        self.with_shift = config.adaln_shift

    def forward(
        self,
        values: Tensor,
        conditioning: Tensor,
        rope_cos: Tensor | None,
        rope_sin: Tensor | None,
    ) -> Tensor:
        if self.with_shift:
            shift_attention, scale_attention, gate_attention, shift_mlp, scale_mlp, gate_mlp = (
                self.adaln(conditioning).chunk(6, dim=-1)
            )
        else:
            scale_attention, gate_attention, scale_mlp, gate_mlp = self.adaln(
                conditioning
            ).chunk(4, dim=-1)
            shift_attention = shift_mlp = None
        rope = None if rope_cos is None or rope_sin is None else (rope_cos, rope_sin)
        values = values + gate_attention.unsqueeze(1) * self.attn(
            modulate(self.norm1(values), shift_attention, scale_attention), rope
        )
        return values + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(values), shift_mlp, scale_mlp)
        )


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, values: Tensor, conditioning: Tensor) -> tuple[Tensor, Tensor]:
        shift, scale = self.adaln(conditioning).chunk(2, dim=-1)
        context = modulate(self.norm(values), shift, scale)
        return self.linear(context), context


@dataclass
class LightningDiTOutput:
    prediction: Tensor
    canonical_prediction: Tensor
    context: Tensor
    token_residual: Tensor
    pixel_residual: Tensor | None


class LightningDiT(nn.Module):
    """Class-conditional LightningDiT over a universal 8x8x768 token grid."""

    def __init__(self, config: ModelConfig, detail_config: DetailHeadsConfig | None = None) -> None:
        super().__init__()
        self.config = config
        self.detail_config = detail_config or DetailHeadsConfig()
        self.input_projection = nn.Linear(config.token_dim, config.hidden_size)
        self.timestep_embedding = TimestepEmbedder(config.hidden_size)
        self.class_embedding = LabelEmbedder(
            config.num_classes, config.hidden_size, config.class_dropout_probability
        )
        self.blocks = nn.ModuleList(LightningDiTBlock(config) for _ in range(config.depth))
        self.final_layer = FinalLayer(config.hidden_size, config.token_dim)
        self.token_detail = TokenDetailHead(
            config.token_dim,
            config.hidden_size,
            depth=self.detail_config.token_depth,
            expansion=self.detail_config.token_expansion,
        )
        self.pixel_detail: PixelPatchDetailer | None = None
        if config.rope:
            cos, sin = precompute_rope_tables(
                GRID_SIZE, GRID_SIZE, config.hidden_size // config.num_heads
            )
        else:
            cos = sin = torch.empty(0)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.projector = OrthogonalProjector(torch.eye(config.token_dim))
        self.stage = "dino12"
        self.gradient_checkpointing = config.gradient_checkpointing
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        nn.init.normal_(self.class_embedding.embedding.weight, std=0.02)
        nn.init.normal_(self.timestep_embedding.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedding.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaln[-1].weight)
            nn.init.zeros_(block.adaln[-1].bias)
        nn.init.zeros_(self.final_layer.adaln[-1].weight)
        nn.init.zeros_(self.final_layer.adaln[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
        nn.init.zeros_(self.token_detail.output_projection.weight)
        nn.init.zeros_(self.token_detail.output_projection.bias)

    def _introduce_pixel_detail(self) -> None:
        if self.pixel_detail is not None or not self.detail_config.pixel_enabled:
            return
        self.pixel_detail = PixelPatchDetailer(
            self.config.hidden_size, tuple(self.detail_config.pixel_channels)
        ).to(
            device=self.input_projection.weight.device,
            dtype=self.input_projection.weight.dtype,
        )

    def set_stage(self, stage: str, projector: Tensor) -> None:
        if stage not in {"dino12", "dino8", "dino4", "rgb"}:
            raise ValueError(f"unknown curriculum stage {stage}")
        if projector.shape != self.projector.matrix.shape:
            raise ValueError("stage projector has an incompatible shape")
        if stage == "rgb":
            self._introduce_pixel_detail()
        self.projector.matrix.copy_(projector.to(self.projector.matrix))
        self.stage = stage

    def _backbone(
        self,
        canonical_noisy: Tensor,
        timestep: Tensor,
        labels: Tensor,
        force_class_drop: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        values = self.input_projection(canonical_noisy)
        conditioning = self.timestep_embedding(timestep) + self.class_embedding(
            labels, force_class_drop
        )
        rope_cos = self.rope_cos if self.rope_cos.numel() else None
        rope_sin = self.rope_sin if self.rope_sin.numel() else None
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                values = checkpoint(
                    block, values, conditioning, rope_cos, rope_sin, use_reentrant=False
                )
            else:
                values = block(values, conditioning, rope_cos, rope_sin)
        return self.final_layer(values, conditioning)

    def forward(
        self,
        noisy_tokens: Tensor,
        timestep: Tensor,
        labels: Tensor,
        *,
        force_class_drop: Tensor | None = None,
        stage: str | None = None,
    ) -> LightningDiTOutput:
        if noisy_tokens.ndim != 3 or noisy_tokens.shape[1:] != (TOKEN_COUNT, TOKEN_DIM):
            raise ValueError("LightningDiT input must have shape [B,64,768]")
        active_stage = self.stage if stage is None else stage
        if active_stage != self.stage:
            raise ValueError("forward stage disagrees with the installed projector")
        canonical_noisy = self.projector.to_canonical(noisy_tokens)
        linear_prediction, context = self._backbone(
            canonical_noisy, timestep, labels, force_class_drop
        )
        token_residual = self.token_detail(canonical_noisy, context)
        canonical_prediction = linear_prediction + token_residual
        prediction = self.projector.from_canonical(canonical_prediction)
        pixel_residual = None
        if active_stage == "rgb" and self.pixel_detail is not None:
            pixel_residual = self.pixel_detail(noisy_tokens, context)
            prediction = prediction + pixel_residual
        return LightningDiTOutput(
            prediction, canonical_prediction, context, token_residual, pixel_residual
        )

    def attach_lora(
        self,
        block_indices: list[int],
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        attach_lora(
            self.blocks, block_indices, rank=rank, alpha=alpha, dropout=dropout
        )

    def merge_lora(self) -> int:
        return merge_lora(self.blocks)

    @torch.no_grad()
    def fold_projector(self) -> None:
        """Fold the active projector into linear I/O and token-detail weights."""
        if has_lora(self):
            raise ValueError("merge LoRA adapters before folding the projector")
        matrix = self.projector.matrix.to(self.input_projection.weight)
        self.input_projection.weight.copy_(self.input_projection.weight @ matrix)
        noisy = self.token_detail.noisy_projection
        noisy.weight.copy_(noisy.weight @ matrix.to(noisy.weight))
        for output in (self.final_layer.linear, self.token_detail.output_projection):
            transform = matrix.T.to(output.weight)
            output.weight.copy_(transform @ output.weight)
            if output.bias is not None:
                output.bias.copy_(transform @ output.bias)
        self.projector.matrix.copy_(torch.eye(matrix.shape[0], device=matrix.device))


def build_model(
    config: ModelConfig, detail_config: DetailHeadsConfig | None = None
) -> LightningDiT:
    return LightningDiT(config, detail_config)


def LightningDiT_B_1(**overrides: object) -> LightningDiT:
    return LightningDiT(ModelConfig(**overrides))
