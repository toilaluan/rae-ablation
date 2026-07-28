from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig, REGLoRAConfig
from .rope import apply_rope_cos_sin, precompute_ddt_rope_tables


class DDTOutput(NamedTuple):
    image: Tensor
    reg: Tensor | None = None
    repa: Tensor | None = None
    reg_early: Tensor | None = None


class DDTREGOutput(NamedTuple):
    reg: Tensor
    reg_early: Tensor


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


def _gate(x: Tensor, gate: Tensor) -> Tensor:
    return x * (gate.unsqueeze(1) if gate.ndim == 2 else gate)


class LowRankAdapter(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(input_size, rank, bias=False)
        self.up = nn.Linear(rank, output_size, bias=False)
        self.scale = alpha / rank
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.down(x)) * self.scale


def classifier_free_labels(
    labels: Tensor,
    num_classes: int,
    probability: float,
    generator: torch.Generator,
) -> Tensor:
    """Replace labels with the null class using an explicit RNG stream."""
    if labels.ndim != 1:
        raise ValueError("labels must have shape [B]")
    if not 0 <= probability < 1:
        raise ValueError("label dropout probability must be in [0, 1)")
    if probability == 0:
        return labels
    dropped = (
        torch.rand(labels.shape, device=labels.device, generator=generator)
        < probability
    )
    return torch.where(
        dropped,
        torch.full_like(labels, num_classes),
        labels,
    )


class GaussianFourierEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_tokens: int = 4,
        embedding_size: int = 256,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or n_tokens < 0 or embedding_size < 1 or scale <= 0:
            raise ValueError(
                "Gaussian Fourier embedding dimensions must be valid"
            )
        self.W = nn.Parameter(
            torch.normal(0, scale, (embedding_size,)),
            requires_grad=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.learnable_tokens = nn.Parameter(
            torch.normal(
                0,
                1 / hidden_size**0.5,
                (n_tokens, hidden_size),
            )
        )

    def forward(
        self,
        timestep: Tensor,
        return_base_embed: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if timestep.ndim != 1:
            raise ValueError("timestep must have shape [B]")
        angles = (
            timestep[:, None].float()
            * self.W[None, :]
            * 2
            * torch.pi
        )
        embedding = torch.cat(
            (torch.sin(angles), torch.cos(angles)),
            dim=-1,
        )
        base = self.mlp(embedding.to(self.mlp[0].weight.dtype))
        context = self.learnable_tokens + base.unsqueeze(1)
        if return_base_embed:
            return base.unsqueeze(1), context
        return context


class ConditionEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_classes: int = 1000,
        context_dim: int = 768,
        condition_type: str = "label",
        n_tokens: int = 8,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or n_tokens < 0:
            raise ValueError("condition embedding dimensions must be valid")
        self.condition_type = condition_type
        if condition_type == "label":
            self.embedding_table: nn.Embedding | None = nn.Embedding(
                num_classes + 1,
                hidden_size,
            )
            self.learnable_tokens = nn.Parameter(
                torch.normal(
                    0,
                    1 / hidden_size**0.5,
                    (n_tokens, hidden_size),
                )
            )
            self.norm: nn.RMSNorm | None = None
            self.projection: nn.Linear | None = None
        elif condition_type == "text":
            self.embedding_table = None
            self.register_parameter("learnable_tokens", None)
            self.norm = nn.RMSNorm(context_dim)
            self.projection = nn.Linear(context_dim, hidden_size)
        else:
            raise ValueError(f"unknown condition type: {condition_type}")

    def base_embedding(self, condition: Tensor) -> Tensor:
        if self.embedding_table is None:
            raise ValueError(
                "base embedding is only available for label conditions"
            )
        return self.embedding_table(condition)

    def forward(self, condition: Tensor) -> Tensor:
        if self.condition_type == "label":
            assert self.learnable_tokens is not None
            return (
                self.learnable_tokens
                + self.base_embedding(condition).unsqueeze(1)
            )
        assert self.norm is not None and self.projection is not None
        return self.projection(self.norm(condition))


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        grid_size: int,
        reg_pool_size: int | None = None,
        condition_tokens: int = 0,
        spatial_reg_rope: bool = False,
        independent_reg_attention: bool = False,
        reg_lora: REGLoRAConfig | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.grid_size = grid_size
        self.reg_pool_size = reg_pool_size
        self.condition_tokens = condition_tokens
        reg_tokens = reg_pool_size**2 if reg_pool_size is not None else 0
        self.reg_tokens = reg_tokens
        self.image_tokens = grid_size**2
        self.independent_reg_attention = independent_reg_attention
        self.expected_tokens = (
            reg_tokens + grid_size**2 + condition_tokens
        )
        rope_cos_fp32, rope_sin_fp32 = precompute_ddt_rope_tables(
            grid_size,
            grid_size,
            self.head_dim,
            reg_pool_size=reg_pool_size,
            condition_tokens=condition_tokens,
            spatial_reg_rope=spatial_reg_rope,
            dtype=torch.float32,
        )
        rope_cos_bf16, rope_sin_bf16 = precompute_ddt_rope_tables(
            grid_size,
            grid_size,
            self.head_dim,
            reg_pool_size=reg_pool_size,
            condition_tokens=condition_tokens,
            spatial_reg_rope=spatial_reg_rope,
            dtype=torch.bfloat16,
        )
        self.register_buffer(
            "rope_cos_fp32",
            rope_cos_fp32,
            persistent=False,
        )
        self.register_buffer(
            "rope_sin_fp32",
            rope_sin_fp32,
            persistent=False,
        )
        self.register_buffer(
            "rope_cos_bf16",
            rope_cos_bf16,
            persistent=False,
        )
        self.register_buffer(
            "rope_sin_bf16",
            rope_sin_bf16,
            persistent=False,
        )
        if independent_reg_attention:
            def reg_rope(table: Tensor) -> Tensor:
                if condition_tokens:
                    return torch.cat(
                        (
                            table[:reg_tokens],
                            table[-condition_tokens:],
                        )
                    )
                return table[:reg_tokens]

            self.register_buffer(
                "reg_rope_cos_fp32",
                reg_rope(rope_cos_fp32),
                persistent=False,
            )
            self.register_buffer(
                "reg_rope_sin_fp32",
                reg_rope(rope_sin_fp32),
                persistent=False,
            )
            self.register_buffer(
                "reg_rope_cos_bf16",
                reg_rope(rope_cos_bf16),
                persistent=False,
            )
            self.register_buffer(
                "reg_rope_sin_bf16",
                reg_rope(rope_sin_bf16),
                persistent=False,
            )
        else:
            self.register_buffer(
                "reg_rope_cos_fp32",
                None,
                persistent=False,
            )
            self.register_buffer(
                "reg_rope_sin_fp32",
                None,
                persistent=False,
            )
            self.register_buffer(
                "reg_rope_cos_bf16",
                None,
                persistent=False,
            )
            self.register_buffer(
                "reg_rope_sin_bf16",
                None,
                persistent=False,
            )
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.reg_qkv_lora = (
            LowRankAdapter(
                hidden_size,
                3 * hidden_size,
                reg_lora.rank,
                reg_lora.alpha,
            )
            if reg_tokens and reg_lora is not None and reg_lora.enabled
            else None
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def _rope_tables(
        self,
        q: Tensor,
        *,
        reg_only: bool,
    ) -> tuple[Tensor, Tensor]:
        if q.dtype == torch.bfloat16:
            rope_cos, rope_sin = (
                (
                    self.reg_rope_cos_bf16
                    if reg_only
                    else self.rope_cos_bf16
                ),
                (
                    self.reg_rope_sin_bf16
                    if reg_only
                    else self.rope_sin_bf16
                ),
            )
        else:
            rope_cos, rope_sin = (
                (
                    self.reg_rope_cos_fp32
                    if reg_only
                    else self.rope_cos_fp32
                ),
                (
                    self.reg_rope_sin_fp32
                    if reg_only
                    else self.rope_sin_fp32
                ),
            )
            assert rope_cos is not None and rope_sin is not None
            if q.dtype != torch.float32:
                rope_cos = rope_cos.to(q.dtype)
                rope_sin = rope_sin.to(q.dtype)
        assert rope_cos is not None and rope_sin is not None
        return rope_cos, rope_sin

    def forward(
        self,
        x: Tensor,
        reg_only: bool = False,
        drop_reg_for_image: bool = False,
    ) -> Tensor:
        batch, tokens, _hidden = x.shape
        expected_tokens = (
            self.reg_tokens + self.condition_tokens
            if reg_only
            else self.expected_tokens
        )
        if reg_only and not self.independent_reg_attention:
            raise ValueError(
                "REG-only attention requires independent REG attention"
            )
        if drop_reg_for_image and (
            reg_only or not self.independent_reg_attention
        ):
            raise ValueError(
                "REG dropout requires full independent REG attention"
            )
        if tokens != expected_tokens:
            raise ValueError(
                "attention token layout does not match its precomputed "
                "DDT RoPE table"
            )
        qkv_values = self.qkv(x)
        if self.reg_qkv_lora is not None:
            qkv_values[:, : self.reg_tokens].add_(
                self.reg_qkv_lora(x[:, : self.reg_tokens])
            )
        qkv = (
            qkv_values
            .reshape(
                batch,
                tokens,
                3,
                self.num_heads,
                self.head_dim,
            )
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        rope_cos, rope_sin = self._rope_tables(q, reg_only=reg_only)
        q, k = apply_rope_cos_sin(
            q,
            k,
            rope_cos,
            rope_sin,
        )
        if reg_only or not self.independent_reg_attention:
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,
            )
        else:
            reg_end = self.reg_tokens
            image_end = reg_end + self.image_tokens
            if self.condition_tokens:
                reg_condition_q = torch.cat(
                    (q[:, :, :reg_end], q[:, :, image_end:]),
                    dim=2,
                )
                reg_condition_k = torch.cat(
                    (k[:, :, :reg_end], k[:, :, image_end:]),
                    dim=2,
                )
                reg_condition_v = torch.cat(
                    (v[:, :, :reg_end], v[:, :, image_end:]),
                    dim=2,
                )
            else:
                reg_condition_q = q[:, :, :reg_end]
                reg_condition_k = k[:, :, :reg_end]
                reg_condition_v = v[:, :, :reg_end]
            reg_condition_attended = F.scaled_dot_product_attention(
                reg_condition_q,
                reg_condition_k,
                reg_condition_v,
                is_causal=False,
            )
            image_k = (
                k[:, :, reg_end:]
                if drop_reg_for_image
                else k
            )
            image_v = (
                v[:, :, reg_end:]
                if drop_reg_for_image
                else v
            )
            image_attended = F.scaled_dot_product_attention(
                q[:, :, reg_end:image_end],
                image_k,
                image_v,
                is_causal=False,
            )
            if self.condition_tokens:
                attended = torch.cat(
                    (
                        reg_condition_attended[:, :, :reg_end],
                        image_attended,
                        reg_condition_attended[:, :, reg_end:],
                    ),
                    dim=2,
                )
            else:
                attended = torch.cat(
                    (reg_condition_attended, image_attended),
                    dim=2,
                )
        attended = attended.transpose(1, 2).reshape(
            batch,
            tokens,
            self.hidden_size,
        )
        return self.projection(attended)


class SwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: float,
        *,
        reg_tokens: int = 0,
        reg_lora: REGLoRAConfig | None = None,
    ) -> None:
        super().__init__()
        intermediate = max(
            1,
            int(2 * hidden_size * mlp_ratio / 3),
        )
        self.reg_tokens = reg_tokens
        self.gate = nn.Linear(hidden_size, intermediate)
        self.value = nn.Linear(hidden_size, intermediate)
        self.projection = nn.Linear(intermediate, hidden_size)
        lora_enabled = (
            reg_tokens
            and reg_lora is not None
            and reg_lora.enabled
            and reg_lora.mlp
        )
        self.reg_gate_lora = (
            LowRankAdapter(
                hidden_size,
                intermediate,
                reg_lora.rank,
                reg_lora.alpha,
            )
            if lora_enabled and reg_lora is not None
            else None
        )
        self.reg_value_lora = (
            LowRankAdapter(
                hidden_size,
                intermediate,
                reg_lora.rank,
                reg_lora.alpha,
            )
            if lora_enabled and reg_lora is not None
            else None
        )
        self.reg_projection_lora = (
            LowRankAdapter(
                intermediate,
                hidden_size,
                reg_lora.rank,
                reg_lora.alpha,
            )
            if lora_enabled and reg_lora is not None
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate(x)
        value = self.value(x)
        if self.reg_gate_lora is not None:
            reg_x = x[:, : self.reg_tokens]
            gate[:, : self.reg_tokens].add_(
                self.reg_gate_lora(reg_x)
            )
            assert self.reg_value_lora is not None
            value[:, : self.reg_tokens].add_(
                self.reg_value_lora(reg_x)
            )
        hidden = F.silu(gate) * value
        projected = self.projection(hidden)
        if self.reg_projection_lora is not None:
            projected[:, : self.reg_tokens].add_(
                self.reg_projection_lora(
                    hidden[:, : self.reg_tokens]
                )
            )
        return projected


class FusedSwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: float,
        *,
        reg_tokens: int = 0,
        reg_lora: REGLoRAConfig | None = None,
    ) -> None:
        super().__init__()
        intermediate = max(
            1,
            int(2 * hidden_size * mlp_ratio / 3),
        )
        self.reg_tokens = reg_tokens
        self.gate_value = nn.Linear(
            hidden_size,
            2 * intermediate,
        )
        self.projection = nn.Linear(intermediate, hidden_size)
        lora_enabled = (
            reg_tokens
            and reg_lora is not None
            and reg_lora.enabled
            and reg_lora.mlp
        )
        self.reg_gate_value_lora = (
            LowRankAdapter(
                hidden_size,
                2 * intermediate,
                reg_lora.rank,
                reg_lora.alpha,
            )
            if lora_enabled and reg_lora is not None
            else None
        )
        self.reg_projection_lora = (
            LowRankAdapter(
                intermediate,
                hidden_size,
                reg_lora.rank,
                reg_lora.alpha,
            )
            if lora_enabled and reg_lora is not None
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        gate_value = self.gate_value(x)
        if self.reg_gate_value_lora is not None:
            gate_value[:, : self.reg_tokens].add_(
                self.reg_gate_value_lora(
                    x[:, : self.reg_tokens]
                )
            )
        gate, value = gate_value.chunk(2, dim=-1)
        hidden = F.silu(gate) * value
        projected = self.projection(hidden)
        if self.reg_projection_lora is not None:
            projected[:, : self.reg_tokens].add_(
                self.reg_projection_lora(
                    hidden[:, : self.reg_tokens]
                )
            )
        return projected


def _build_swiglu(
    hidden_size: int,
    mlp_ratio: float,
    *,
    fused: bool,
    reg_tokens: int,
    reg_lora: REGLoRAConfig | None,
) -> SwiGLU | FusedSwiGLU:
    implementation = FusedSwiGLU if fused else SwiGLU
    return implementation(
        hidden_size,
        mlp_ratio,
        reg_tokens=reg_tokens,
        reg_lora=reg_lora,
    )


class DDTEncoderBlock(nn.Module):
    """Unconditioned encoder block from the DDT architecture."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        grid_size: int,
        reg_pool_size: int | None,
        condition_tokens: int,
        spatial_reg_rope: bool,
        independent_reg_attention: bool,
        reg_lora: REGLoRAConfig | None,
        fused_mlp: bool,
    ) -> None:
        super().__init__()
        reg_tokens = (
            reg_pool_size**2 if reg_pool_size is not None else 0
        )
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attention = Attention(
            hidden_size,
            num_heads,
            grid_size=grid_size,
            reg_pool_size=reg_pool_size,
            condition_tokens=condition_tokens,
            spatial_reg_rope=spatial_reg_rope,
            independent_reg_attention=independent_reg_attention,
            reg_lora=reg_lora,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.feed_forward = _build_swiglu(
            hidden_size,
            mlp_ratio,
            fused=fused_mlp,
            reg_tokens=reg_tokens,
            reg_lora=reg_lora,
        )

    def forward(
        self,
        x: Tensor,
        reg_only: bool = False,
        drop_reg_for_image: bool = False,
    ) -> Tensor:
        x = x + self.attention(
            self.norm1(x),
            reg_only=reg_only,
            drop_reg_for_image=drop_reg_for_image,
        )
        return x + self.feed_forward(self.norm2(x))


class DDTDecoderBlock(nn.Module):
    """Joint REG/image decoder block with token-aligned adaLN-Zero."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        grid_size: int,
        reg_pool_size: int | None,
        spatial_reg_rope: bool,
        independent_reg_attention: bool,
        reg_lora: REGLoRAConfig | None,
        fused_mlp: bool,
    ) -> None:
        super().__init__()
        reg_tokens = (
            reg_pool_size**2 if reg_pool_size is not None else 0
        )
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attention = Attention(
            hidden_size,
            num_heads,
            grid_size=grid_size,
            reg_pool_size=reg_pool_size,
            spatial_reg_rope=spatial_reg_rope,
            independent_reg_attention=independent_reg_attention,
            reg_lora=reg_lora,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.feed_forward = _build_swiglu(
            hidden_size,
            mlp_ratio,
            fused=fused_mlp,
            reg_tokens=reg_tokens,
            reg_lora=reg_lora,
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(
        self,
        x: Tensor,
        conditioning: Tensor,
        reg_only: bool = False,
        drop_reg_for_image: bool = False,
    ) -> Tensor:
        if x.shape != conditioning.shape:
            raise ValueError(
                "DDT decoder states and token-aligned conditioning must "
                "have equal shapes"
            )
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.modulation(conditioning).chunk(6, dim=-1)
        )
        attended = self.attention(
            _modulate(self.norm1(x), shift_a, scale_a),
            reg_only=reg_only,
            drop_reg_for_image=drop_reg_for_image,
        )
        x = x + _gate(attended, gate_a)
        fed_forward = self.feed_forward(
            _modulate(self.norm2(x), shift_m, scale_m)
        )
        return x + _gate(fed_forward, gate_m)


class DDTFinalLayer(nn.Module):
    """Shared final normalization/modulation with typed output projections."""

    def __init__(
        self,
        hidden_size: int,
        image_output_size: int,
        reg_output_size: int | None,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.image_projection = nn.Linear(
            hidden_size,
            image_output_size,
        )
        self.reg_projection = (
            nn.Linear(hidden_size, reg_output_size)
            if reg_output_size is not None
            else None
        )

    def forward(
        self,
        x: Tensor,
        conditioning: Tensor,
        reg_tokens: int,
    ) -> tuple[Tensor, Tensor | None]:
        if x.shape != conditioning.shape:
            raise ValueError(
                "DDT final states and token-aligned conditioning must "
                "have equal shapes"
            )
        if not 0 <= reg_tokens <= x.shape[1]:
            raise ValueError("DDT final REG token count is invalid")
        shift, scale = self.modulation(conditioning).chunk(2, dim=-1)
        decoded = _modulate(self.norm(x), shift, scale)
        image = self.image_projection(decoded[:, reg_tokens:])
        if reg_tokens:
            if self.reg_projection is None:
                raise ValueError(
                    "DDT final REG projection is not configured"
                )
            reg = self.reg_projection(decoded[:, :reg_tokens])
        else:
            reg = None
        return image, reg

    def forward_reg(
        self,
        x: Tensor,
        conditioning: Tensor,
    ) -> Tensor:
        if self.reg_projection is None:
            raise ValueError(
                "DDT final REG projection is not configured"
            )
        if x.shape != conditioning.shape:
            raise ValueError(
                "DDT final REG states and conditioning must have equal shapes"
            )
        shift, scale = self.modulation(conditioning).chunk(2, dim=-1)
        decoded = _modulate(self.norm(x), shift, scale)
        return self.reg_projection(decoded)


class DDTInternalREGHead(nn.Module):
    """Predict clean REG tokens from an intermediate encoder state."""

    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.projection = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        if x.shape != conditioning.shape:
            raise ValueError(
                "internal REG states and conditioning must have equal shapes"
            )
        shift, scale = self.modulation(conditioning).chunk(2, dim=-1)
        return self.projection(_modulate(self.norm(x), shift, scale))


class REPAProjection(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DDT(nn.Module):
    """DDT over pixel or normalized FLUX.2 latent tensors."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.input_size < 1 or config.patch_size < 1:
            raise ValueError("image size and patch size must be positive")
        if config.input_size % config.patch_size:
            raise ValueError(
                "image size must be divisible by patch size"
            )
        if (
            config.hidden_size < 1
            or config.num_heads < 1
            or config.depth < 1
        ):
            raise ValueError("DDT encoder dimensions must be positive")
        if config.hidden_size % config.num_heads:
            raise ValueError(
                "encoder hidden size must be divisible by number of heads"
            )
        if (config.hidden_size // config.num_heads) % 4:
            raise ValueError(
                "encoder attention head dimension must be divisible by four"
            )
        decoder = config.decoder
        if (
            decoder.hidden_size < 1
            or decoder.num_heads < 1
            or decoder.depth < 1
        ):
            raise ValueError("DDT decoder dimensions must be positive")
        if decoder.hidden_size % decoder.num_heads:
            raise ValueError(
                "decoder hidden size must be divisible by number of heads"
            )
        if (decoder.hidden_size // decoder.num_heads) % 4:
            raise ValueError(
                "decoder attention head dimension must be divisible by four"
            )
        features = config.features
        reg = features.reg
        repa = features.repa
        grid_size = config.input_size // config.patch_size
        token_counts = (
            config.num_img_t_tokens,
            config.num_reg_t_tokens,
            config.num_c_tokens,
        )
        if any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for count in token_counts
        ):
            raise ValueError(
                "conditioning token counts must be non-negative integers"
            )
        if config.num_reg_t_tokens and not reg.enabled:
            raise ValueError(
                "REG-time tokens require REG to be enabled"
            )
        if reg.enabled and not 1 <= reg.pool_size <= grid_size:
            raise ValueError(
                "REG pool size must not exceed the model grid"
            )
        if reg.enabled and (
            not isinstance(reg.internal_guidance_block, int)
            or isinstance(reg.internal_guidance_block, bool)
            or not 1 <= reg.internal_guidance_block <= config.depth
        ):
            raise ValueError(
                "REG internal-guidance block must be within encoder depth"
            )
        if reg.enabled and reg.lora.enabled:
            if (
                not isinstance(reg.lora.rank, int)
                or isinstance(reg.lora.rank, bool)
                or reg.lora.rank < 1
            ):
                raise ValueError(
                    "REG LoRA rank must be a positive integer"
                )
            if not math.isfinite(reg.lora.alpha) or reg.lora.alpha <= 0:
                raise ValueError(
                    "REG LoRA alpha must be finite and positive"
                )
        if repa.enabled and not 1 <= repa.block <= config.depth:
            raise ValueError(
                "REPA block must be within encoder depth"
            )
        if (
            repa.enabled
            and repa.token_scope == "image_and_reg"
            and not reg.enabled
        ):
            raise ValueError(
                "REPA image_and_reg scope requires REG"
            )

        self.config = config
        self.grid_size = grid_size
        self.encoder_patch_projection = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.decoder_patch_projection = nn.Conv2d(
            config.in_channels,
            decoder.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.img_t_embedder = GaussianFourierEmbedding(
            config.hidden_size,
            n_tokens=config.num_img_t_tokens,
        )
        self.reg_t_embedder = (
            GaussianFourierEmbedding(
                config.hidden_size,
                n_tokens=config.num_reg_t_tokens,
            )
            if reg.enabled
            else None
        )
        self.condition_embedder = ConditionEmbedder(
            config.hidden_size,
            num_classes=config.num_classes,
            condition_type="label",
            n_tokens=config.num_c_tokens,
        )
        reg_pool_size = reg.pool_size if reg.enabled else None
        spatial_reg_rope = reg.enabled and reg.spatial_rope
        reg_lora = reg.lora if reg.enabled else None
        self.encoder_blocks = nn.ModuleList(
            [
                DDTEncoderBlock(
                    config.hidden_size,
                    config.num_heads,
                    config.mlp_ratio,
                    grid_size=self.grid_size,
                    reg_pool_size=reg_pool_size,
                    condition_tokens=self.num_condition_tokens,
                    spatial_reg_rope=spatial_reg_rope,
                    independent_reg_attention=(
                        reg.enabled and reg.independent_attention
                    ),
                    reg_lora=reg_lora,
                    fused_mlp=config.fused_mlp,
                )
                for _ in range(config.depth)
            ]
        )
        self.context_projection: nn.Module = (
            nn.Linear(
                config.hidden_size,
                decoder.hidden_size,
            )
            if config.hidden_size != decoder.hidden_size
            else nn.Identity()
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DDTDecoderBlock(
                    decoder.hidden_size,
                    decoder.num_heads,
                    decoder.mlp_ratio,
                    grid_size=self.grid_size,
                    reg_pool_size=reg_pool_size,
                    spatial_reg_rope=spatial_reg_rope,
                    independent_reg_attention=(
                        reg.enabled and reg.independent_attention
                    ),
                    reg_lora=reg_lora,
                    fused_mlp=config.fused_mlp,
                )
                for _ in range(decoder.depth)
            ]
        )
        patch_values = (
            config.patch_size
            * config.patch_size
            * config.in_channels
        )
        feature_dim = features.encoder.feature_dim
        self.final_layer = DDTFinalLayer(
            decoder.hidden_size,
            patch_values,
            feature_dim if reg.enabled else None,
        )
        self.internal_reg_head: DDTInternalREGHead | None = (
            DDTInternalREGHead(config.hidden_size, feature_dim)
            if reg.enabled
            else None
        )
        if reg.enabled:
            self.encoder_reg_projection: nn.Linear | None = nn.Linear(
                feature_dim,
                config.hidden_size,
            )
            self.decoder_reg_projection: nn.Linear | None = nn.Linear(
                feature_dim,
                decoder.hidden_size,
            )
        else:
            self.encoder_reg_projection = None
            self.decoder_reg_projection = None
        self.repa_projection: REPAProjection | None = (
            REPAProjection(
                config.hidden_size,
                repa.projection_hidden_size,
                feature_dim,
            )
            if repa.enabled
            else None
        )
        self.gradient_checkpointing = config.gradient_checkpointing
        self._initialize_weights()

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def num_reg_tokens(self) -> int:
        reg = self.config.features.reg
        return reg.num_tokens if reg.enabled else 0

    @property
    def num_primary_tokens(self) -> int:
        return self.num_reg_tokens + self.num_patches

    @property
    def num_condition_tokens(self) -> int:
        return (
            self.config.num_reg_t_tokens
            + self.config.num_img_t_tokens
            + self.config.num_c_tokens
        )

    @property
    def image_token_start(self) -> int:
        return self.num_reg_tokens

    @property
    def condition_token_start(self) -> int:
        return self.num_primary_tokens

    def _initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        for module in self.modules():
            if isinstance(module, LowRankAdapter):
                module.reset_parameters()
        for projection in (
            self.encoder_patch_projection,
            self.decoder_patch_projection,
        ):
            nn.init.xavier_uniform_(projection.weight.flatten(1))
            nn.init.zeros_(projection.bias)
        if self.condition_embedder.embedding_table is not None:
            nn.init.normal_(
                self.condition_embedder.embedding_table.weight,
                std=0.02,
            )
        time_embedders = [self.img_t_embedder]
        if self.reg_t_embedder is not None:
            time_embedders.append(self.reg_t_embedder)
        for embedder in time_embedders:
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)
        for block in self.decoder_blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.modulation[-1].weight)
        nn.init.zeros_(self.final_layer.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.image_projection.weight)
        nn.init.zeros_(self.final_layer.image_projection.bias)
        if self.final_layer.reg_projection is not None:
            nn.init.zeros_(self.final_layer.reg_projection.weight)
            nn.init.zeros_(self.final_layer.reg_projection.bias)
        if self.internal_reg_head is not None:
            nn.init.zeros_(self.internal_reg_head.modulation[-1].weight)
            nn.init.zeros_(self.internal_reg_head.modulation[-1].bias)
            nn.init.zeros_(self.internal_reg_head.projection.weight)
            nn.init.zeros_(self.internal_reg_head.projection.bias)

    def set_gradient_checkpointing(
        self,
        enabled: bool = True,
    ) -> None:
        self.gradient_checkpointing = enabled

    def _checkpointing(self) -> bool:
        return (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )

    def _unpatchify(self, tokens: Tensor) -> Tensor:
        batch = tokens.shape[0]
        patch = self.config.patch_size
        channels = self.config.in_channels
        values = tokens.reshape(
            batch,
            self.grid_size,
            self.grid_size,
            patch,
            patch,
            channels,
        )
        return torch.einsum(
            "bhwpqc->bchpwq",
            values,
        ).reshape(
            batch,
            channels,
            self.config.input_size,
            self.config.input_size,
        )

    def forward_reg(
        self,
        noisy_reg: Tensor,
        image_timestep: Tensor,
        labels: Tensor,
        *,
        reg_timestep: Tensor,
    ) -> DDTREGOutput:
        reg_config = self.config.features.reg
        if not reg_config.enabled or not reg_config.independent_attention:
            raise ValueError(
                "REG-only forward requires independent REG attention"
            )
        batch = noisy_reg.shape[0]
        expected_reg = (
            batch,
            self.num_reg_tokens,
            self.config.features.encoder.feature_dim,
        )
        if noisy_reg.ndim != 3 or tuple(noisy_reg.shape) != expected_reg:
            raise ValueError(
                "REG-only forward requires noisy_reg with shape "
                f"[B, {self.num_reg_tokens}, "
                f"{self.config.features.encoder.feature_dim}]"
            )
        for name, value in (
            ("image_timestep", image_timestep),
            ("reg_timestep", reg_timestep),
            ("labels", labels),
        ):
            if value.shape != (batch,):
                raise ValueError(f"{name} must have shape [B]")
        assert (
            self.reg_t_embedder is not None
            and self.encoder_reg_projection is not None
            and self.decoder_reg_projection is not None
            and self.internal_reg_head is not None
        )
        reg_time_result = self.reg_t_embedder(
            reg_timestep,
            return_base_embed=True,
        )
        assert isinstance(reg_time_result, tuple)
        reg_time_base, reg_time_tokens = reg_time_result
        image_time_tokens = self.img_t_embedder(image_timestep)
        assert isinstance(image_time_tokens, Tensor)
        class_tokens = self.condition_embedder(labels)
        sequence = torch.cat(
            (
                self.encoder_reg_projection(noisy_reg),
                reg_time_tokens,
                image_time_tokens,
                class_tokens,
            ),
            dim=1,
        )

        early_reg_state: Tensor | None = None
        for index, block in enumerate(self.encoder_blocks, start=1):
            sequence = block(sequence, reg_only=True)
            if index == reg_config.internal_guidance_block:
                early_reg_state = sequence[:, : self.num_reg_tokens]
        if early_reg_state is None:
            raise RuntimeError(
                "REG-only forward did not capture its guidance state"
            )
        encoded_reg = sequence[:, : self.num_reg_tokens]
        early_reg_context = F.silu(
            early_reg_state + reg_time_base
        )
        reg_early = self.internal_reg_head(
            early_reg_context,
            early_reg_context,
        )
        decoder_context = self.context_projection(
            F.silu(encoded_reg + reg_time_base)
        )
        decoded = self.decoder_reg_projection(noisy_reg)
        for block in self.decoder_blocks:
            decoded = block(
                decoded,
                decoder_context,
                reg_only=True,
            )
        reg = self.final_layer.forward_reg(
            decoded,
            decoder_context,
        )
        return DDTREGOutput(reg=reg, reg_early=reg_early)

    def forward(
        self,
        noisy_image: Tensor,
        timestep: Tensor,
        labels: Tensor,
        noisy_reg: Tensor | None = None,
        reg_timestep: Tensor | None = None,
        return_repa: bool = False,
        drop_reg_for_image: bool = False,
    ) -> DDTOutput:
        expected = (
            noisy_image.shape[0],
            self.config.in_channels,
            self.config.input_size,
            self.config.input_size,
        )
        if noisy_image.ndim != 4 or tuple(noisy_image.shape) != expected:
            raise ValueError(
                "noisy_image must have shape "
                f"[B, {self.config.in_channels}, "
                f"{self.config.input_size}, {self.config.input_size}]"
            )
        batch = noisy_image.shape[0]
        if timestep.shape != (batch,) or labels.shape != (batch,):
            raise ValueError(
                "timestep and labels must both have shape [B]"
            )
        reg_config = self.config.features.reg
        if drop_reg_for_image and (
            not reg_config.enabled
            or not reg_config.independent_attention
        ):
            raise ValueError(
                "REG dropout requires independent REG attention"
            )

        image_time_result = self.img_t_embedder(
            timestep,
            return_base_embed=True,
        )
        assert isinstance(image_time_result, tuple)
        image_time_base, image_time_tokens = image_time_result
        class_tokens = self.condition_embedder(labels)
        encoder_image = (
            self.encoder_patch_projection(noisy_image)
            .flatten(2)
            .transpose(1, 2)
        )
        decoder_image = (
            self.decoder_patch_projection(noisy_image)
            .flatten(2)
            .transpose(1, 2)
        )

        encoder_reg: Tensor | None = None
        decoder_reg: Tensor | None = None
        reg_time_base: Tensor | None = None
        reg_time_tokens: Tensor | None = None
        if reg_config.enabled:
            expected_reg = (
                batch,
                self.num_reg_tokens,
                self.config.features.encoder.feature_dim,
            )
            if noisy_reg is None or tuple(noisy_reg.shape) != expected_reg:
                raise ValueError(
                    "REG requires noisy_reg with shape "
                    f"[B, {self.num_reg_tokens}, "
                    f"{self.config.features.encoder.feature_dim}]"
                )
            if reg_timestep is None or reg_timestep.shape != (batch,):
                raise ValueError(
                    "REG requires reg_timestep with shape [B]"
                )
            assert (
                self.reg_t_embedder is not None
                and self.encoder_reg_projection is not None
                and self.decoder_reg_projection is not None
            )
            reg_time_result = self.reg_t_embedder(
                reg_timestep,
                return_base_embed=True,
            )
            assert isinstance(reg_time_result, tuple)
            reg_time_base, reg_time_tokens = reg_time_result
            encoder_reg = self.encoder_reg_projection(noisy_reg)
            decoder_reg = self.decoder_reg_projection(noisy_reg)
        elif noisy_reg is not None or reg_timestep is not None:
            raise ValueError(
                "noisy_reg and reg_timestep are only valid when REG is enabled"
            )

        sequence_parts = []
        if encoder_reg is not None:
            sequence_parts.append(encoder_reg)
        sequence_parts.append(encoder_image)
        for tokens in (
            reg_time_tokens,
            image_time_tokens,
            class_tokens,
        ):
            if tokens is not None:
                sequence_parts.append(tokens)
        sequence = torch.cat(sequence_parts, dim=1)

        repa: Tensor | None = None
        early_reg_state: Tensor | None = None
        for index, block in enumerate(
            self.encoder_blocks,
            start=1,
        ):
            if self._checkpointing():
                sequence = checkpoint(
                    block,
                    sequence,
                    False,
                    drop_reg_for_image,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                sequence = block(
                    sequence,
                    drop_reg_for_image=drop_reg_for_image,
                )
            if (
                self.internal_reg_head is not None
                and index == reg_config.internal_guidance_block
            ):
                early_reg_state = sequence[:, : self.num_reg_tokens]
            if (
                return_repa
                and self.repa_projection is not None
                and index == self.config.features.repa.block
            ):
                repa_tokens = (
                    sequence[:, : self.num_primary_tokens]
                    if (
                        self.config.features.repa.token_scope
                        == "image_and_reg"
                    )
                    else sequence[
                        :,
                        self.image_token_start : self.num_primary_tokens,
                    ]
                )
                repa = self.repa_projection(repa_tokens)

        encoded_primary = sequence[:, : self.num_primary_tokens]
        context_parts = []
        reg_early: Tensor | None = None
        if self.num_reg_tokens:
            assert (
                reg_time_base is not None
                and early_reg_state is not None
                and self.internal_reg_head is not None
            )
            early_reg_context = F.silu(early_reg_state + reg_time_base)
            reg_early = self.internal_reg_head(
                early_reg_context,
                early_reg_context,
            )
            context_parts.append(
                F.silu(
                    encoded_primary[:, : self.num_reg_tokens]
                    + reg_time_base
                )
            )
        context_parts.append(
            F.silu(
                encoded_primary[:, self.image_token_start :]
                + image_time_base
            )
        )
        decoder_context = self.context_projection(
            torch.cat(context_parts, dim=1)
        )

        decoder_parts = []
        if decoder_reg is not None:
            decoder_parts.append(decoder_reg)
        decoder_parts.append(decoder_image)
        decoded = torch.cat(decoder_parts, dim=1)
        for block in self.decoder_blocks:
            if self._checkpointing():
                decoded = checkpoint(
                    block,
                    decoded,
                    decoder_context,
                    False,
                    drop_reg_for_image,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                decoded = block(
                    decoded,
                    decoder_context,
                    drop_reg_for_image=drop_reg_for_image,
                )
        image_tokens, reg = self.final_layer(
            decoded,
            decoder_context,
            self.num_reg_tokens,
        )
        image = self._unpatchify(image_tokens)
        return DDTOutput(
            image=image,
            reg=reg,
            repa=repa,
            reg_early=reg_early,
        )
