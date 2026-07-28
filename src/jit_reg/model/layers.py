from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import REGLoRAConfig
from .rope import apply_rope_cos_sin, precompute_ddt_rope_tables
from .tokens import DDTTokenLayout


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


def _optional_lora(
    input_size: int,
    output_size: int,
    config: REGLoRAConfig | None,
    *,
    mlp: bool = False,
) -> LowRankAdapter | None:
    enabled = (
        config is not None
        and config.enabled
        and (not mlp or config.mlp)
    )
    return (
        LowRankAdapter(
            input_size,
            output_size,
            config.rank,
            config.alpha,
        )
        if enabled and config is not None
        else None
    )


def _table_stream(
    table: Tensor,
    slices: tuple[slice, ...],
) -> Tensor:
    parts = [
        table[token_slice]
        for token_slice in slices
        if token_slice.stop > token_slice.start
    ]
    return torch.cat(parts) if parts else table[:0]


class DDTAttention(nn.Module):
    """Joint or leak-free asymmetric attention over a fixed token layout."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        grid_size: int,
        layout: DDTTokenLayout,
        reg_pool_size: int | None = None,
        spatial_reg_rope: bool = False,
        attention_mode: str = "joint",
        reg_lora: REGLoRAConfig | None = None,
    ) -> None:
        super().__init__()
        if attention_mode not in {"joint", "asymmetric"}:
            raise ValueError(
                "attention_mode must be joint or asymmetric"
            )
        if hidden_size % num_heads:
            raise ValueError(
                "attention hidden size must be divisible by heads"
            )
        if layout.image_data_tokens != grid_size**2:
            raise ValueError(
                "attention image layout must match the spatial grid"
            )
        expected_reg = (
            reg_pool_size**2 if reg_pool_size is not None else 0
        )
        if layout.reg_data_tokens != expected_reg:
            raise ValueError(
                "attention REG layout must match the REG pool"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layout = layout
        self.attention_mode = attention_mode
        self.expected_tokens = layout.total_tokens

        for dtype_name, dtype in (
            ("fp32", torch.float32),
            ("bf16", torch.bfloat16),
        ):
            cos, sin = precompute_ddt_rope_tables(
                grid_size,
                grid_size,
                self.head_dim,
                reg_pool_size=reg_pool_size,
                condition_tokens=layout.condition_tokens,
                spatial_reg_rope=spatial_reg_rope,
                dtype=dtype,
            )
            self.register_buffer(
                f"rope_cos_{dtype_name}",
                cos,
                persistent=False,
            )
            self.register_buffer(
                f"rope_sin_{dtype_name}",
                sin,
                persistent=False,
            )
            self.register_buffer(
                f"reg_rope_cos_{dtype_name}",
                _table_stream(cos, layout.reg_stream_slices),
                persistent=False,
            )
            self.register_buffer(
                f"reg_rope_sin_{dtype_name}",
                _table_stream(sin, layout.reg_stream_slices),
                persistent=False,
            )

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.reg_qkv_lora = (
            _optional_lora(
                hidden_size,
                3 * hidden_size,
                reg_lora,
            )
            if layout.reg_data_tokens
            else None
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def _rope_tables(
        self,
        q: Tensor,
        *,
        reg_stream: bool,
    ) -> tuple[Tensor, Tensor]:
        suffix = "bf16" if q.dtype == torch.bfloat16 else "fp32"
        prefix = "reg_rope" if reg_stream else "rope"
        cos = getattr(self, f"{prefix}_cos_{suffix}")
        sin = getattr(self, f"{prefix}_sin_{suffix}")
        if q.dtype not in {torch.float32, torch.bfloat16}:
            cos = cos.to(q.dtype)
            sin = sin.to(q.dtype)
        return cos, sin

    def _qkv(
        self,
        x: Tensor,
        *,
        reg_data_tokens: int,
        reg_stream: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, tokens, _hidden = x.shape
        values = self.qkv(x)
        if self.reg_qkv_lora is not None and reg_data_tokens:
            lora_values = self.reg_qkv_lora(
                x[:, :reg_data_tokens]
            )
            values = torch.cat(
                (
                    values[:, :reg_data_tokens] + lora_values,
                    values[:, reg_data_tokens:],
                ),
                dim=1,
            )
        qkv = (
            values.reshape(
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
        cos, sin = self._rope_tables(
            q,
            reg_stream=reg_stream,
        )
        q, k = apply_rope_cos_sin(q, k, cos, sin)
        return q, k, v

    @staticmethod
    def _attend(
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        if q.shape[-2] == 0:
            return q
        if k.shape[-2] == 0:
            raise ValueError(
                "non-empty attention queries require at least one K/V token"
            )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
        )

    def _finish(self, attended: Tensor) -> Tensor:
        batch, _heads, tokens, _head_dim = attended.shape
        return self.projection(
            attended.transpose(1, 2).reshape(
                batch,
                tokens,
                self.hidden_size,
            )
        )

    def forward(
        self,
        x: Tensor,
        image_attends_reg: bool = True,
    ) -> Tensor:
        if x.shape[-2] != self.layout.total_tokens:
            raise ValueError(
                "attention token count does not match its DDTTokenLayout"
            )
        if (
            not image_attends_reg
            and self.attention_mode != "asymmetric"
        ):
            raise ValueError(
                "stream isolation requires asymmetric attention"
            )
        q, k, v = self._qkv(
            x,
            reg_data_tokens=self.layout.reg_data_tokens,
            reg_stream=False,
        )
        if self.attention_mode == "joint":
            return self._finish(self._attend(q, k, v))

        reg_q = self.layout.gather_reg_stream(q)
        reg_k = self.layout.gather_reg_stream(k)
        reg_v = self.layout.gather_reg_stream(v)
        reg_attended = self._attend(reg_q, reg_k, reg_v)

        image_k = self.layout.gather_image_stream(k)
        image_v = self.layout.gather_image_stream(v)
        image_data_q = q[..., self.layout.image_data, :]
        image_data_attended = self._attend(
            image_data_q,
            k if image_attends_reg else image_k,
            v if image_attends_reg else image_v,
        )
        image_condition_q = self.layout.gather_image_conditions(q)
        image_condition_attended = self._attend(
            image_condition_q,
            image_k,
            image_v,
        )
        image_attended = torch.cat(
            (image_data_attended, image_condition_attended),
            dim=-2,
        )
        attended = self.layout.reconstruct_streams(
            reg_attended,
            image_attended,
        )
        return self._finish(attended)

    def forward_reg(self, x: Tensor) -> Tensor:
        if self.attention_mode != "asymmetric":
            raise ValueError(
                "REG-only attention requires asymmetric attention"
            )
        if x.shape[-2] != self.layout.reg_stream_tokens:
            raise ValueError(
                "REG-only token count does not match its DDTTokenLayout"
            )
        q, k, v = self._qkv(
            x,
            reg_data_tokens=self.layout.reg_data_tokens,
            reg_stream=True,
        )
        return self._finish(self._attend(q, k, v))


def _intermediate_size(
    hidden_size: int,
    mlp_ratio: float,
) -> int:
    return max(1, int(2 * hidden_size * mlp_ratio / 3))


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
        intermediate = _intermediate_size(hidden_size, mlp_ratio)
        self.reg_tokens = reg_tokens
        self.gate = nn.Linear(hidden_size, intermediate)
        self.value = nn.Linear(hidden_size, intermediate)
        self.projection = nn.Linear(intermediate, hidden_size)
        self.reg_gate_lora = (
            _optional_lora(
                hidden_size,
                intermediate,
                reg_lora,
                mlp=True,
            )
            if reg_tokens
            else None
        )
        self.reg_value_lora = (
            _optional_lora(
                hidden_size,
                intermediate,
                reg_lora,
                mlp=True,
            )
            if reg_tokens
            else None
        )
        self.reg_projection_lora = (
            _optional_lora(
                intermediate,
                hidden_size,
                reg_lora,
                mlp=True,
            )
            if reg_tokens
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate(x)
        value = self.value(x)
        if self.reg_gate_lora is not None:
            reg_x = x[:, : self.reg_tokens]
            gate = torch.cat(
                (
                    gate[:, : self.reg_tokens]
                    + self.reg_gate_lora(reg_x),
                    gate[:, self.reg_tokens :],
                ),
                dim=1,
            )
            assert self.reg_value_lora is not None
            value = torch.cat(
                (
                    value[:, : self.reg_tokens]
                    + self.reg_value_lora(reg_x),
                    value[:, self.reg_tokens :],
                ),
                dim=1,
            )
        hidden = F.silu(gate) * value
        projected = self.projection(hidden)
        if self.reg_projection_lora is not None:
            projected = torch.cat(
                (
                    projected[:, : self.reg_tokens]
                    + self.reg_projection_lora(
                        hidden[:, : self.reg_tokens]
                    ),
                    projected[:, self.reg_tokens :],
                ),
                dim=1,
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
        intermediate = _intermediate_size(hidden_size, mlp_ratio)
        self.reg_tokens = reg_tokens
        self.gate_value = nn.Linear(
            hidden_size,
            2 * intermediate,
        )
        self.projection = nn.Linear(intermediate, hidden_size)
        self.reg_gate_value_lora = (
            _optional_lora(
                hidden_size,
                2 * intermediate,
                reg_lora,
                mlp=True,
            )
            if reg_tokens
            else None
        )
        self.reg_projection_lora = (
            _optional_lora(
                intermediate,
                hidden_size,
                reg_lora,
                mlp=True,
            )
            if reg_tokens
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        gate_value = self.gate_value(x)
        if self.reg_gate_value_lora is not None:
            gate_value = torch.cat(
                (
                    gate_value[:, : self.reg_tokens]
                    + self.reg_gate_value_lora(
                        x[:, : self.reg_tokens]
                    ),
                    gate_value[:, self.reg_tokens :],
                ),
                dim=1,
            )
        gate, value = gate_value.chunk(2, dim=-1)
        hidden = F.silu(gate) * value
        projected = self.projection(hidden)
        if self.reg_projection_lora is not None:
            projected = torch.cat(
                (
                    projected[:, : self.reg_tokens]
                    + self.reg_projection_lora(
                        hidden[:, : self.reg_tokens]
                    ),
                    projected[:, self.reg_tokens :],
                ),
                dim=1,
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
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        grid_size: int,
        layout: DDTTokenLayout,
        reg_pool_size: int | None,
        spatial_reg_rope: bool,
        attention_mode: str,
        reg_lora: REGLoRAConfig | None,
        fused_mlp: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attention = DDTAttention(
            hidden_size,
            num_heads,
            grid_size=grid_size,
            layout=layout,
            reg_pool_size=reg_pool_size,
            spatial_reg_rope=spatial_reg_rope,
            attention_mode=attention_mode,
            reg_lora=reg_lora,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.feed_forward = _build_swiglu(
            hidden_size,
            mlp_ratio,
            fused=fused_mlp,
            reg_tokens=layout.reg_data_tokens,
            reg_lora=reg_lora,
        )

    def forward(
        self,
        x: Tensor,
        image_attends_reg: bool = True,
    ) -> Tensor:
        x = x + self.attention(
            self.norm1(x),
            image_attends_reg=image_attends_reg,
        )
        return x + self.feed_forward(self.norm2(x))

    def forward_reg(self, x: Tensor) -> Tensor:
        x = x + self.attention.forward_reg(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class DDTDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        *,
        grid_size: int,
        layout: DDTTokenLayout,
        reg_pool_size: int | None,
        spatial_reg_rope: bool,
        attention_mode: str,
        reg_lora: REGLoRAConfig | None,
        fused_mlp: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attention = DDTAttention(
            hidden_size,
            num_heads,
            grid_size=grid_size,
            layout=layout,
            reg_pool_size=reg_pool_size,
            spatial_reg_rope=spatial_reg_rope,
            attention_mode=attention_mode,
            reg_lora=reg_lora,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.feed_forward = _build_swiglu(
            hidden_size,
            mlp_ratio,
            fused=fused_mlp,
            reg_tokens=layout.reg_data_tokens,
            reg_lora=reg_lora,
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def _forward(
        self,
        x: Tensor,
        conditioning: Tensor,
        *,
        reg_stream: bool,
        image_attends_reg: bool,
    ) -> Tensor:
        if x.shape != conditioning.shape:
            raise ValueError(
                "DDT decoder states and token-aligned conditioning must "
                "have equal shapes"
            )
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.modulation(conditioning).chunk(6, dim=-1)
        )
        normalized = _modulate(
            self.norm1(x),
            shift_a,
            scale_a,
        )
        attended = (
            self.attention.forward_reg(normalized)
            if reg_stream
            else self.attention(
                normalized,
                image_attends_reg=image_attends_reg,
            )
        )
        x = x + _gate(attended, gate_a)
        fed_forward = self.feed_forward(
            _modulate(self.norm2(x), shift_m, scale_m)
        )
        return x + _gate(fed_forward, gate_m)

    def forward(
        self,
        x: Tensor,
        conditioning: Tensor,
        image_attends_reg: bool = True,
    ) -> Tensor:
        return self._forward(
            x,
            conditioning,
            reg_stream=False,
            image_attends_reg=image_attends_reg,
        )

    def forward_reg(
        self,
        x: Tensor,
        conditioning: Tensor,
    ) -> Tensor:
        return self._forward(
            x,
            conditioning,
            reg_stream=True,
            image_attends_reg=False,
        )


class DDTFinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        image_output_size: int,
        reg_output_size: int | None,
        layout: DDTTokenLayout,
    ) -> None:
        super().__init__()
        self.layout = layout
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

    def _decode(
        self,
        x: Tensor,
        conditioning: Tensor,
    ) -> Tensor:
        if x.shape != conditioning.shape:
            raise ValueError(
                "DDT final states and conditioning must have equal shapes"
            )
        shift, scale = self.modulation(conditioning).chunk(2, dim=-1)
        return _modulate(self.norm(x), shift, scale)

    def _project_reg(self, decoded: Tensor) -> Tensor:
        if self.reg_projection is None:
            raise ValueError(
                "DDT final REG projection is not configured"
            )
        return self.reg_projection(decoded)

    def forward(
        self,
        x: Tensor,
        conditioning: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        if x.shape[-2] != self.layout.total_tokens:
            raise ValueError(
                "DDT final states do not match decoder token layout"
            )
        decoded = self._decode(x, conditioning)
        image = self.image_projection(
            decoded[:, self.layout.image_data]
        )
        reg = None
        if self.layout.reg_data_tokens:
            reg = self._project_reg(
                decoded[:, self.layout.reg_data]
            )
        return image, reg

    def forward_reg(
        self,
        x: Tensor,
        conditioning: Tensor,
    ) -> Tensor:
        return self._project_reg(
            self._decode(x, conditioning)
        )


class DDTInternalREGHead(nn.Module):
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
        return self.projection(
            _modulate(self.norm(x), shift, scale)
        )


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
