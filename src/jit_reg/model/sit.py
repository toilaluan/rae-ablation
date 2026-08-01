from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig, REPAConfig
from .rope import apply_rope_cos_sin, precompute_rope_tables


class SiTOutput(NamedTuple):
    x_prediction: Tensor
    repa_prediction: Tensor | None = None


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
        if timestep.ndim != 1:
            raise ValueError("timestep must have shape [B]")
        half = self.frequency_size // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / half
        )
        angles = timestep.float()[:, None] * 1_000 * frequencies[None]
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if self.frequency_size % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class SiTAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, grid_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.projection = nn.Linear(hidden_size, hidden_size)
        for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            cos, sin = precompute_rope_tables(
                grid_size,
                grid_size,
                self.head_dim,
                dtype=dtype,
            )
            self.register_buffer(f"rope_cos_{name}", cos, persistent=False)
            self.register_buffer(f"rope_sin_{name}", sin, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        suffix = "bf16" if q.dtype == torch.bfloat16 else "fp32"
        cos = getattr(self, f"rope_cos_{suffix}")
        sin = getattr(self, f"rope_sin_{suffix}")
        if cos.dtype != q.dtype:
            cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q, k = apply_rope_cos_sin(q, k, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.projection(
            attended.transpose(1, 2).reshape(batch, tokens, self.hidden_size)
        )


class FusedSwiGLU(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        intermediate = max(1, int(2 * hidden_size * mlp_ratio / 3))
        self.gate_value = nn.Linear(hidden_size, 2 * intermediate)
        self.projection = nn.Linear(intermediate, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.gate_value(x).chunk(2, dim=-1)
        return self.projection(F.silu(gate) * value)


class SiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        grid_size: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attention = SiTAttention(hidden_size, num_heads, grid_size)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.feed_forward = FusedSwiGLU(hidden_size, mlp_ratio)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.modulation(conditioning).chunk(6, dim=-1)
        )
        x = x + gate_a.unsqueeze(1) * self.attention(
            _modulate(self.norm1(x), shift_a, scale_a)
        )
        return x + gate_m.unsqueeze(1) * self.feed_forward(
            _modulate(self.norm2(x), shift_m, scale_m)
        )


class SiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.projection = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale = self.modulation(conditioning).chunk(2, dim=-1)
        return self.projection(_modulate(self.norm(x), shift, scale))


class REPAProjection(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
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


class SiT(nn.Module):
    """Class-conditional SiT that predicts clean x0 in pixel or VAE space."""

    def __init__(self, config: ModelConfig, repa_config: REPAConfig | None = None) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.repa_config = repa_config or REPAConfig()
        if self.repa_config.enabled and not 1 <= self.repa_config.block <= config.depth:
            raise ValueError("repa.block must be within model.depth")
        self.grid_size = config.grid_size
        self.patch_projection = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.timestep_embedder = TimestepEmbedder(config.hidden_size)
        self.class_embedder = nn.Embedding(config.num_classes + 1, config.hidden_size)
        self.blocks = nn.ModuleList(
            SiTBlock(
                config.hidden_size,
                config.num_heads,
                config.mlp_ratio,
                config.grid_size,
            )
            for _ in range(config.depth)
        )
        patch_values = config.patch_size**2 * config.in_channels
        self.final_layer = SiTFinalLayer(config.hidden_size, patch_values)
        self.repa_projection = (
            REPAProjection(
                config.hidden_size,
                self.repa_config.projection_hidden_size,
                self.repa_config.encoder.feature_dim,
            )
            if self.repa_config.enabled
            else None
        )
        self.gradient_checkpointing = config.gradient_checkpointing
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        nn.init.xavier_uniform_(self.patch_projection.weight.flatten(1))
        if self.patch_projection.bias is not None:
            nn.init.zeros_(self.patch_projection.bias)
        nn.init.normal_(self.class_embedder.weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.modulation[-1].weight)
        nn.init.zeros_(self.final_layer.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.projection.weight)
        nn.init.zeros_(self.final_layer.projection.bias)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def _validate_inputs(self, noisy: Tensor, timestep: Tensor, labels: Tensor) -> None:
        expected = (noisy.shape[0], *self.config.input_shape)
        if noisy.ndim != 4 or tuple(noisy.shape) != expected:
            raise ValueError(f"noisy input must have shape {expected}")
        if timestep.shape != (noisy.shape[0],) or labels.shape != (noisy.shape[0],):
            raise ValueError("timestep and labels must have shape [B]")

    def _embed(self, noisy: Tensor, timestep: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_inputs(noisy, timestep, labels)
        states = self.patch_projection(noisy).flatten(2).transpose(1, 2)
        conditioning = self.timestep_embedder(timestep) + self.class_embedder(labels)
        return states, conditioning

    def _run_block(self, block: SiTBlock, states: Tensor, conditioning: Tensor) -> Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                block,
                states,
                conditioning,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return block(states, conditioning)

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
        return torch.einsum("bhwpqc->bchpwq", values).reshape(
            batch,
            channels,
            self.config.input_size,
            self.config.input_size,
        )

    def forward_repa(self, noisy: Tensor, timestep: Tensor, labels: Tensor) -> Tensor:
        if self.repa_projection is None:
            raise RuntimeError("REPA is not configured")
        states, conditioning = self._embed(noisy, timestep, labels)
        for block in self.blocks[: self.repa_config.block]:
            states = self._run_block(block, states, conditioning)
        return self.repa_projection(states)

    def forward(
        self,
        noisy: Tensor,
        timestep: Tensor,
        labels: Tensor,
        *,
        return_repa: bool = False,
    ) -> SiTOutput:
        if return_repa and self.repa_projection is None:
            raise ValueError("return_repa requires repa.enabled=true")
        states, conditioning = self._embed(noisy, timestep, labels)
        repa_prediction = None
        for index, block in enumerate(self.blocks, start=1):
            states = self._run_block(block, states, conditioning)
            if return_repa and index == self.repa_config.block:
                assert self.repa_projection is not None
                repa_prediction = self.repa_projection(states)
        prediction = self.final_layer(states, conditioning)
        return SiTOutput(self._unpatchify(prediction), repa_prediction)
