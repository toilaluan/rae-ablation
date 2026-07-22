from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .rope import apply_2d_rope


class SiTOutput(NamedTuple):
    image: Tensor
    cls: Tensor | None = None
    repa: Tensor | None = None


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
    dropped = torch.rand(labels.shape, device=labels.device, generator=generator) < probability
    return torch.where(dropped, torch.full_like(labels, num_classes), labels)


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        if frequency_embedding_size % 2:
            raise ValueError("frequency embedding size must be even")
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timestep: Tensor) -> Tensor:
        if timestep.ndim != 1:
            raise ValueError("timestep must have shape [B]")
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / half
        )
        angles = timestep.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.cos(), angles.sin()), dim=1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor, height: int, width: int, patch_start: int) -> Tensor:
        batch, tokens, _hidden = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        patch_q, patch_k = apply_2d_rope(
            q[:, :, patch_start:], k[:, :, patch_start:], height, width
        )
        if patch_start:
            q = torch.cat((q[:, :, :patch_start], patch_q), dim=2)
            k = torch.cat((k[:, :, :patch_start], patch_k), dim=2)
        else:
            q, k = patch_q, patch_k
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, tokens, self.hidden_size)
        return self.projection(attended)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        intermediate = max(1, int(2 * hidden_size * mlp_ratio / 3))
        self.gate = nn.Linear(hidden_size, intermediate)
        self.value = nn.Linear(hidden_size, intermediate)
        self.projection = nn.Linear(intermediate, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(F.silu(self.gate(x)) * self.value(x))


class SiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attention = Attention(hidden_size, num_heads)
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.feed_forward = SwiGLU(hidden_size, mlp_ratio)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(
        self,
        x: Tensor,
        conditioning: Tensor,
        height: int,
        width: int,
        patch_start: int,
    ) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(
            conditioning
        ).chunk(6, dim=-1)
        x = x + gate_a.unsqueeze(1) * self.attention(
            _modulate(self.norm1(x), shift_a, scale_a), height, width, patch_start
        )
        return x + gate_m.unsqueeze(1) * self.feed_forward(
            _modulate(self.norm2(x), shift_m, scale_m)
        )


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
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
    """Pixel-space, class-conditional SiT with optional REG token entanglement."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.image_size < 1 or config.patch_size < 1:
            raise ValueError("image size and patch size must be positive")
        if config.image_size % config.patch_size:
            raise ValueError("image size must be divisible by patch size")
        if config.hidden_size < 1 or config.num_heads < 1 or config.depth < 1:
            raise ValueError("SiT dimensions must be positive")
        if config.hidden_size % config.num_heads:
            raise ValueError("hidden size must be divisible by number of heads")
        if (config.hidden_size // config.num_heads) % 4:
            raise ValueError("attention head dimension must be divisible by four")
        if config.reg.enabled and not 1 <= config.reg.repa_block <= config.depth:
            raise ValueError("REG REPA block must be within model depth")
        self.config = config
        self.grid_size = config.image_size // config.patch_size
        self.patch_projection = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.timestep_embedding = TimestepEmbedding(config.hidden_size)
        self.class_embedding = nn.Embedding(config.num_classes + 1, config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                SiTBlock(
                    config.hidden_size,
                    config.num_heads,
                    config.mlp_ratio,
                )
                for _ in range(config.depth)
            ]
        )
        patch_values = config.patch_size * config.patch_size * config.in_channels
        self.final_layer = FinalLayer(config.hidden_size, patch_values)
        if config.reg.enabled:
            self.cls_input: nn.Linear | None = nn.Linear(
                config.reg.feature_dim, config.hidden_size
            )
            self.cls_final_layer: FinalLayer | None = FinalLayer(
                config.hidden_size, config.reg.feature_dim
            )
            self.repa_projection: REPAProjection | None = REPAProjection(
                config.hidden_size,
                config.reg.repa_hidden_size,
                config.reg.feature_dim,
            )
        else:
            self.cls_input = None
            self.cls_final_layer = None
            self.repa_projection = None
        self.gradient_checkpointing = config.gradient_checkpointing
        self._initialize_weights()

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size

    def _initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        nn.init.xavier_uniform_(self.patch_projection.weight.flatten(1))
        nn.init.zeros_(self.patch_projection.bias)
        nn.init.normal_(self.class_embedding.weight, std=0.02)
        for layer in (self.timestep_embedding.mlp[0], self.timestep_embedding.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        final_layers = [self.final_layer]
        if self.cls_final_layer is not None:
            final_layers.append(self.cls_final_layer)
        for layer in final_layers:
            nn.init.zeros_(layer.modulation[-1].weight)
            nn.init.zeros_(layer.modulation[-1].bias)
            nn.init.zeros_(layer.projection.weight)
            nn.init.zeros_(layer.projection.bias)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def _checkpointing(self) -> bool:
        return self.gradient_checkpointing and self.training and torch.is_grad_enabled()

    def _unpatchify(self, tokens: Tensor) -> Tensor:
        batch = tokens.shape[0]
        patch = self.config.patch_size
        channels = self.config.in_channels
        values = tokens.reshape(
            batch, self.grid_size, self.grid_size, patch, patch, channels
        )
        return torch.einsum("bhwpqc->bchpwq", values).reshape(
            batch, channels, self.config.image_size, self.config.image_size
        )

    def forward(
        self,
        noisy_image: Tensor,
        timestep: Tensor,
        labels: Tensor,
        noisy_cls: Tensor | None = None,
        return_repa: bool = False,
    ) -> SiTOutput:
        expected = (
            noisy_image.shape[0],
            self.config.in_channels,
            self.config.image_size,
            self.config.image_size,
        )
        if noisy_image.ndim != 4 or tuple(noisy_image.shape) != expected:
            raise ValueError(
                "noisy_image must have shape "
                f"[B, {self.config.in_channels}, {self.config.image_size}, {self.config.image_size}]"
            )
        batch = noisy_image.shape[0]
        if timestep.shape != (batch,) or labels.shape != (batch,):
            raise ValueError("timestep and labels must both have shape [B]")
        if torch.any(labels < 0) or torch.any(labels > self.config.num_classes):
            raise ValueError("labels contain an invalid class index")
        conditioning = self.timestep_embedding(timestep) + self.class_embedding(labels)
        sequence = self.patch_projection(noisy_image).flatten(2).transpose(1, 2)
        patch_start = 0
        if self.config.reg.enabled:
            if noisy_cls is None or noisy_cls.shape != (batch, self.config.reg.feature_dim):
                raise ValueError(
                    f"REG requires noisy_cls with shape [B, {self.config.reg.feature_dim}]"
                )
            assert self.cls_input is not None
            sequence = torch.cat((self.cls_input(noisy_cls).unsqueeze(1), sequence), dim=1)
            patch_start = 1
        elif noisy_cls is not None:
            raise ValueError("noisy_cls is only valid when model.reg.enabled is true")

        repa: Tensor | None = None
        for index, block in enumerate(self.blocks, start=1):
            if self._checkpointing():
                sequence = checkpoint(
                    block,
                    sequence,
                    conditioning,
                    self.grid_size,
                    self.grid_size,
                    patch_start,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                sequence = block(
                    sequence,
                    conditioning,
                    self.grid_size,
                    self.grid_size,
                    patch_start,
                )
            if (
                return_repa
                and self.repa_projection is not None
                and index == self.config.reg.repa_block
            ):
                repa = self.repa_projection(sequence)

        patch_tokens = sequence[:, patch_start:]
        image = self._unpatchify(self.final_layer(patch_tokens, conditioning))
        cls: Tensor | None = None
        if patch_start:
            assert self.cls_final_layer is not None
            cls = self.cls_final_layer(sequence[:, :1], conditioning).squeeze(1)
        return SiTOutput(image=image, cls=cls, repa=repa)
