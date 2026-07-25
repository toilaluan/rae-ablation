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
    reg: Tensor | None = None
    repa: Tensor | None = None


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


def _gate(x: Tensor, gate: Tensor) -> Tensor:
    return x * (gate.unsqueeze(1) if gate.ndim == 2 else gate)


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
        qkv = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
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
        image_conditioning: Tensor,
        height: int,
        width: int,
        patch_start: int,
        reg_conditioning: Tensor | None = None,
    ) -> Tensor:
        image_parameters = self.modulation(image_conditioning).chunk(6, dim=-1)
        if patch_start:
            if reg_conditioning is None:
                raise ValueError("REG prefix requires REG block conditioning")
            reg_parameters = self.modulation(reg_conditioning).chunk(6, dim=-1)
            parameters = tuple(
                torch.cat(
                    (
                        reg_value[:, None].expand(-1, patch_start, -1),
                        image_value[:, None].expand(
                            -1, x.shape[1] - patch_start, -1
                        ),
                    ),
                    dim=1,
                )
                for reg_value, image_value in zip(
                    reg_parameters, image_parameters, strict=True
                )
            )
        else:
            parameters = image_parameters
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = parameters
        attended = self.attention(
            _modulate(self.norm1(x), shift_a, scale_a),
            height,
            width,
            patch_start,
        )
        x = x + _gate(attended, gate_a)
        fed_forward = self.feed_forward(
            _modulate(self.norm2(x), shift_m, scale_m)
        )
        return x + _gate(fed_forward, gate_m)


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


class WideREGResidualLayer(nn.Module):
    """Conditioned, attention-free adaLN-Zero SwiGLU residual layer."""

    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.feed_forward = SwiGLU(hidden_size, mlp_ratio)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        scale, gate = self.modulation(conditioning).chunk(2, dim=-1)
        normalized = self.norm(x) * (1 + scale.unsqueeze(1))
        return x + gate.unsqueeze(1) * self.feed_forward(normalized)


class WideREGOutput(NamedTuple):
    x0: Tensor


class WideREGPredictor(nn.Module):
    """Predict clean DINO REG tokens from an intermediate hidden REG prefix."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        depth: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.token_projection = nn.Linear(input_size, hidden_size)
        self.conditioning_projection = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList(
            [WideREGResidualLayer(hidden_size, mlp_ratio) for _ in range(depth)]
        )
        self.output = FinalLayer(hidden_size, output_size)

    def forward(self, x: Tensor, conditioning: Tensor) -> WideREGOutput:
        x = self.token_projection(x)
        conditioning = self.conditioning_projection(conditioning)
        for layer in self.layers:
            x = layer(x, conditioning)
        return WideREGOutput(self.output(x, conditioning))


class SiT(nn.Module):
    """Pixel-space SiT with independent REG denoising and REPA alignment."""

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
        features = config.features
        reg = features.reg
        repa = features.repa
        if reg.enabled and reg.token_mode not in {"cls", "cls_spatial"}:
            raise ValueError("unknown REG token mode")
        if reg.enabled and not 1 <= reg.pool_size <= 16:
            raise ValueError("REG pool size must be within the native 16x16 feature grid")
        if reg.enabled and reg.head.kind not in {"standard", "wide_mlp"}:
            raise ValueError("unknown REG head kind")
        if reg.enabled and reg.early_prediction.block is not None and not (
            1 <= reg.early_prediction.block < config.depth
        ):
            raise ValueError("early REG prediction block must be before the final block")
        if repa.enabled and not 1 <= repa.block <= config.depth:
            raise ValueError("REPA block must be within model depth")
        if repa.enabled and repa.token_scope == "image_and_reg" and not reg.enabled:
            raise ValueError("REPA image_and_reg scope requires REG")
        self.config = config
        self._reg_prediction_block = reg.early_prediction.block
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
        feature_dim = features.encoder.feature_dim
        if reg.enabled:
            self.reg_input: nn.Linear | None = nn.Linear(feature_dim, config.hidden_size)
            if reg.token_mode == "cls_spatial":
                self.reg_position_embedding = nn.Parameter(
                    torch.empty(1, reg.pool_size**2, config.hidden_size)
                )
            else:
                self.register_parameter("reg_position_embedding", None)
            if reg.head.kind == "standard":
                self.reg_head: nn.Module | None = FinalLayer(
                    config.hidden_size, feature_dim
                )
            else:
                self.reg_head = WideREGPredictor(
                    config.hidden_size,
                    reg.head.hidden_size,
                    feature_dim,
                    reg.head.depth,
                    reg.head.mlp_ratio,
                )
            self.reg_reinjection: nn.Sequential | None = (
                nn.Sequential(
                    nn.Linear(
                        feature_dim,
                        reg.early_prediction.projection_hidden_size,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        reg.early_prediction.projection_hidden_size,
                        config.hidden_size,
                    ),
                )
                if self._reg_prediction_block is not None
                else None
            )
        else:
            self.reg_input = None
            self.register_parameter("reg_position_embedding", None)
            self.reg_head = None
            self.reg_reinjection = None
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
        if self.reg_position_embedding is not None:
            nn.init.normal_(self.reg_position_embedding, std=0.02)
        for layer in (self.timestep_embedding.mlp[0], self.timestep_embedding.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        if isinstance(self.reg_head, WideREGPredictor):
            for layer in self.reg_head.layers:
                nn.init.zeros_(layer.modulation[-1].weight)
                nn.init.zeros_(layer.modulation[-1].bias)
        final_layers = [self.final_layer]
        if isinstance(self.reg_head, FinalLayer):
            final_layers.append(self.reg_head)
        if isinstance(self.reg_head, WideREGPredictor):
            final_layers.append(self.reg_head.output)
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
        values = tokens.reshape(batch, self.grid_size, self.grid_size, patch, patch, channels)
        return torch.einsum("bhwpqc->bchpwq", values).reshape(
            batch, channels, self.config.image_size, self.config.image_size
        )

    def _add_reg_positions(self, tokens: Tensor) -> Tensor:
        if self.reg_position_embedding is None:
            return tokens
        return torch.cat((tokens[:, :1], tokens[:, 1:] + self.reg_position_embedding), dim=1)

    def forward(
        self,
        noisy_image: Tensor,
        timestep: Tensor,
        labels: Tensor,
        noisy_reg: Tensor | None = None,
        reg_timestep: Tensor | None = None,
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
        class_conditioning = self.class_embedding(labels)
        image_conditioning = self.timestep_embedding(timestep) + class_conditioning
        sequence = self.patch_projection(noisy_image).flatten(2).transpose(1, 2)
        patch_start = 0
        reg_config = self.config.features.reg
        reg_conditioning: Tensor | None = None
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
                raise ValueError("REG requires reg_timestep with shape [B]")
            reg_conditioning = (
                self.timestep_embedding(reg_timestep) + class_conditioning
            )
            assert self.reg_input is not None
            reg_tokens = self._add_reg_positions(self.reg_input(noisy_reg))
            sequence = torch.cat((reg_tokens, sequence), dim=1)
            patch_start = self.num_reg_tokens
        elif noisy_reg is not None or reg_timestep is not None:
            raise ValueError(
                "noisy_reg and reg_timestep are only valid when REG is enabled"
            )

        reg: Tensor | None = None
        repa: Tensor | None = None
        for index, block in enumerate(self.blocks, start=1):
            if self._checkpointing():
                sequence = checkpoint(
                    block,
                    sequence,
                    image_conditioning,
                    self.grid_size,
                    self.grid_size,
                    patch_start,
                    reg_conditioning,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                sequence = block(
                    sequence,
                    image_conditioning,
                    self.grid_size,
                    self.grid_size,
                    patch_start,
                    reg_conditioning,
                )
            if (
                return_repa
                and self.repa_projection is not None
                and index == self.config.features.repa.block
            ):
                repa_tokens = (
                    sequence
                    if self.config.features.repa.token_scope == "image_and_reg"
                    else sequence[:, patch_start:]
                )
                repa = self.repa_projection(repa_tokens)
            if self.reg_head is not None and index == self._reg_prediction_block:
                assert reg_conditioning is not None
                if self._checkpointing():
                    predicted = checkpoint(
                        self.reg_head,
                        sequence[:, :patch_start],
                        reg_conditioning,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    predicted = self.reg_head(
                        sequence[:, :patch_start],
                        reg_conditioning,
                    )
                reg = predicted.x0 if isinstance(predicted, WideREGOutput) else predicted
                assert self.reg_reinjection is not None
                clean_reg = self._add_reg_positions(self.reg_reinjection(reg))
                sequence = torch.cat((clean_reg, sequence[:, patch_start:]), dim=1)

        patch_tokens = sequence[:, patch_start:]
        image = self._unpatchify(self.final_layer(patch_tokens, image_conditioning))
        if patch_start and reg is None:
            assert self.reg_head is not None and reg_conditioning is not None
            predicted = self.reg_head(
                sequence[:, :patch_start],
                reg_conditioning,
            )
            reg = predicted.x0 if isinstance(predicted, WideREGOutput) else predicted
        return SiTOutput(image=image, reg=reg, repa=repa)
