from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..config import GRID_SIZE, PATCH_SIZE, TOKEN_COUNT, ModelConfig
from .rope import apply_rope_cos_sin, precompute_rope_tables


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


class SwiGLU(nn.Module):
    def __init__(self, width: int, ratio: float) -> None:
        super().__init__()
        hidden = max(1, int(2 / 3 * width * ratio))
        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.output = nn.Linear(hidden, width)

    def forward(self, values: Tensor) -> Tensor:
        return self.output(F.silu(self.gate(values)) * self.value(values))


class Attention(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        *,
        token_count: int,
        spatial_tokens: int,
        qk_norm: bool,
        rope: bool,
        sdpa: bool,
    ) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.sdpa = sdpa
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.query_norm = RMSNorm(self.head_width) if qk_norm else nn.Identity()
        self.key_norm = RMSNorm(self.head_width) if qk_norm else nn.Identity()
        if rope:
            cosine, sine = precompute_rope_tables(
                GRID_SIZE, GRID_SIZE, self.head_width, dtype=torch.float32
            )
            suffix = token_count - spatial_tokens
            if suffix:
                cosine = torch.cat((cosine, torch.ones(suffix, self.head_width)))
                sine = torch.cat((sine, torch.zeros(suffix, self.head_width)))
        else:
            cosine = sine = torch.empty(0)
        self.register_buffer("rope_cosine", cosine, persistent=False)
        self.register_buffer("rope_sine", sine, persistent=False)

    def forward(self, values: Tensor) -> Tensor:
        batch, token_count, _ = values.shape
        shape = (batch, token_count, self.heads, self.head_width)
        query = self.query(values).reshape(shape).transpose(1, 2)
        key = self.key(values).reshape(shape).transpose(1, 2)
        value = self.value(values).reshape(shape).transpose(1, 2)
        query, key = self.query_norm(query), self.key_norm(key)
        if self.rope_cosine.numel():
            query, key = apply_rope_cos_sin(
                query,
                key,
                self.rope_cosine.to(query),
                self.rope_sine.to(query),
            )
        if self.sdpa:
            attended = F.scaled_dot_product_attention(query, key, value)
        else:
            weights = (query * self.head_width**-0.5) @ key.transpose(-2, -1)
            attended = weights.softmax(dim=-1) @ value
        return self.output(
            attended.transpose(1, 2).reshape(batch, token_count, self.width)
        )


class DDTEncoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, global_tokens: int) -> None:
        super().__init__()
        path = config.encoder
        self.norm_attention = RMSNorm(path.width)
        self.attention = Attention(
            path.width,
            path.heads,
            token_count=TOKEN_COUNT + global_tokens,
            spatial_tokens=TOKEN_COUNT,
            qk_norm=config.attention.qk_norm,
            rope=config.attention.rope,
            sdpa=config.attention.sdpa,
        )
        self.norm_mlp = RMSNorm(path.width)
        self.mlp = SwiGLU(path.width, config.feedforward.mlp_ratio)

    def forward(self, values: Tensor) -> Tensor:
        values = values + self.attention(self.norm_attention(values))
        return values + self.mlp(self.norm_mlp(values))


class DDTDecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        path = config.semantic_decoder
        self.norm_attention = RMSNorm(path.width)
        self.attention = Attention(
            path.width,
            path.heads,
            token_count=TOKEN_COUNT,
            spatial_tokens=TOKEN_COUNT,
            qk_norm=config.attention.qk_norm,
            rope=config.attention.rope,
            sdpa=config.attention.sdpa,
        )
        self.norm_mlp = RMSNorm(path.width)
        self.mlp = SwiGLU(path.width, config.feedforward.mlp_ratio)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(path.width, 6 * path.width))

    def forward(self, values: Tensor, conditioning: Tensor) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaln(
            conditioning
        ).chunk(6, dim=-1)
        values = values + gate_a * self.attention(
            _modulate(self.norm_attention(values), shift_a, scale_a)
        )
        return values + gate_m * self.mlp(
            _modulate(self.norm_mlp(values), shift_m, scale_m)
        )


class PixelTransformerBlock(nn.Module):
    """An ordinary, non-AdaLN RoPE transformer encoder block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        path = config.pixel_decoder
        self.norm_attention = RMSNorm(path.width)
        self.attention = Attention(
            path.width,
            path.heads,
            token_count=TOKEN_COUNT,
            spatial_tokens=TOKEN_COUNT,
            qk_norm=config.attention.qk_norm,
            rope=config.attention.rope,
            sdpa=config.attention.sdpa,
        )
        self.norm_mlp = RMSNorm(path.width)
        self.mlp = SwiGLU(path.width, config.feedforward.mlp_ratio)

    def forward(self, values: Tensor) -> Tensor:
        values = values + self.attention(self.norm_attention(values))
        return values + self.mlp(self.norm_mlp(values))


class LayerAttentionBlock(nn.Module):
    """A transformer block that mixes feature depth, not spatial positions."""

    def __init__(
        self,
        width: int,
        heads: int,
        mlp_ratio: float,
        *,
        layer_tokens: int,
        qk_norm: bool,
        sdpa: bool,
    ) -> None:
        super().__init__()
        self.norm_attention = RMSNorm(width)
        self.attention = Attention(
            width,
            heads,
            token_count=layer_tokens,
            spatial_tokens=0,
            qk_norm=qk_norm,
            rope=False,
            sdpa=sdpa,
        )
        self.norm_mlp = RMSNorm(width)
        self.mlp = SwiGLU(width, mlp_ratio)

    def forward(self, values: Tensor) -> Tensor:
        values = values + self.attention(self.norm_attention(values))
        return values + self.mlp(self.norm_mlp(values))


class LayerAttentionFusion(nn.Module):
    """Fuse aligned flow states with attention over their network depths."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        fusion = config.pixel_decoder.fusion.layer_attention
        source_widths = (
            [config.encoder.width] * len(config.pixel_decoder.encoder_features)
            + [config.semantic_decoder.width]
            * len(config.pixel_decoder.decoder_features)
        )
        self.source_norms = nn.ModuleList(RMSNorm(width) for width in source_widths)
        self.source_projections = nn.ModuleList(
            nn.Linear(width, fusion.width) for width in source_widths
        )
        self.source_embeddings = nn.Parameter(
            torch.randn(len(source_widths), fusion.width) / fusion.width**0.5
        )
        self.context_token = nn.Parameter(
            torch.randn(fusion.width) / fusion.width**0.5
        )
        layer_tokens = len(source_widths) + 1
        self.blocks = nn.ModuleList(
            LayerAttentionBlock(
                fusion.width,
                fusion.heads,
                fusion.mlp_ratio,
                layer_tokens=layer_tokens,
                qk_norm=config.attention.qk_norm,
                sdpa=config.attention.sdpa,
            )
            for _ in range(fusion.depth)
        )
        self.final_norm = RMSNorm(fusion.width)
        self.spatial_projection = nn.Linear(fusion.width, config.pixel_decoder.width)
        self.context_projection = nn.Linear(fusion.width, config.pixel_decoder.width)
        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(self, features: tuple[Tensor, ...]) -> tuple[Tensor, Tensor]:
        if len(features) != len(self.source_projections):
            raise ValueError("layer fusion received the wrong number of feature tensors")
        projected = [
            projection(norm(values))
            for values, norm, projection in zip(
                features, self.source_norms, self.source_projections, strict=True
            )
        ]
        shape = projected[0].shape[:2]
        if any(values.shape[:2] != shape for values in projected):
            raise ValueError("layer fusion features must share batch and spatial dimensions")
        states = torch.stack(projected, dim=2)
        states = states + self.source_embeddings[None, None].to(states)
        context = self.context_token[None, None, None].expand(
            states.shape[0], states.shape[1], 1, -1
        )
        states = torch.cat((states, context.to(states)), dim=2)
        batch, spatial, layers, width = states.shape
        states = states.reshape(batch * spatial, layers, width)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                states = checkpoint(block, states, use_reentrant=False)
            else:
                states = block(states)
        states = self.final_norm(states).reshape(batch, spatial, layers, width)
        spatial_tokens = states[:, :, :-1].mean(dim=2)
        context_tokens = states[:, :, -1]
        return (
            self.spatial_projection(spatial_tokens),
            self.context_projection(context_tokens),
        )


class DDTFinalLayer(nn.Module):
    def __init__(self, width: int, output_width: int) -> None:
        super().__init__()
        self.norm = RMSNorm(width)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.projection = nn.Linear(width, output_width)

    def forward(self, values: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale = self.adaln(conditioning).chunk(2, dim=-1)
        return self.projection(_modulate(self.norm(values), shift, scale))


class TimestepEmbedder(nn.Module):
    def __init__(self, width: int, token_count: int, frequency_width: int = 256) -> None:
        super().__init__()
        self.register_buffer(
            "frequencies", torch.randn(frequency_width), persistent=True
        )
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
    def __init__(
        self,
        width: int,
        num_classes: int,
        token_count: int,
        dropout_probability: float,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_probability = dropout_probability
        self.embedding = nn.Embedding(num_classes + 1, width)
        self.tokens = nn.Parameter(torch.randn(token_count, width) / width**0.5)

    def forward(self, labels: Tensor, force_drop: Tensor | None = None) -> Tensor:
        if labels.ndim != 1 or torch.any(labels < 0) or torch.any(labels >= self.num_classes):
            raise ValueError("class labels are outside the configured range")
        if force_drop is None and self.training and self.dropout_probability:
            force_drop = torch.rand(labels.shape, device=labels.device) < self.dropout_probability
        if force_drop is not None:
            if force_drop.shape != labels.shape:
                raise ValueError("force_class_drop must have shape [B]")
            labels = torch.where(force_drop.bool(), self.num_classes, labels)
        return self.embedding(labels)[:, None] + self.tokens


class DiPPatchDecoder(nn.Module):
    """Reference DiP local decoder adapted to independent RGB 16x16 patches."""

    def __init__(self, context_width: int) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.SiLU())
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.SiLU())
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.SiLU())
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.SiLU())
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512 + context_width, 512, 1), nn.SiLU()
        )
        self.up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4 = nn.Sequential(nn.Conv2d(1024, 256, 3, padding=1), nn.SiLU())
        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.SiLU())
        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.SiLU())
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.SiLU())
        self.out_conv = nn.Conv2d(64, 3, 1)

    def forward(self, coarse: Tensor, context: Tensor) -> Tensor:
        if coarse.ndim != 4 or coarse.shape[1:] != (3, PATCH_SIZE, PATCH_SIZE):
            raise ValueError("DiP coarse patches must have shape [N,3,16,16]")
        if context.ndim != 2 or context.shape[0] != coarse.shape[0]:
            raise ValueError("DiP context must have shape [N,D]")
        skip1 = self.enc1(coarse)
        skip2 = self.enc2(self.pool1(skip1))
        skip3 = self.enc3(self.pool2(skip2))
        skip4 = self.enc4(self.pool3(skip3))
        bottleneck = self.pool4(skip4)
        conditioned = self.bottleneck(
            torch.cat((bottleneck, context[:, :, None, None].to(bottleneck)), dim=1)
        )
        decoded = self.dec4(torch.cat((self.up4(conditioned), skip4), dim=1))
        decoded = self.dec3(torch.cat((self.up3(decoded), skip3), dim=1))
        decoded = self.dec2(torch.cat((self.up2(decoded), skip2), dim=1))
        decoded = self.dec1(torch.cat((self.up1(decoded), skip1), dim=1))
        return self.out_conv(decoded)


LocalDecoder = DiPPatchDecoder


class PixelDecoderBackbone(nn.Module):
    """Pixel transformer and DiP detailer shared by pretraining and joint DDT."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        width = config.pixel_decoder.width
        self.blocks = nn.ModuleList(
            PixelTransformerBlock(config) for _ in range(config.pixel_decoder.depth)
        )
        self.context_norm = RMSNorm(width)
        self.context_mlp = SwiGLU(width, config.feedforward.mlp_ratio)
        self.coarse_patch_projection = nn.Linear(
            width, 3 * PATCH_SIZE * PATCH_SIZE
        )
        self.patch_detailer = DiPPatchDecoder(width)
        self.gradient_checkpointing = config.gradient_checkpointing

    def _block(self, block: nn.Module, values: Tensor) -> Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, values, use_reentrant=False)
        return block(values)

    def forward(self, tokens: Tensor, context: Tensor) -> Tensor:
        if tokens.ndim != 3 or context.shape != tokens.shape:
            raise ValueError(
                "pixel decoder tokens and context must have equal [B,64,D] shapes"
            )
        if tokens.shape[1] != TOKEN_COUNT:
            raise ValueError("pixel decoder requires 64 spatial tokens")
        batch = tokens.shape[0]
        context = context + self.context_mlp(self.context_norm(context))
        for block in self.blocks:
            tokens = self._block(block, tokens)
        coarse = self.coarse_patch_projection(tokens).reshape(
            batch * TOKEN_COUNT, 3, PATCH_SIZE, PATCH_SIZE
        )
        detailed = self.patch_detailer(
            coarse, context.reshape(batch * TOKEN_COUNT, context.shape[-1])
        )
        patch_tokens = detailed.reshape(batch, TOKEN_COUNT, -1)
        return F.fold(
            patch_tokens.transpose(1, 2),
            output_size=(GRID_SIZE * PATCH_SIZE, GRID_SIZE * PATCH_SIZE),
            kernel_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        )


class SemanticPixelDecoder(nn.Module):
    """Standalone normalized-DINO-to-RGB decoder used for pretraining."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        width = config.pixel_decoder.width
        self.token_projection = nn.Linear(config.token_dim, width)
        self.context_projection = nn.Linear(config.token_dim, width)
        self.backbone = PixelDecoderBackbone(config)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)

    def forward(self, semantic_tokens: Tensor) -> Tensor:
        if semantic_tokens.ndim != 3 or semantic_tokens.shape[1:] != (
            TOKEN_COUNT,
            self.config.token_dim,
        ):
            raise ValueError("semantic pixel decoder input must have shape [B,64,768]")
        return self.backbone(
            self.token_projection(semantic_tokens),
            self.context_projection(semantic_tokens),
        )


@dataclass(frozen=True)
class DDTOutput:
    full_x0: Tensor
    base_x0: Tensor
    rgb: Tensor | None = None

    @property
    def pixels(self) -> Tensor | None:
        return self.rgb

    @property
    def pixel_rgb(self) -> Tensor | None:
        return self.rgb


@dataclass(frozen=True)
class PixelFeatureBundle:
    encoder: tuple[Tensor, ...]
    decoder: tuple[Tensor, ...]
    semantic: Tensor | None = None

    @property
    def all(self) -> tuple[Tensor, ...]:
        return self.encoder + self.decoder

    def detached(self) -> PixelFeatureBundle:
        return PixelFeatureBundle(
            tuple(values.detach() for values in self.encoder),
            tuple(values.detach() for values in self.decoder),
            None if self.semantic is None else self.semantic.detach(),
        )

    def index_select(self, indices: Tensor) -> PixelFeatureBundle:
        if indices.ndim != 1 or indices.dtype != torch.int64:
            raise ValueError("pixel feature indices must be a one-dimensional int64 tensor")
        return PixelFeatureBundle(
            tuple(values.index_select(0, indices) for values in self.encoder),
            tuple(values.index_select(0, indices) for values in self.decoder),
            None
            if self.semantic is None
            else self.semantic.index_select(0, indices),
        )


class DDT(nn.Module):
    """Joint semantic flow with RAEv2 base prediction and DiP RGB decoding."""

    def __setattr__(self, name: str, value: object) -> None:
        if name == "patch_detailer":
            modules = self.__dict__.get("_modules", {})
            decoder = modules.get("pixel_decoder")
            if decoder is not None:
                decoder.patch_detailer = value
                return
        super().__setattr__(name, value)

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self._validate_config(config)
        self.config = config
        encoder_width = config.encoder.width
        semantic_width = config.semantic_decoder.width
        pixel_width = config.pixel_decoder.width
        conditioning = config.conditioning
        global_tokens = conditioning.timestep_tokens + conditioning.class_tokens

        self.source_projection = nn.Linear(config.token_dim, encoder_width)
        self.target_projection = nn.Linear(config.token_dim, semantic_width)
        self.timestep_embedder = TimestepEmbedder(
            encoder_width, conditioning.timestep_tokens
        )
        self.class_embedder = LabelEmbedder(
            encoder_width,
            conditioning.num_classes,
            conditioning.class_tokens,
            conditioning.class_dropout_probability,
        )
        self.encoder_blocks = nn.ModuleList(
            DDTEncoderBlock(config, global_tokens) for _ in range(config.encoder.depth)
        )
        self.encoder_to_decoder = (
            nn.Linear(encoder_width, semantic_width)
            if encoder_width != semantic_width
            else nn.Identity()
        )
        self.semantic_decoder_blocks = nn.ModuleList(
            DDTDecoderBlock(config) for _ in range(config.semantic_decoder.depth)
        )
        self.full_head = DDTFinalLayer(semantic_width, config.token_dim)
        self.base_head = DDTFinalLayer(encoder_width, config.token_dim)

        fusion_width = (
            len(config.pixel_decoder.encoder_features) * encoder_width
            + len(config.pixel_decoder.decoder_features) * semantic_width
        )
        self.pixel_fusion_kind = config.pixel_decoder.fusion.kind
        if self.pixel_fusion_kind == "concatenate":
            self.pixel_flow_token_projection = nn.Linear(fusion_width, pixel_width)
            self.pixel_flow_context_projection = nn.Linear(fusion_width, pixel_width)
        else:
            self.pixel_layer_fusion = LayerAttentionFusion(config)
        joint_width = config.token_dim + pixel_width
        self.pixel_input_projection = nn.Linear(joint_width, pixel_width)
        self.pixel_detail_context_projection = nn.Linear(joint_width, pixel_width)
        self.pixel_decoder = PixelDecoderBackbone(config)
        self.register_buffer(
            "pixel_semantic_scale", torch.ones(config.token_dim), persistent=True
        )
        self.register_buffer(
            "pixel_semantic_bias", torch.zeros(config.token_dim), persistent=True
        )
        self.gradient_checkpointing = config.gradient_checkpointing
        self.initialize_weights()
        self._zero_initialize_flow_slices()

    @property
    def semantic_head(self) -> DDTFinalLayer:
        return self.full_head

    @property
    def base_final_layer(self) -> DDTFinalLayer:
        return self.base_head

    @property
    def pixel_transformer_blocks(self) -> nn.ModuleList:
        return self.pixel_decoder.blocks

    @property
    def pixel_blocks(self) -> nn.ModuleList:
        return self.pixel_decoder.blocks

    @property
    def coarse_patch_projection(self) -> nn.Linear:
        return self.pixel_decoder.coarse_patch_projection

    @property
    def patch_detailer(self) -> DiPPatchDecoder:
        return self.pixel_decoder.patch_detailer

    def set_pixel_detailer(self, detailer: nn.Module) -> None:
        """Replace the DiP detailer, primarily for lightweight tests."""
        self.pixel_decoder.patch_detailer = detailer

    def pixel_parameter_names(self) -> set[str]:
        return {
            name
            for name, _parameter in self.named_parameters()
            if name.startswith("pixel_")
        }

    def pixel_decoder_parameter_names(self) -> set[str]:
        return {
            name
            for name, _parameter in self.named_parameters()
            if name.startswith("pixel_decoder.")
        }

    def pixel_adapter_parameter_names(self) -> set[str]:
        return self.pixel_parameter_names() - self.pixel_decoder_parameter_names()

    def flow_parameter_names(self) -> set[str]:
        pixel = self.pixel_parameter_names()
        return {name for name, _parameter in self.named_parameters() if name not in pixel}

    def set_pixel_decoder_frozen(self, frozen: bool) -> None:
        self.pixel_decoder.requires_grad_(not frozen)

    def pixel_decoder_is_frozen(self) -> bool:
        parameters = tuple(self.pixel_decoder.parameters())
        return bool(parameters) and not any(parameter.requires_grad for parameter in parameters)

    @staticmethod
    def _validate_config(config: ModelConfig) -> None:
        if not 1 <= config.base_prediction_depth <= config.encoder.depth:
            raise ValueError("base_prediction_depth must select an encoder block")
        fusion = config.pixel_decoder.fusion
        if fusion.kind not in {"concatenate", "layer_attention"}:
            raise ValueError("pixel fusion kind must be concatenate or layer_attention")
        layer = fusion.layer_attention
        if (
            layer.width <= 0
            or layer.depth <= 0
            or layer.heads <= 0
            or layer.mlp_ratio <= 0
            or layer.width % layer.heads
            or (layer.width // layer.heads) % 4
        ):
            raise ValueError("layer-attention fusion dimensions are invalid")
        for name, selected, depth in (
            ("encoder_features", config.pixel_decoder.encoder_features, config.encoder.depth),
            (
                "decoder_features",
                config.pixel_decoder.decoder_features,
                config.semantic_decoder.depth,
            ),
        ):
            if not selected or len(selected) != len(set(selected)):
                raise ValueError(f"{name} must contain unique block selections")
            if any(not 1 <= index <= depth for index in selected):
                raise ValueError(f"{name} contains an out-of-range block")

    def initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        nn.init.normal_(self.class_embedder.embedding.weight, std=0.02)
        for layer in (self.timestep_embedder.mlp[0], self.timestep_embedder.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
        for block in self.semantic_decoder_blocks:
            nn.init.zeros_(block.adaln[-1].weight)
            nn.init.zeros_(block.adaln[-1].bias)
        for head in (self.full_head, self.base_head):
            nn.init.zeros_(head.adaln[-1].weight)
            nn.init.zeros_(head.adaln[-1].bias)
            nn.init.zeros_(head.projection.weight)
            nn.init.zeros_(head.projection.bias)

    def _zero_initialize_flow_slices(self) -> None:
        semantic_width = self.config.token_dim
        with torch.no_grad():
            self.pixel_input_projection.weight[:, semantic_width:].zero_()
            self.pixel_detail_context_projection.weight[:, semantic_width:].zero_()

    def initialize_pretrained_pixel_decoder(
        self,
        decoder: SemanticPixelDecoder,
        *,
        flow_mean: Tensor,
        flow_std: Tensor,
        decoder_mean: Tensor,
        decoder_std: Tensor,
    ) -> None:
        """Install a standalone decoder and its semantic normalization contract."""
        decoder_path = decoder.config.pixel_decoder
        active_path = self.config.pixel_decoder
        if (
            decoder_path.width,
            decoder_path.depth,
            decoder_path.heads,
        ) != (
            active_path.width,
            active_path.depth,
            active_path.heads,
        ):
            raise ValueError("pretrained pixel decoder architecture does not match the DDT")
        if decoder.config.feedforward != self.config.feedforward:
            raise ValueError("pretrained pixel feedforward configuration does not match")
        if decoder.config.attention != self.config.attention:
            raise ValueError("pretrained pixel attention configuration does not match")
        expected = (self.config.token_dim,)
        for name, values in (
            ("flow mean", flow_mean),
            ("flow std", flow_std),
            ("decoder mean", decoder_mean),
            ("decoder std", decoder_std),
        ):
            if values.shape != expected or not torch.isfinite(values).all():
                raise ValueError(f"{name} must be a finite width-768 tensor")
        if torch.any(flow_std <= 0) or torch.any(decoder_std <= 0):
            raise ValueError("semantic normalization standard deviations must be positive")
        self.pixel_decoder.load_state_dict(decoder.backbone.state_dict())
        semantic_width = self.config.token_dim
        with torch.no_grad():
            for target, source in (
                (self.pixel_input_projection, decoder.token_projection),
                (self.pixel_detail_context_projection, decoder.context_projection),
            ):
                target.weight.zero_()
                target.weight[:, :semantic_width].copy_(source.weight)
                target.bias.copy_(source.bias)
            self.pixel_semantic_scale.copy_(flow_std / decoder_std)
            self.pixel_semantic_bias.copy_((flow_mean - decoder_mean) / decoder_std)

    def _plain_block(self, block: nn.Module, values: Tensor) -> Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, values, use_reentrant=False)
        return block(values)

    def _decoder_block(
        self, block: DDTDecoderBlock, values: Tensor, conditioning: Tensor
    ) -> Tensor:
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, values, conditioning, use_reentrant=False)
        return block(values, conditioning)

    @staticmethod
    def _autocast(values: Tensor):
        if values.device.type == "cuda" and not torch.is_autocast_enabled():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _decode_pixels(self, features: PixelFeatureBundle) -> Tensor:
        if features.semantic is None:
            raise ValueError("pixel decoding requires detached predicted-clean tokens")
        semantic = features.semantic.detach()
        if semantic.shape[1:] != (TOKEN_COUNT, self.config.token_dim):
            raise ValueError("pixel semantic conditioning must have shape [B,64,768]")
        semantic = (
            semantic * self.pixel_semantic_scale.to(semantic)
            + self.pixel_semantic_bias.to(semantic)
        )
        if self.pixel_fusion_kind == "concatenate":
            concatenated = torch.cat(features.all, dim=-1)
            pooled_tokens = self.pixel_flow_token_projection(concatenated)
            pooled_context = self.pixel_flow_context_projection(concatenated)
        else:
            pooled_tokens, pooled_context = self.pixel_layer_fusion(features.all)
        pixel_tokens = self.pixel_input_projection(
            torch.cat((semantic, pooled_tokens), dim=-1)
        )
        context = self.pixel_detail_context_projection(
            torch.cat((semantic, pooled_context), dim=-1)
        )
        return self.pixel_decoder(pixel_tokens, context)

    def forward_semantic(
        self,
        noisy_tokens: Tensor,
        timestep: Tensor,
        labels: Tensor,
        *,
        capture_pixel_features: bool,
        force_class_drop: Tensor | None = None,
    ) -> tuple[DDTOutput, PixelFeatureBundle | None]:
        """Run semantic flow and optionally retain the selected raw hidden states."""
        if noisy_tokens.ndim != 3 or noisy_tokens.shape[1:] != (
            TOKEN_COUNT,
            self.config.token_dim,
        ):
            raise ValueError("DDT input must have shape [B,64,768]")
        batch = noisy_tokens.shape[0]
        if timestep.shape != (batch,) or labels.shape != (batch,):
            raise ValueError("timestep and labels must have shape [B]")

        with self._autocast(noisy_tokens):
            timestep_base, timestep_tokens = self.timestep_embedder(timestep)
            class_tokens = self.class_embedder(labels, force_class_drop)
            states = torch.cat(
                (self.source_projection(noisy_tokens), timestep_tokens, class_tokens), dim=1
            )
            encoder_features: dict[int, Tensor] = {}
            base_spatial: Tensor | None = None
            selected_encoder = set(self.config.pixel_decoder.encoder_features)
            for depth, block in enumerate(self.encoder_blocks, start=1):
                states = self._plain_block(block, states)
                if depth == self.config.base_prediction_depth:
                    base_spatial = states[:, :TOKEN_COUNT]
                if capture_pixel_features and depth in selected_encoder:
                    encoder_features[depth] = states[:, :TOKEN_COUNT]
            assert base_spatial is not None
            final_spatial = states[:, :TOKEN_COUNT]
            base_conditioning = F.silu(timestep_base + base_spatial)
            base_x0 = self.base_head(base_conditioning, base_conditioning)
            conditioning = self.encoder_to_decoder(F.silu(timestep_base + final_spatial))

            decoded = self.target_projection(noisy_tokens)
            decoder_features: dict[int, Tensor] = {}
            selected_decoder = set(self.config.pixel_decoder.decoder_features)
            for depth, block in enumerate(self.semantic_decoder_blocks, start=1):
                decoded = self._decoder_block(block, decoded, conditioning)
                if capture_pixel_features and depth in selected_decoder:
                    decoder_features[depth] = decoded
            full_x0 = self.full_head(decoded, conditioning)
            output = DDTOutput(full_x0, base_x0)
            if not capture_pixel_features:
                return output, None
            features = PixelFeatureBundle(
                tuple(
                    encoder_features[index]
                    for index in self.config.pixel_decoder.encoder_features
                ),
                tuple(
                    decoder_features[index]
                    for index in self.config.pixel_decoder.decoder_features
                ),
                full_x0.detach(),
            )
            return output, features

    def decode_pixel_features(
        self,
        features: PixelFeatureBundle,
        *,
        detach_flow: bool = False,
    ) -> Tensor:
        if detach_flow:
            features = features.detached()
        with self._autocast(features.all[0]):
            return self._decode_pixels(features)

    def forward(
        self,
        noisy_tokens: Tensor,
        timestep: Tensor,
        labels: Tensor,
        *,
        decode_pixels: bool,
        force_class_drop: Tensor | None = None,
    ) -> DDTOutput:
        semantic, features = self.forward_semantic(
            noisy_tokens,
            timestep,
            labels,
            capture_pixel_features=decode_pixels,
            force_class_drop=force_class_drop,
        )
        if features is None:
            return semantic
        return DDTOutput(
            semantic.full_x0,
            semantic.base_x0,
            self.decode_pixel_features(features),
        )


def build_model(config: ModelConfig) -> DDT:
    return DDT(config)
