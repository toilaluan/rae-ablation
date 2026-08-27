from __future__ import annotations

from .config import PATCH_SIZE, TOKEN_COUNT, TOKEN_DIM, ExperimentConfig


def _attention_flops(tokens: int, width: int) -> int:
    return 8 * tokens * width * width + 4 * tokens * tokens * width


def _swiglu_flops(tokens: int, width: int, ratio: float) -> int:
    intermediate = int((2 / 3) * width * ratio)
    return 6 * tokens * width * intermediate


def _transformer_flops(tokens: int, width: int, ratio: float) -> int:
    return _attention_flops(tokens, width) + _swiglu_flops(tokens, width, ratio)


def _conv_flops(side: int, source: int, target: int, kernel: int) -> int:
    return 2 * side * side * source * target * kernel * kernel


def _dip_patch_flops(context_width: int) -> int:
    total = _conv_flops(16, 3, 64, 3)
    total += _conv_flops(8, 64, 128, 3)
    total += _conv_flops(4, 128, 256, 3)
    total += _conv_flops(2, 256, 512, 3)
    total += _conv_flops(1, 512 + context_width, 512, 1)
    total += _conv_flops(2, 1024, 256, 3)
    total += _conv_flops(4, 512, 128, 3)
    total += _conv_flops(8, 256, 64, 3)
    total += _conv_flops(16, 128, 64, 3)
    total += _conv_flops(16, 64, 3, 1)
    return total


def _pixel_backbone_flops(config: ExperimentConfig) -> int:
    model = config.model
    pixel = model.pixel_decoder
    ratio = model.feedforward.mlp_ratio
    total = _swiglu_flops(TOKEN_COUNT, pixel.width, ratio)
    total += pixel.depth * _transformer_flops(TOKEN_COUNT, pixel.width, ratio)
    total += 2 * TOKEN_COUNT * pixel.width * (3 * PATCH_SIZE * PATCH_SIZE)
    total += TOKEN_COUNT * _dip_patch_flops(pixel.width)
    return total


def semantic_pixel_decoder_forward_flops(config: ExperimentConfig) -> int:
    pixel = config.model.pixel_decoder
    projections = 4 * TOKEN_COUNT * TOKEN_DIM * pixel.width
    return projections + _pixel_backbone_flops(config)


def pixel_pretraining_flops_per_step(config: ExperimentConfig) -> int:
    return (
        config.pixel_pretraining.optimizer.global_batch_size
        * 3
        * semantic_pixel_decoder_forward_flops(config)
    )


def ddt_forward_flops(config: ExperimentConfig, *, decode_pixels: bool = True) -> int:
    model = config.model
    encoder = model.encoder
    semantic = model.semantic_decoder
    pixel = model.pixel_decoder
    global_tokens = model.conditioning.timestep_tokens + model.conditioning.class_tokens
    encoder_tokens = TOKEN_COUNT + global_tokens
    ratio = model.feedforward.mlp_ratio
    total = 2 * TOKEN_COUNT * TOKEN_DIM * encoder.width
    total += encoder.depth * _transformer_flops(encoder_tokens, encoder.width, ratio)
    total += 4 * TOKEN_COUNT * encoder.width * encoder.width
    total += 2 * TOKEN_COUNT * encoder.width * TOKEN_DIM
    if encoder.width != semantic.width:
        total += 2 * TOKEN_COUNT * encoder.width * semantic.width
    total += 2 * TOKEN_COUNT * TOKEN_DIM * semantic.width
    total += semantic.depth * (
        _transformer_flops(TOKEN_COUNT, semantic.width, ratio)
        + 12 * TOKEN_COUNT * semantic.width * semantic.width
    )
    total += 4 * TOKEN_COUNT * semantic.width * semantic.width
    total += 2 * TOKEN_COUNT * semantic.width * TOKEN_DIM
    if not decode_pixels:
        return total
    fusion = (
        len(pixel.encoder_features) * encoder.width
        + len(pixel.decoder_features) * semantic.width
    )
    if pixel.fusion.kind == "concatenate":
        total += 4 * TOKEN_COUNT * fusion * pixel.width
    else:
        layer = pixel.fusion.layer_attention
        layer_tokens = (
            len(pixel.encoder_features) + len(pixel.decoder_features) + 1
        )
        total += 2 * TOKEN_COUNT * fusion * layer.width
        total += TOKEN_COUNT * layer.depth * _transformer_flops(
            layer_tokens, layer.width, layer.mlp_ratio
        )
        total += 4 * TOKEN_COUNT * layer.width * pixel.width
    total += 4 * TOKEN_COUNT * (TOKEN_DIM + pixel.width) * pixel.width
    total += _pixel_backbone_flops(config)
    return total


def model_flops_per_step(
    config: ExperimentConfig,
    *,
    decoder_active: bool = True,
    decoder_updates: int = 1,
    decoder_sample_fraction: float = 1.0,
    decoder_backbone_trainable: bool = True,
) -> int:
    if decoder_updates < 0 or (not decoder_active and decoder_updates):
        raise ValueError("decoder update count is inconsistent with decoder activity")
    if not 0 <= decoder_sample_fraction <= 1:
        raise ValueError("decoder sample fraction must be in [0,1]")
    semantic = ddt_forward_flops(config, decode_pixels=False)
    decoder = ddt_forward_flops(config, decode_pixels=True) - semantic
    updates = decoder_updates if decoder_active else 0
    backbone = _pixel_backbone_flops(config)
    if decoder_backbone_trainable:
        decoder_multiplier = 3 * decoder
    else:
        integration = decoder - backbone
        decoder_multiplier = 3 * integration + 2 * backbone
    decoder_cost = int(
        updates * decoder_sample_fraction * decoder_multiplier
    )
    return config.optimizer.global_batch_size * (3 * semantic + decoder_cost)
