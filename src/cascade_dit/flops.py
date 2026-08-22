from __future__ import annotations

from typing import Literal

from .config import IMAGE_LATENT_CHANNELS, SEMANTIC_CHANNELS, ExperimentConfig


def _attention_flops(tokens: int, width: int, heads: int) -> int:
    head_width = width // heads
    projections = 2 * tokens * width * (3 * width)
    attention = 4 * heads * tokens * tokens * head_width
    output = 2 * tokens * width * width
    return projections + attention + output


def _mlp_flops(tokens: int, width: int, ratio: float) -> int:
    hidden = max(1, int(2 / 3 * width * ratio))
    return 6 * tokens * width * hidden


def ddt_forward_flops(
    config: ExperimentConfig,
    route: Literal["semantic", "image_latent"],
) -> int:
    model = config.model
    spatial = 64
    encoder_width, decoder_width = model.hidden_size
    encoder_depth, decoder_depth = model.depth
    encoder_heads, decoder_heads = model.num_heads
    encoder_tokens = spatial + model.num_timestep_tokens + model.num_class_tokens
    if route == "semantic":
        input_width = SEMANTIC_CHANNELS
        output_width = SEMANTIC_CHANNELS
    else:
        input_width = model.image_latent_patch_size**2 * IMAGE_LATENT_CHANNELS
        if model.architecture != "latent":
            input_width += SEMANTIC_CHANNELS
        output_width = model.image_latent_patch_size**2 * IMAGE_LATENT_CHANNELS
    total = 2 * spatial * input_width * (encoder_width + decoder_width)
    total += encoder_depth * (
        _attention_flops(encoder_tokens, encoder_width, encoder_heads)
        + _mlp_flops(encoder_tokens, encoder_width, model.mlp_ratio)
    )
    total += 2 * spatial * encoder_width * decoder_width
    total += decoder_depth * (
        _attention_flops(spatial, decoder_width, decoder_heads)
        + _mlp_flops(spatial, decoder_width, model.mlp_ratio)
        + 12 * spatial * decoder_width**2
    )
    total += 4 * spatial * decoder_width**2
    total += 2 * spatial * decoder_width * output_width
    if route == "semantic":
        total += 4 * spatial * encoder_width**2
        total += 2 * spatial * encoder_width * SEMANTIC_CHANNELS
    return total


def model_flops_per_step(config: ExperimentConfig) -> int:
    """Approximate global forward/backward FLOPs, excluding frozen encoders."""
    routes = ddt_forward_flops(config, "image_latent")
    if config.model.architecture != "latent":
        routes += ddt_forward_flops(config, "semantic")
    return config.train.global_batch_size * 3 * routes
