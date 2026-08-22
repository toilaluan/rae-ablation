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
    hidden = max(1, int(width * ratio))
    return 4 * tokens * width * hidden


def dit_forward_flops(
    config: ExperimentConfig,
    route: Literal["semantic", "image_latent"],
) -> int:
    model = config.model
    spatial = 64
    tokens = spatial + model.num_timestep_tokens + model.num_class_tokens
    input_width = (
        SEMANTIC_CHANNELS if route == "semantic" else 4 * IMAGE_LATENT_CHANNELS
    )
    output_width = input_width
    total = 2 * spatial * input_width * model.hidden_size
    total += model.depth * (
        _attention_flops(tokens, model.hidden_size, model.num_heads)
        + _mlp_flops(tokens, model.hidden_size, model.mlp_ratio)
    )
    total += 2 * spatial * model.hidden_size * output_width
    if route == "semantic":
        total += 2 * spatial * model.hidden_size * output_width
    else:
        total += 2 * spatial * SEMANTIC_CHANNELS * model.hidden_size
        total += model.depth * 12 * spatial * model.hidden_size**2
    return total


def model_flops_per_step(config: ExperimentConfig) -> int:
    """Approximate global forward/backward FLOPs, excluding frozen encoders."""
    routes = dit_forward_flops(config, "image_latent")
    if config.model.architecture != "latent":
        routes += dit_forward_flops(config, "semantic")
    return config.train.global_batch_size * 3 * routes
