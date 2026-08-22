from __future__ import annotations

from .config import ExperimentConfig


def _attention_flops(tokens: int, width: int, heads: int) -> int:
    head_dim = width // heads
    projections = 2 * tokens * width * (3 * width)
    attention = 4 * heads * tokens * tokens * head_dim
    output = 2 * tokens * width * width
    return projections + attention + output


def _mlp_flops(tokens: int, width: int, ratio: float) -> int:
    hidden = max(1, int(2 * width * ratio / 3))
    return 6 * tokens * width * hidden


def semantic_dit_forward_flops(config: ExperimentConfig) -> int:
    """Approximate FLOPs for one semantic-conditioned DiT forward pass."""
    model = config.model
    spatial = model.grid_size**2
    width = model.hidden_size
    patch_width = model.patch_size**2 * model.in_channels
    total = 4 * spatial * patch_width * width
    total += 2 * 256 * width + 2 * width * width
    total += model.depth * (
        _attention_flops(spatial, width, model.num_heads)
        + _mlp_flops(spatial, width, model.mlp_ratio)
        + 12 * spatial * width * width
    )
    total += 4 * spatial * width * width
    total += 2 * spatial * width * patch_width
    return total


def model_flops_per_step(config: ExperimentConfig) -> int:
    """Approximate global DiT forward/backward FLOPs, excluding frozen DINO."""
    return config.flow.train.global_batch_size * 3 * semantic_dit_forward_flops(config)
