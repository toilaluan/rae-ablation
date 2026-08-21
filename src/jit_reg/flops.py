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


def ddt_forward_flops(
    config: ExperimentConfig, *, include_repa: bool | None = None
) -> int:
    """Approximate FLOPs for one v16 diffusion forward pass."""
    model = config.model
    spatial = model.grid_size**2
    condition = model.num_timestep_tokens + model.num_stage_tokens + model.num_class_tokens
    enc_tokens = spatial + condition
    enc_width, dec_width = model.hidden_size
    enc_heads, dec_heads = model.num_heads
    enc_patch = model.patch_size[0] ** 2 * model.in_channels
    dec_patch = model.patch_size[1] ** 2 * model.in_channels
    total = 2 * spatial * enc_patch * enc_width
    total += model.depth[0] * (
        _attention_flops(enc_tokens, enc_width, enc_heads)
        + _mlp_flops(enc_tokens, enc_width, model.mlp_ratio)
    )
    total += 2 * spatial * enc_width * dec_width
    total += 2 * spatial * dec_patch * dec_width
    total += model.depth[1] * (
        _attention_flops(spatial, dec_width, dec_heads)
        + _mlp_flops(spatial, dec_width, model.mlp_ratio)
        + 12 * spatial * dec_width * dec_width
    )
    total += 2 * spatial * dec_width * dec_patch + 4 * spatial * dec_width * dec_width
    total += 2 * spatial * enc_width * enc_patch + 4 * spatial * enc_width * enc_width
    if include_repa is None:
        include_repa = config.repa.enabled
    if include_repa:
        repa_width = config.repa.projector_width
        total += 2 * spatial * (
            enc_width * repa_width
            + repa_width * repa_width
            + repa_width * model.in_channels
        )
    return total


def model_flops_per_step(config: ExperimentConfig) -> int:
    """Approximate global forward/backward DDT FLOPs, excluding frozen DINO."""
    return config.train.global_batch_size * 3 * ddt_forward_flops(config)
