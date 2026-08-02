from __future__ import annotations

from .config import ExperimentConfig


def _attention_flops(tokens: int, width: int, heads: int) -> int:
    head_dim = width // heads
    return 2 * tokens * width * (3 * width) + 4 * heads * tokens * tokens * head_dim + 2 * tokens * width * width


def _mlp_flops(tokens: int, width: int, ratio: float) -> int:
    hidden = max(1, int(2 * width * ratio / 3))
    return 4 * tokens * width * hidden + 2 * tokens * hidden * width


def ddt_base_forward_flops(config: ExperimentConfig) -> int:
    model = config.model
    spatial = model.grid_size**2
    tokens = spatial + model.num_timestep_tokens + model.num_class_tokens
    width, _ = model.hidden_size
    heads, _ = model.num_heads
    patch_values = model.patch_size[0] ** 2 * model.in_channels
    projection = 2 * spatial * patch_values * width
    blocks = model.base_model_depth * (
        _attention_flops(tokens, width, heads) + _mlp_flops(tokens, width, model.mlp_ratio)
    )
    base_head = 2 * spatial * width * patch_values + 4 * spatial * width * width
    return projection + blocks + base_head


def ddt_full_forward_flops(config: ExperimentConfig, *, include_base_head: bool = False) -> int:
    model = config.model
    spatial = model.grid_size**2
    enc_tokens = spatial + model.num_timestep_tokens + model.num_class_tokens
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
    if include_base_head:
        total += 2 * spatial * enc_width * enc_patch + 4 * spatial * enc_width * enc_width
    return total


def model_flops_per_step(config: ExperimentConfig) -> int:
    """Approximate global DDT FLOPs for scoring plus winner-only forward/backward."""
    batch = config.train.global_batch_size
    winner_forward = ddt_full_forward_flops(config, include_base_head=True)
    total = batch * 3 * winner_forward
    explore = config.exploration
    if explore.k_early > 1:
        total += batch * explore.k_total * ddt_base_forward_flops(config)
    if explore.k_full > 1:
        total += batch * explore.k_full * ddt_full_forward_flops(config)
    return total
