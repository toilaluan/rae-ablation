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
    tokens = (
        spatial
        + model.num_timestep_tokens
        + model.num_ratio_tokens
        + model.num_class_tokens
    )
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
    enc_tokens = (
        spatial
        + model.num_timestep_tokens
        + model.num_ratio_tokens
        + model.num_class_tokens
    )
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
    """Approximate global DDT FLOPs for one forward/backward training step."""
    batch = config.train.global_batch_size
    winner_forward = ddt_full_forward_flops(config, include_base_head=True)
    return batch * 3 * winner_forward


def pixel_encoder_flops_per_step(config: ExperimentConfig) -> int:
    """Approximate global FLOPs for the frozen online pixel-encoder forward."""
    widths = config.pixel_codec.widths
    size = 256

    def convolution(spatial: int, in_channels: int, out_channels: int, kernel: int) -> int:
        return 2 * spatial**2 * in_channels * out_channels * kernel**2

    total = convolution(size, 3, widths[0], 3)
    for index in range(len(widths) - 1):
        total += config.pixel_codec.residual_blocks_per_scale * 2 * convolution(
            size, widths[index], widths[index], 3
        )
        size //= 2
        total += convolution(size, widths[index], widths[index + 1], 3)
    total += config.pixel_codec.residual_blocks_per_scale * 2 * convolution(
        size, widths[-1], widths[-1], 3
    )
    total += convolution(size, widths[-1], config.pixel_codec.latent_channels, 1)
    return config.train.global_batch_size * total


def total_training_flops_per_step(config: ExperimentConfig) -> int:
    return model_flops_per_step(config) + pixel_encoder_flops_per_step(config)
