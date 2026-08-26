from __future__ import annotations

from typing import Literal

from .config import TOKEN_COUNT, TOKEN_DIM, ExperimentConfig


def _attention_flops(tokens: int, width: int) -> int:
    return 8 * tokens * width * width + 4 * tokens * tokens * width


def _swiglu_flops(tokens: int, width: int, ratio: float) -> int:
    intermediate = int((2 / 3) * width * ratio)
    return 6 * tokens * width * intermediate


def _decoder_block_flops(config: ExperimentConfig, blocks: int) -> int:
    decoder = config.model.decoder
    return blocks * (
        _attention_flops(TOKEN_COUNT, decoder.width)
        + _swiglu_flops(
            TOKEN_COUNT, decoder.width, config.model.feedforward.mlp_ratio
        )
        + 12 * TOKEN_COUNT * decoder.width * decoder.width
    )


def _head_flops(config: ExperimentConfig) -> int:
    width = config.model.decoder.width
    return 4 * TOKEN_COUNT * width * width + 2 * TOKEN_COUNT * width * TOKEN_DIM


def ddt_forward_flops(
    config: ExperimentConfig,
    prediction_mode: Literal["semantic", "mixed", "pixel"],
) -> int:
    model = config.model
    encoder = model.encoder
    decoder = model.decoder
    global_tokens = (
        model.conditioning.timestep_tokens + model.conditioning.class_tokens + 1
    )
    encoder_tokens = TOKEN_COUNT + global_tokens
    total = 2 * TOKEN_COUNT * TOKEN_DIM * encoder.width
    total += encoder.depth * (
        _attention_flops(encoder_tokens, encoder.width)
        + _swiglu_flops(
            encoder_tokens, encoder.width, model.feedforward.mlp_ratio
        )
    )
    if encoder.width != decoder.width:
        total += 2 * TOKEN_COUNT * encoder.width * decoder.width
    total += 2 * TOKEN_COUNT * TOKEN_DIM * decoder.width
    total += _decoder_block_flops(config, decoder.semantic_depth)
    if prediction_mode != "semantic":
        total += _decoder_block_flops(
            config, decoder.depth - decoder.semantic_depth
        )
    total += _head_flops(config)
    if prediction_mode == "mixed":
        total += _head_flops(config)
    return total


def model_flops_per_step(
    config: ExperimentConfig, phase: Literal["semantic", "pixel"]
) -> int:
    mode = "mixed" if phase == "semantic" else "pixel"
    return config.optimizer.global_batch_size * 3 * ddt_forward_flops(config, mode)
