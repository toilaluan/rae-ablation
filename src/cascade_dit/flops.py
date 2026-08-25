from __future__ import annotations

from .config import TOKEN_COUNT, TOKEN_DIM, ExperimentConfig


def _attention_flops(tokens: int, width: int) -> int:
    return 8 * tokens * width * width + 4 * tokens * tokens * width


def _swiglu_flops(tokens: int, width: int, ratio: float) -> int:
    intermediate = int((2 / 3) * width * ratio)
    return 6 * tokens * width * intermediate


def ddt_forward_flops(config: ExperimentConfig) -> int:
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
    total += decoder.depth * (
        _attention_flops(TOKEN_COUNT, decoder.width)
        + _swiglu_flops(TOKEN_COUNT, decoder.width, model.feedforward.mlp_ratio)
        + 12 * TOKEN_COUNT * decoder.width * decoder.width
    )
    total += 2 * TOKEN_COUNT * decoder.width * TOKEN_DIM
    return total


def model_flops_per_step(config: ExperimentConfig) -> int:
    return config.optimizer.global_batch_size * 3 * ddt_forward_flops(config)
