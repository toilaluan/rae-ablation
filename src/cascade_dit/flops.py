from __future__ import annotations

from .config import TOKEN_COUNT, TOKEN_DIM, ExperimentConfig


def _attention_flops(tokens: int, width: int) -> int:
    return 8 * tokens * width * width + 4 * tokens * tokens * width


def _swiglu_flops(tokens: int, width: int, ratio: float) -> int:
    intermediate = int((2 / 3) * width * ratio)
    return 6 * tokens * width * intermediate


def lightningdit_forward_flops(config: ExperimentConfig, *, include_pixel_detail: bool = True) -> int:
    model = config.model
    total = 2 * TOKEN_COUNT * TOKEN_DIM * model.hidden_size
    total += model.depth * (
        _attention_flops(TOKEN_COUNT, model.hidden_size)
        + _swiglu_flops(TOKEN_COUNT, model.hidden_size, model.mlp_ratio)
        + 12 * model.hidden_size * model.hidden_size
    )
    total += 2 * TOKEN_COUNT * model.hidden_size * TOKEN_DIM
    detail_hidden = int(TOKEN_DIM * config.detail_heads.token_expansion)
    total += 4 * TOKEN_COUNT * TOKEN_DIM * model.hidden_size
    total += config.detail_heads.token_depth * 4 * TOKEN_COUNT * TOKEN_DIM * detail_hidden
    if include_pixel_detail and config.detail_heads.pixel_enabled:
        # Convolution estimate is deliberately explicit and architecture-specific.
        resolutions = (16, 8, 4, 2)
        inputs = (3, 64, 128, 256)
        outputs = (64, 128, 256, 512)
        convolution = 0
        for side, input_channels, output_channels in zip(
            resolutions, inputs, outputs, strict=True
        ):
            convolution += 2 * side**2 * 9 * input_channels * output_channels
            convolution += 2 * (side // 2) ** 2 * 16 * output_channels**2
        total += TOKEN_COUNT * convolution * 2  # down and comparable up paths
    return total


def model_flops_per_step(config: ExperimentConfig, *, stage: str = "rgb") -> int:
    forward = lightningdit_forward_flops(config, include_pixel_detail=stage == "rgb")
    return config.optimizer.global_batch_size * 3 * forward
