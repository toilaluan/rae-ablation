from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .reg import negative_cosine_repa_loss


class JiTLoss(NamedTuple):
    total: Tensor
    image_velocity: Tensor
    reg_velocity: Tensor
    repa: Tensor
    reg_x0_mse: Tensor
    reg_x0_cosine: Tensor


def sample_timesteps(
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str,
    *,
    loc: float = 0.0,
    scale: float = 1.0,
) -> Tensor:
    normal = torch.randn(batch_size, generator=generator, device=device, dtype=torch.float32)
    return (loc + scale * normal).sigmoid()


def denoising_step_bounds(
    interval: Sequence[float], steps: int
) -> tuple[int, int]:
    """Quantize a fractional denoising interval onto global solver-step edges."""
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    if len(interval) != 2:
        raise ValueError("denoising interval must contain exactly [start, end]")
    start, end = interval
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or not 0 <= start < end <= 1
    ):
        raise ValueError("denoising interval must satisfy 0 <= start < end <= 1")
    start_step = math.floor(start * steps + 0.5)
    end_step = math.floor(end * steps + 0.5)
    if start_step >= end_step:
        raise ValueError("denoising interval collapses to zero steps after rounding")
    return start_step, end_step


def local_denoising_schedule(
    interval: Sequence[float],
    steps: int,
    *,
    device: torch.device | str,
) -> Tensor:
    """Return a local 0-to-1 clock frozen outside its quantized interval."""
    start_step, end_step = denoising_step_bounds(interval, steps)
    edges = torch.arange(steps + 1, device=device, dtype=torch.float32)
    return ((edges - start_step) / (end_step - start_step)).clamp(0, 1)


def _expand(timestep: Tensor, like: Tensor) -> Tensor:
    result = timestep.float()
    while result.ndim < like.ndim:
        result = result.unsqueeze(-1)
    return result


def noisy_interpolate(clean: Tensor, noise: Tensor, timestep: Tensor) -> Tensor:
    if clean.shape != noise.shape:
        raise ValueError("clean data and noise shapes must match")
    t = _expand(timestep, clean)
    return t * clean.float() + (1 - t) * noise.float()


def x_to_velocity(
    noisy: Tensor, clean_prediction: Tensor, timestep: Tensor, floor: float = 0.05
) -> Tensor:
    if noisy.shape != clean_prediction.shape:
        raise ValueError("noisy data and clean prediction shapes must match")
    denominator = (1 - timestep.float()).clamp_min(floor)
    while denominator.ndim < noisy.ndim:
        denominator = denominator.unsqueeze(-1)
    return (clean_prediction.float() - noisy.float()) / denominator


def clean_prediction_velocity_loss(
    clean: Tensor,
    noise: Tensor,
    timestep: Tensor,
    prediction: Tensor,
) -> Tensor:
    noisy = noisy_interpolate(clean, noise, timestep)
    target_velocity = x_to_velocity(noisy, clean, timestep)
    predicted_velocity = x_to_velocity(noisy, prediction, timestep)
    return F.mse_loss(predicted_velocity, target_velocity)


@torch.compile
def jit_losses(
    clean_image: Tensor,
    image_noise: Tensor,
    timestep: Tensor,
    image_prediction: Tensor,
    *,
    clean_reg: Tensor | None = None,
    reg_noise: Tensor | None = None,
    reg_prediction: Tensor | None = None,
    reg_timestep: Tensor | None = None,
    repa_prediction: Tensor | None = None,
    repa_target: Tensor | None = None,
    reg_weight: float = 0.03,
    repa_weight: float = 0.5,
) -> JiTLoss:
    image_loss = clean_prediction_velocity_loss(
        clean_image, image_noise, timestep, image_prediction
    )
    zero = image_loss.new_zeros(())
    reg_values = (clean_reg, reg_noise, reg_prediction, reg_timestep)
    if any(value is not None for value in reg_values):
        if not all(value is not None for value in reg_values):
            raise ValueError(
                "clean REG tokens, REG noise, REG prediction, and REG timestep "
                "must be provided together"
            )
        assert (
            clean_reg is not None
            and reg_noise is not None
            and reg_prediction is not None
            and reg_timestep is not None
        )
        reg_loss = clean_prediction_velocity_loss(
            clean_reg, reg_noise, reg_timestep, reg_prediction
        )
        reg_x0_mse = F.mse_loss(reg_prediction.float(), clean_reg.float())
        reg_x0_cosine = F.cosine_similarity(
            reg_prediction.float(), clean_reg.float(), dim=-1
        ).mean()
    else:
        reg_loss = zero
        reg_x0_mse = zero
        reg_x0_cosine = zero
    if repa_prediction is None and repa_target is None:
        repa_loss = zero
    elif repa_prediction is None or repa_target is None:
        raise ValueError("REPA prediction and target must be provided together")
    else:
        repa_loss = negative_cosine_repa_loss(repa_prediction, repa_target)
    total = image_loss + reg_weight * reg_loss + repa_weight * repa_loss
    return JiTLoss(
        total,
        image_loss,
        reg_loss,
        repa_loss,
        reg_x0_mse,
        reg_x0_cosine,
    )


def shift_timestep(u: Tensor, alpha: float | Tensor) -> Tensor:
    return alpha * u / (1 + (alpha - 1) * u)
