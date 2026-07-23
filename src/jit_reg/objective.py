from __future__ import annotations

import math
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


def sample_reg_timesteps(
    image_timestep: Tensor,
    generator: torch.Generator,
    *,
    min_relative_shift: float,
    max_relative_shift: float,
) -> tuple[Tensor, Tensor]:
    """Sample per-example log-uniform REG shifts relative to image time."""
    if image_timestep.ndim != 1:
        raise ValueError("image timestep must have shape [B]")
    if (
        not math.isfinite(min_relative_shift)
        or not math.isfinite(max_relative_shift)
        or min_relative_shift <= 0
        or max_relative_shift <= 0
        or min_relative_shift > max_relative_shift
    ):
        raise ValueError("relative REG timestep shifts must be finite, positive, and ordered")
    if min_relative_shift == max_relative_shift:
        relative_shift = torch.full_like(image_timestep, min_relative_shift)
    else:
        log_shift = torch.empty_like(image_timestep).uniform_(
            math.log(min_relative_shift),
            math.log(max_relative_shift),
            generator=generator,
        )
        relative_shift = log_shift.exp()
    return shift_timestep(image_timestep, relative_shift), relative_shift


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
    reg_values = (clean_reg, reg_noise, reg_prediction)
    if any(value is not None for value in reg_values):
        if not all(value is not None for value in reg_values):
            raise ValueError(
                "clean REG tokens, REG noise, and REG prediction must be provided together"
            )
        assert clean_reg is not None and reg_noise is not None and reg_prediction is not None
        resolved_reg_timestep = timestep if reg_timestep is None else reg_timestep
        reg_loss = clean_prediction_velocity_loss(
            clean_reg, reg_noise, resolved_reg_timestep, reg_prediction
        )
        reg_x0_mse = F.mse_loss(reg_prediction.float(), clean_reg.float())
        reg_x0_cosine = F.cosine_similarity(
            reg_prediction.float(), clean_reg.float(), dim=-1
        ).mean()
    else:
        if reg_timestep is not None:
            raise ValueError("REG timestep requires clean REG values and a REG prediction")
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
