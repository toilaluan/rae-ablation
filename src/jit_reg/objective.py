from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .reg import negative_cosine_repa_loss


class JiTLoss(NamedTuple):
    total: Tensor
    image_velocity: Tensor
    cls_velocity: Tensor
    repa: Tensor


def sample_timesteps(
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str,
    *,
    loc: float = 0.0,
    scale: float = 1.0,
) -> Tensor:
    normal = torch.randn(
        batch_size, generator=generator, device=device, dtype=torch.float32
    )
    return (loc + scale * normal).sigmoid()


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


def jit_losses(
    clean_image: Tensor,
    image_noise: Tensor,
    timestep: Tensor,
    image_prediction: Tensor,
    *,
    clean_cls: Tensor | None = None,
    cls_noise: Tensor | None = None,
    cls_prediction: Tensor | None = None,
    repa_prediction: Tensor | None = None,
    repa_target: Tensor | None = None,
    cls_weight: float = 0.03,
    repa_weight: float = 0.5,
) -> JiTLoss:
    image_loss = clean_prediction_velocity_loss(
        clean_image, image_noise, timestep, image_prediction
    )
    zero = image_loss.new_zeros(())
    cls_values = (clean_cls, cls_noise, cls_prediction)
    if any(value is not None for value in cls_values):
        if not all(value is not None for value in cls_values):
            raise ValueError("clean CLS, CLS noise, and CLS prediction must be provided together")
        assert clean_cls is not None and cls_noise is not None and cls_prediction is not None
        cls_loss = clean_prediction_velocity_loss(
            clean_cls, cls_noise, timestep, cls_prediction
        )
    else:
        cls_loss = zero
    if repa_prediction is None and repa_target is None:
        repa_loss = zero
    elif repa_prediction is None or repa_target is None:
        raise ValueError("REPA prediction and target must be provided together")
    else:
        repa_loss = negative_cosine_repa_loss(repa_prediction, repa_target)
    total = image_loss + cls_weight * cls_loss + repa_weight * repa_loss
    return JiTLoss(total, image_loss, cls_loss, repa_loss)


def shift_timestep(u: Tensor, alpha: float) -> Tensor:
    return alpha * u / (1 + (alpha - 1) * u)
