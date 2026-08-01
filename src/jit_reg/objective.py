from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


class SiTLoss(NamedTuple):
    total: Tensor
    velocity: Tensor
    repa: Tensor


def sample_normal(
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str,
    *,
    loc: float = 0.0,
    scale: float = 1.0,
) -> Tensor:
    return loc + scale * torch.randn(
        batch_size,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )


def sample_timesteps(
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str,
    *,
    loc: float = 0.0,
    scale: float = 1.0,
) -> Tensor:
    return sample_normal(
        batch_size,
        generator,
        device,
        loc=loc,
        scale=scale,
    ).sigmoid()


def _expand(timestep: Tensor, like: Tensor) -> Tensor:
    value = timestep.float()
    while value.ndim < like.ndim:
        value = value.unsqueeze(-1)
    return value


def noisy_interpolate(clean: Tensor, noise: Tensor, timestep: Tensor) -> Tensor:
    if clean.shape != noise.shape:
        raise ValueError("clean data and noise shapes must match")
    t = _expand(timestep, clean)
    return t * clean.float() + (1 - t) * noise.float()


def x_to_velocity(
    noisy: Tensor,
    x_prediction: Tensor,
    timestep: Tensor,
    floor: float = 0.05,
) -> Tensor:
    if noisy.shape != x_prediction.shape:
        raise ValueError("noisy data and x-prediction shapes must match")
    denominator = (1 - timestep.float()).clamp_min(floor)
    while denominator.ndim < noisy.ndim:
        denominator = denominator.unsqueeze(-1)
    return (x_prediction.float() - noisy.float()) / denominator


def target_velocity(clean: Tensor, noise: Tensor) -> Tensor:
    if clean.shape != noise.shape:
        raise ValueError("clean data and noise shapes must match")
    return clean.float() - noise.float()


def per_sample_velocity_mse(
    clean: Tensor,
    noise: Tensor,
    noisy: Tensor,
    timestep: Tensor,
    x_prediction: Tensor,
    *,
    floor: float = 0.05,
) -> Tensor:
    predicted = x_to_velocity(noisy, x_prediction, timestep, floor)
    target = target_velocity(clean, noise)
    return (predicted - target).square().flatten(1).mean(dim=1)


def per_sample_repa_cosine_loss(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("REPA prediction and target shapes must match")
    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return 1 - (prediction * target).sum(dim=-1).mean(dim=-1)


def sit_losses(
    clean: Tensor,
    noise: Tensor,
    noisy: Tensor,
    timestep: Tensor,
    x_prediction: Tensor,
    *,
    repa_prediction: Tensor | None = None,
    repa_target: Tensor | None = None,
    repa_weight: float = 0.5,
    velocity_denominator_floor: float = 0.05,
) -> SiTLoss:
    velocity = per_sample_velocity_mse(
        clean,
        noise,
        noisy,
        timestep,
        x_prediction,
        floor=velocity_denominator_floor,
    ).mean()
    zero = velocity.new_zeros(())
    if repa_prediction is None and repa_target is None:
        repa = zero
    elif repa_prediction is None or repa_target is None:
        raise ValueError("REPA prediction and target must be provided together")
    else:
        repa = per_sample_repa_cosine_loss(repa_prediction, repa_target).mean()
    return SiTLoss(velocity + repa_weight * repa, velocity, repa)


def shift_timestep(u: Tensor, alpha: float | Tensor) -> Tensor:
    return alpha * u / (1 + (alpha - 1) * u)


def denoising_step_bounds(interval: tuple[float, float], steps: int) -> tuple[int, int]:
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    start, end = interval
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= 1:
        raise ValueError("denoising interval must satisfy 0 <= start < end <= 1")
    start_step = math.floor(start * steps + 0.5)
    end_step = math.floor(end * steps + 0.5)
    if start_step >= end_step:
        raise ValueError("denoising interval collapses to zero steps after rounding")
    return start_step, end_step


def local_denoising_schedule(
    interval: tuple[float, float],
    steps: int,
    *,
    device: torch.device | str,
) -> Tensor:
    start_step, end_step = denoising_step_bounds(interval, steps)
    edges = torch.arange(steps + 1, device=device, dtype=torch.float32)
    return ((edges - start_step) / (end_step - start_step)).clamp(0, 1)
