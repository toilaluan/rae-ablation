from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class DDTLoss(NamedTuple):
    total: Tensor
    full_velocity: Tensor
    base_velocity: Tensor

    @property
    def velocity(self) -> Tensor:
        return self.full_velocity


def sample_normal(
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str,
    *,
    loc: float = 0.0,
    scale: float = 1.0,
) -> Tensor:
    return loc + scale * torch.randn(
        batch_size, generator=generator, device=device, dtype=torch.float32
    )


def shift_timestep(timestep: Tensor, shift: float = 4.0) -> Tensor:
    """Apply the resolution-dependent logit shift used by RAE stage-2."""
    if shift <= 0:
        raise ValueError("timestep shift must be positive")
    return shift * timestep / (1 + (shift - 1) * timestep)


def sample_timesteps(
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str,
    *,
    loc: float = 0.0,
    scale: float = 1.0,
    shift: float = 4.0,
) -> Tensor:
    unshifted = sample_normal(
        batch_size, generator, device, loc=loc, scale=scale
    ).sigmoid()
    return shift_timestep(unshifted, shift)


def _expand(timestep: Tensor, like: Tensor) -> Tensor:
    value = timestep.float()
    while value.ndim < like.ndim:
        value = value.unsqueeze(-1)
    return value


def noisy_interpolate(clean: Tensor, noise: Tensor, timestep: Tensor) -> Tensor:
    """Straight flow path: x_t = (1 - t) * clean + t * noise."""
    if clean.shape != noise.shape:
        raise ValueError("clean data and noise shapes must match")
    t = _expand(timestep, clean)
    return (1 - t) * clean.float() + t * noise.float()


def x_to_velocity(
    noisy: Tensor,
    x_prediction: Tensor,
    timestep: Tensor,
    floor: float = 0.05,
) -> Tensor:
    """Convert a clean-x prediction to dx/dt with max(t, floor) denominator."""
    if noisy.shape != x_prediction.shape:
        raise ValueError("noisy data and x-prediction shapes must match")
    if floor <= 0:
        raise ValueError("velocity denominator floor must be positive")
    denominator = timestep.float().clamp_min(floor)
    while denominator.ndim < noisy.ndim:
        denominator = denominator.unsqueeze(-1)
    return (noisy.float() - x_prediction.float()) / denominator


def target_velocity(clean: Tensor, noise: Tensor) -> Tensor:
    if clean.shape != noise.shape:
        raise ValueError("clean data and noise shapes must match")
    return noise.float() - clean.float()


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


def ddt_losses(
    clean: Tensor,
    noise: Tensor,
    noisy: Tensor,
    timestep: Tensor,
    full_prediction: Tensor,
    base_prediction: Tensor,
    *,
    base_loss_weight: float = 1.0,
    velocity_denominator_floor: float = 0.05,
) -> DDTLoss:
    if base_loss_weight < 0:
        raise ValueError("base loss weight cannot be negative")
    full = per_sample_velocity_mse(
        clean,
        noise,
        noisy,
        timestep,
        full_prediction,
        floor=velocity_denominator_floor,
    ).mean()
    base = per_sample_velocity_mse(
        clean,
        noise,
        noisy,
        timestep,
        base_prediction,
        floor=velocity_denominator_floor,
    ).mean()
    return DDTLoss(full + base_loss_weight * base, full, base)
