from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


class FlowLoss(NamedTuple):
    total: Tensor
    full: Tensor
    early: Tensor


class TimestepSample(NamedTuple):
    conditioning: Tensor
    interpolation: Tensor


def shift_timestep(u: Tensor, alpha: float) -> Tensor:
    return alpha * u / (1 + (alpha - 1) * u)


def sample_timesteps(
    batch_size: int,
    latent_shape: tuple[int, int, int],
    generator: torch.Generator,
    device: torch.device | str,
    *,
    use_patch_forcing: bool = False,
    loc: float = 0.0,
    scale: float = 1.0,
) -> TimestepSample:
    normal = torch.randn(batch_size, generator=generator, device=device, dtype=torch.float32)
    maximum = (loc + scale * normal).sigmoid()
    if not use_patch_forcing:
        return TimestepSample(maximum, maximum)

    height, width = latent_shape[-2:]
    maximum_per_patch = maximum.reshape(batch_size, 1, 1, 1)
    deviation = torch.randn(
        (batch_size, 1, height, width),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ) * (maximum_per_patch / 2)
    patchwise = maximum_per_patch - deviation.abs()
    replacement = torch.rand(
        patchwise.shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ) * maximum_per_patch
    patchwise = torch.where(patchwise < 0, replacement, patchwise)
    return TimestepSample(maximum, patchwise)


def x_to_velocity(
    x_t: Tensor, x_prediction: Tensor, timestep: Tensor, floor: float = 0.05
) -> Tensor:
    denominator = (1 - timestep.float()).clamp_min(floor)
    while denominator.ndim < x_t.ndim:
        denominator = denominator.unsqueeze(-1)
    return (x_prediction.float() - x_t.float()) / denominator


def flow_losses(
    clean: Tensor,
    noise: Tensor,
    timestep: Tensor,
    full_prediction: Tensor,
    early_prediction: Tensor,
) -> FlowLoss:
    target = clean.float() - noise.float()
    full_velocity = x_to_velocity(
        noisy_interpolate(clean, noise, timestep),
        full_prediction,
        timestep,
    )
    early_velocity = x_to_velocity(
        noisy_interpolate(clean, noise, timestep),
        early_prediction,
        timestep,
    )
    full = F.mse_loss(full_velocity, target)
    early = F.mse_loss(early_velocity, target)
    return FlowLoss(full + early, full, early)


def _expand(timestep: Tensor, like: Tensor) -> Tensor:
    result = timestep.float()
    while result.ndim < like.ndim:
        result = result.unsqueeze(-1)
    return result


def noisy_interpolate(clean: Tensor, noise: Tensor, timestep: Tensor) -> Tensor:
    t = _expand(timestep, clean)
    return t * clean.float() + (1 - t) * noise.float()


def internal_guidance(
    early: Tensor,
    full: Tensor,
    scale: float = 1.78,
) -> Tensor:
    return early + scale * (full - early)
