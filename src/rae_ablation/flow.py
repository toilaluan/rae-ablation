from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


class FlowLoss(NamedTuple):
    total: Tensor
    full: Tensor
    early: Tensor


def timestep_shift_alpha(latent_shape: tuple[int, int, int]) -> float:
    """Resolution-aware shift ``sqrt(C*H*W / 4096)`` (8 for RAE, 4 compact)."""
    return math.sqrt(math.prod(latent_shape) / 4096)


def shift_timestep(u: Tensor, alpha: float) -> Tensor:
    return alpha * u / (1 + (alpha - 1) * u)


def sample_timesteps(
    batch_size: int,
    latent_shape: tuple[int, int, int],
    generator: torch.Generator,
    device: torch.device | str,
) -> Tensor:
    normal = torch.randn(batch_size, generator=generator, device=device, dtype=torch.float32)
    return shift_timestep(normal.sigmoid(), timestep_shift_alpha(latent_shape))


def x_to_velocity(
    x_t: Tensor, x_prediction: Tensor, timestep: Tensor, floor: float = 0.05
) -> Tensor:
    denominator = timestep.float().clamp_min(floor)
    while denominator.ndim < x_t.ndim:
        denominator = denominator.unsqueeze(-1)
    return (x_t.float() - x_prediction.float()) / denominator


def flow_losses(
    clean: Tensor,
    noise: Tensor,
    timestep: Tensor,
    full_prediction: Tensor,
    early_prediction: Tensor,
) -> FlowLoss:
    target = noise.float() - clean.float()
    full_velocity = x_to_velocity(
        (1 - _expand(timestep, clean)) * clean.float() + _expand(timestep, clean) * noise.float(),
        full_prediction,
        timestep,
    )
    early_velocity = x_to_velocity(
        (1 - _expand(timestep, clean)) * clean.float() + _expand(timestep, clean) * noise.float(),
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
    return (1 - t) * clean.float() + t * noise.float()


def internal_guidance(
    early: Tensor,
    full: Tensor,
    timestep: Tensor | float,
    scale: float = 1.78,
    start: float = 0.1,
) -> Tensor:
    if isinstance(timestep, Tensor):
        active = (timestep >= start).to(full.dtype)
        while active.ndim < full.ndim:
            active = active.unsqueeze(-1)
        effective = 1 + active * (scale - 1)
    else:
        effective = scale if timestep >= start else 1.0
    return early + effective * (full - early)
