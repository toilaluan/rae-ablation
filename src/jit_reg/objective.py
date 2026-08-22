from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


class FlowPath(NamedTuple):
    noisy: Tensor
    clean: Tensor
    noise: Tensor
    timestep: Tensor


class FlowLoss(NamedTuple):
    total: Tensor
    endpoint: Tensor


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


def _expand(value: Tensor, like: Tensor) -> Tensor:
    result = value.float()
    while result.ndim < like.ndim:
        result = result.unsqueeze(-1)
    return result


def flow_path(endpoint: Tensor, noise: Tensor, timestep: Tensor) -> FlowPath:
    """Construct an endpoint-to-standard-normal rectified-flow path."""
    if endpoint.shape != noise.shape:
        raise ValueError("endpoint and noise tensors must have matching shapes")
    if timestep.shape != (endpoint.shape[0],):
        raise ValueError("timestep must have shape [B]")
    if torch.any((timestep < 0) | (timestep > 1)):
        raise ValueError("timesteps must be in [0,1]")
    t = _expand(timestep, endpoint)
    endpoint_f = endpoint.float()
    noise_f = noise.float()
    noisy = (1 - t) * endpoint_f + t * noise_f
    return FlowPath(noisy, endpoint_f, noise_f, timestep.float())


def x_to_velocity(
    noisy: Tensor,
    x_prediction: Tensor,
    timestep: Tensor,
) -> Tensor:
    if noisy.shape != x_prediction.shape:
        raise ValueError("noisy data and endpoint prediction shapes must match")
    if timestep.shape != (noisy.shape[0],):
        raise ValueError("timestep must have shape [B]")
    if torch.any(timestep <= 0):
        raise ValueError("velocity conversion requires strictly positive timesteps")
    return (noisy.float() - x_prediction.float()) / _expand(timestep.float(), noisy)


def flow_losses(
    endpoint_prediction: Tensor,
    endpoint_target: Tensor,
) -> FlowLoss:
    if endpoint_prediction.shape != endpoint_target.shape:
        raise ValueError("endpoint prediction and target shapes must match")
    endpoint = F.mse_loss(endpoint_prediction.float(), endpoint_target.float())
    return FlowLoss(endpoint, endpoint)
