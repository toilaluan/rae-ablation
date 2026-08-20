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


def sample_timesteps(
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str,
    *,
    loc: float = 0.0,
    scale: float = 1.0,
) -> Tensor:
    return sample_normal(
        batch_size, generator, device, loc=loc, scale=scale
    ).sigmoid()


def sample_mix_ratios(
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str,
    *,
    loc: float = 0.0,
    scale: float = 1.0,
    rae_endpoint_probability: float = 0.1,
    pixel_endpoint_probability: float = 0.1,
    fixed_ratio: float | None = None,
) -> Tensor:
    if batch_size < 1 or scale <= 0:
        raise ValueError("ratio sampling requires positive batch size and scale")
    if fixed_ratio is not None:
        if not 0 <= fixed_ratio <= 1:
            raise ValueError("fixed ratio must be in [0, 1]")
        return torch.full((batch_size,), fixed_ratio, device=device, dtype=torch.float32)
    if (
        not 0 <= rae_endpoint_probability <= 1
        or not 0 <= pixel_endpoint_probability <= 1
        or rae_endpoint_probability + pixel_endpoint_probability > 1
    ):
        raise ValueError("ratio endpoint probabilities are invalid")
    selector = torch.rand(batch_size, generator=generator, device=device)
    continuous = sample_normal(
        batch_size, generator, device, loc=loc, scale=scale
    ).sigmoid()
    result = torch.where(
        selector < rae_endpoint_probability,
        torch.zeros_like(continuous),
        continuous,
    )
    return torch.where(
        selector >= 1 - pixel_endpoint_probability,
        torch.ones_like(result),
        result,
    )


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


def mix_endpoints(pixel: Tensor, rae: Tensor, ratio: Tensor) -> Tensor:
    if pixel.shape != rae.shape:
        raise ValueError("pixel and RAE token shapes must match")
    if ratio.shape != (pixel.shape[0],):
        raise ValueError("mix ratio must have shape [B]")
    r = _expand(ratio, pixel)
    return r * pixel.float() + (1 - r) * rae.float()


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
