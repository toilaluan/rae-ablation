from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import MixingConfig


class DiffusionLoss(NamedTuple):
    total: Tensor
    velocity: Tensor
    repa: Tensor
    cosine_similarity: Tensor


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
    unshifted = sample_normal(batch_size, generator, device, loc=loc, scale=scale).sigmoid()
    return shift_timestep(unshifted, shift)


def sample_mix_ratio(
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str,
    config: MixingConfig,
) -> Tensor:
    if config.mode == "target_baseline":
        return torch.ones(batch_size, device=device, dtype=torch.float32)
    continuous = sample_normal(
        batch_size,
        generator,
        device,
        loc=config.logit_normal_loc,
        scale=config.logit_normal_scale,
    ).sigmoid()
    endpoint = torch.rand(
        batch_size, generator=generator, device=device, dtype=torch.float32
    ) < config.endpoint_probability
    return torch.where(endpoint, torch.ones_like(continuous), continuous)


def _expand(value: Tensor, like: Tensor) -> Tensor:
    result = value.float()
    while result.ndim < like.ndim:
        result = result.unsqueeze(-1)
    return result


def mix_endpoints(semantic: Tensor, target: Tensor, ratio: Tensor) -> Tensor:
    if semantic.shape != target.shape or ratio.shape != (target.shape[0],):
        raise ValueError("endpoint tensors or mix ratio have incompatible shapes")
    r = _expand(ratio, target)
    return semantic.float() * (1 - r) + target.float() * r


def noisy_interpolate(clean: Tensor, noise: Tensor, timestep: Tensor) -> Tensor:
    if clean.shape != noise.shape:
        raise ValueError("clean data and noise shapes must match")
    t = _expand(timestep, clean)
    return (1 - t) * clean.float() + t * noise.float()


def target_velocity(clean: Tensor, noise: Tensor) -> Tensor:
    if clean.shape != noise.shape:
        raise ValueError("clean data and noise shapes must match")
    return noise.float() - clean.float()


def repa_cosine_loss(projection: Tensor, semantic: Tensor) -> tuple[Tensor, Tensor]:
    if projection.ndim != 3 or semantic.ndim != 4:
        raise ValueError("REPA expects [B,N,C] projections and [B,C,H,W] targets")
    target = semantic.flatten(2).transpose(1, 2)
    if projection.shape != target.shape:
        raise ValueError("REPA projection and semantic target shapes must match")
    similarity = (
        F.normalize(projection.float(), dim=-1) * F.normalize(target.float(), dim=-1)
    ).sum(dim=-1)
    return -similarity.mean(), similarity.mean()


def diffusion_losses(
    clean: Tensor,
    noise: Tensor,
    prediction: Tensor,
    projection: Tensor,
    semantic: Tensor,
    *,
    repa_coefficient: float = 0.5,
) -> DiffusionLoss:
    if prediction.shape != clean.shape:
        raise ValueError("velocity prediction and clean data shapes must match")
    if repa_coefficient < 0:
        raise ValueError("REPA coefficient cannot be negative")
    velocity = F.mse_loss(prediction.float(), target_velocity(clean, noise))
    repa, similarity = repa_cosine_loss(projection, semantic)
    return DiffusionLoss(velocity + repa_coefficient * repa, velocity, repa, similarity)
