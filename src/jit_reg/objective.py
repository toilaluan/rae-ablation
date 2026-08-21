from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import CascadeConfig

SEMANTIC_STAGE = 0
TARGET_STAGE = 1


class CascadePath(NamedTuple):
    noisy: Tensor
    clean: Tensor
    velocity: Tensor
    timestep: Tensor
    stage: Tensor
    semantic_endpoint: Tensor


class DiffusionLoss(NamedTuple):
    total: Tensor
    full_velocity: Tensor
    base_semantic_velocity: Tensor
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


def sample_stages(
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str,
    semantic_probability: float,
) -> Tensor:
    # if not 0 < semantic_probability < 1:
    #     raise ValueError("semantic stage probability must be in (0,1)")
    semantic = torch.rand(
        batch_size, generator=generator, device=device, dtype=torch.float32
    ) < semantic_probability
    return torch.where(
        semantic,
        torch.full_like(semantic, SEMANTIC_STAGE, dtype=torch.long),
        torch.full_like(semantic, TARGET_STAGE, dtype=torch.long),
    )


def _expand(value: Tensor, like: Tensor) -> Tensor:
    result = value.float()
    while result.ndim < like.ndim:
        result = result.unsqueeze(-1)
    return result


def pool_semantic(semantic: Tensor, config: CascadeConfig) -> Tensor:
    if semantic.ndim != 4 or semantic.shape[-1] != semantic.shape[-2]:
        raise ValueError("semantic tokens must have square [B,C,H,W] shape")
    if semantic.shape[-1] % config.pool_size:
        raise ValueError("semantic token grid must be divisible by cascade pool size")
    side = semantic.shape[-1]
    pooled_side = side // config.pool_size
    down_kwargs: dict[str, object] = {}
    if config.downsample_mode == "bilinear":
        down_kwargs["align_corners"] = config.align_corners
    pooled = F.interpolate(
        semantic.float(),
        size=(pooled_side, pooled_side),
        mode=config.downsample_mode,
        **down_kwargs,
    )
    up_kwargs: dict[str, object] = {}
    if config.upsample_mode == "bilinear":
        up_kwargs["align_corners"] = config.align_corners
    return F.interpolate(
        pooled,
        size=(side, side),
        mode=config.upsample_mode,
        **up_kwargs,
    )


def cascade_path(
    target: Tensor,
    semantic: Tensor,
    noise: Tensor,
    local_timestep: Tensor,
    stage: Tensor,
    config: CascadeConfig,
) -> CascadePath:
    if target.shape != semantic.shape or target.shape != noise.shape:
        raise ValueError("target, semantic, and noise shapes must match")
    batch = target.shape[0]
    if local_timestep.shape != (batch,) or stage.shape != (batch,):
        raise ValueError("local timestep and stage must have shape [B]")
    if torch.any((local_timestep < 0) | (local_timestep > 1)):
        raise ValueError("local timesteps must be in [0,1]")

    semantic_endpoint = pool_semantic(semantic, config)
    is_semantic = stage == SEMANTIC_STAGE
    timestep = local_timestep.float()
    t = _expand(timestep, target)
    semantic_noisy = (1 - t) * semantic_endpoint + t * noise.float()
    target_noisy = (1 - t) * target.float() + t * semantic_endpoint
    mask = _expand(is_semantic, target).bool()
    noisy = torch.where(mask, semantic_noisy, target_noisy)
    clean = torch.where(mask, semantic_endpoint, target.float())
    semantic_velocity = noise.float() - semantic_endpoint
    target_velocity = semantic_endpoint - target.float()
    velocity = torch.where(mask, semantic_velocity, target_velocity)
    return CascadePath(
        noisy,
        clean,
        velocity,
        timestep,
        stage.to(torch.long),
        semantic_endpoint,
    )


def x_to_velocity(
    noisy: Tensor,
    x_prediction: Tensor,
    timestep: Tensor,
    floor: float = 0.05,
) -> Tensor:
    if noisy.shape != x_prediction.shape:
        raise ValueError("noisy data and x-prediction shapes must match")
    if floor <= 0:
        raise ValueError("velocity denominator floor must be positive")
    return (noisy.float() - x_prediction.float()) / _expand(
        timestep.float().clamp_min(floor), noisy
    )


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
    noisy: Tensor,
    velocity_target: Tensor,
    timestep: Tensor,
    stage: Tensor,
    full_prediction: Tensor,
    base_prediction: Tensor,
    projection: Tensor | None,
    semantic: Tensor,
    *,
    base_loss_weight: float = 1.0,
    repa_coefficient: float = 0.5,
    velocity_denominator_floor: float = 0.05,
) -> DiffusionLoss:
    if velocity_target.shape != noisy.shape:
        raise ValueError("velocity target and noisy data shapes must match")
    if stage.shape != (noisy.shape[0],):
        raise ValueError("loss stage must have shape [B]")
    if torch.any((stage != SEMANTIC_STAGE) & (stage != TARGET_STAGE)):
        raise ValueError("loss stage contains an unknown cascade stage")
    if base_loss_weight < 0 or repa_coefficient < 0:
        raise ValueError("loss coefficients cannot be negative")
    full = F.mse_loss(
        x_to_velocity(noisy, full_prediction, timestep, velocity_denominator_floor),
        velocity_target.float(),
    )
    base_error = (
        x_to_velocity(noisy, base_prediction, timestep, velocity_denominator_floor)
        .sub(velocity_target.float())
        .square()
        .flatten(1)
        .mean(1)
    )
    semantic_mask = (stage == SEMANTIC_STAGE).to(base_error.dtype)
    base = (base_error * semantic_mask).sum() / semantic_mask.sum().clamp_min(1)
    if projection is None:
        if repa_coefficient:
            raise ValueError("REPA coefficient must be zero when its projector is disabled")
        repa = noisy.new_zeros((), dtype=torch.float32)
        similarity = noisy.new_zeros((), dtype=torch.float32)
    else:
        repa, similarity = repa_cosine_loss(projection, semantic)
    total = full + base_loss_weight * base + repa_coefficient * repa
    return DiffusionLoss(total, full, base, repa, similarity)
