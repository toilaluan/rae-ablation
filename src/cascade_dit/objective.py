from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import ObjectiveConfig


@dataclass(frozen=True)
class GaussianFlowPath:
    clean: Tensor
    noise: Tensor
    noisy: Tensor
    velocity: Tensor
    timestep: Tensor


@dataclass(frozen=True)
class FlowLoss:
    total: Tensor
    velocity_mse: Tensor
    cosine: Tensor
    source_consistency: Tensor


def _broadcast_timestep(timestep: Tensor, values: Tensor) -> Tensor:
    if timestep.ndim != 1 or timestep.shape[0] != values.shape[0]:
        raise ValueError("timesteps must have shape [B]")
    return timestep.reshape(timestep.shape[0], *((1,) * (values.ndim - 1))).to(values)


def shift_timestep(timestep: Tensor, shift: float) -> Tensor:
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
    if batch_size < 1 or scale <= 0 or shift <= 0:
        raise ValueError("invalid shifted-logit-normal timestep settings")
    logits = torch.randn(batch_size, generator=generator, device=generator.device)
    timesteps = (logits.mul(scale).add(loc)).sigmoid().to(device)
    return shift_timestep(timesteps, shift).clamp_(torch.finfo(torch.float32).eps, 1 - 1e-6)


def gaussian_flow_path(clean: Tensor, noise: Tensor, timestep: Tensor) -> GaussianFlowPath:
    if clean.shape != noise.shape:
        raise ValueError("clean targets and noise must have equal shapes")
    time = _broadcast_timestep(timestep, clean)
    noisy = (1 - time) * clean + time * noise
    return GaussianFlowPath(clean, noise, noisy, noise - clean, timestep)


def x0_to_velocity(
    noisy: Tensor,
    clean_prediction: Tensor,
    timestep: Tensor,
    denominator_floor: float = 0.05,
) -> Tensor:
    if noisy.shape != clean_prediction.shape:
        raise ValueError("noisy values and x0 predictions must have equal shapes")
    if denominator_floor <= 0:
        raise ValueError("velocity denominator floor must be positive")
    denominator = _broadcast_timestep(timestep, noisy).clamp_min(denominator_floor)
    return (noisy - clean_prediction) / denominator


def velocity_to_x0(noisy: Tensor, velocity: Tensor, timestep: Tensor) -> Tensor:
    if noisy.shape != velocity.shape:
        raise ValueError("noisy values and velocity must have equal shapes")
    return noisy - _broadcast_timestep(timestep, noisy) * velocity


def clean_prediction_loss(
    path: GaussianFlowPath,
    clean_prediction: Tensor,
    config: ObjectiveConfig,
    *,
    source_prediction: Tensor | None = None,
    source_consistency_weight: float = 0.0,
) -> FlowLoss:
    predicted_velocity = x0_to_velocity(
        path.noisy, clean_prediction, path.timestep, config.denominator_floor
    )
    velocity_mse = F.mse_loss(predicted_velocity.float(), path.velocity.float())
    if config.cosine_loss_weight:
        prediction_flat = predicted_velocity.float().reshape(predicted_velocity.shape[0], -1)
        target_flat = path.velocity.float().reshape(path.velocity.shape[0], -1)
        cosine = (1 - F.cosine_similarity(
            prediction_flat, target_flat, dim=-1, eps=config.cosine_loss_epsilon
        )).mean()
    else:
        cosine = velocity_mse.new_zeros(())
    if source_prediction is not None and source_consistency_weight:
        source_consistency = F.mse_loss(clean_prediction.float(), source_prediction.detach().float())
    else:
        source_consistency = velocity_mse.new_zeros(())
    total = (
        velocity_mse
        + config.cosine_loss_weight * cosine
        + source_consistency_weight * source_consistency
    )
    return FlowLoss(total, velocity_mse, cosine, source_consistency)
