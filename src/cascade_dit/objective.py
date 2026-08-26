from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import ObjectiveConfig
from .model import DDTOutput


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
    pixel_auxiliary_mse: Tensor
    weighted_pixel_auxiliary_mse: Tensor
    pure_semantic_count: int


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
    return shift_timestep(timesteps, shift).clamp_(
        torch.finfo(torch.float32).eps, 1 - 1e-6
    )


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


def flow_matching_loss(
    path: GaussianFlowPath,
    output: DDTOutput,
    pixel_target: Tensor,
    pixel_ratio: Tensor,
    config: ObjectiveConfig,
) -> FlowLoss:
    if pixel_target.shape != path.clean.shape:
        raise ValueError("pixel target must have the clean endpoint shape")
    if pixel_ratio.shape != (path.clean.shape[0],):
        raise ValueError("pixel_ratio must have shape [B]")
    predicted_velocity = x0_to_velocity(
        path.noisy,
        output.prediction,
        path.timestep,
        config.denominator_floor,
    )
    velocity_mse = F.mse_loss(predicted_velocity.float(), path.velocity.float())
    pure_semantic = pixel_ratio == 0
    pure_semantic_count = int(pure_semantic.sum().item())
    if pure_semantic_count:
        if output.pixel_x0 is None:
            raise ValueError("pure-semantic items require a pixel prediction")
        pixel_auxiliary_mse = F.mse_loss(
            output.pixel_x0[pure_semantic].float(),
            pixel_target[pure_semantic].float(),
        )
    elif output.pixel_x0 is not None:
        pixel_auxiliary_mse = output.pixel_x0.float().sum() * 0
    else:
        pixel_auxiliary_mse = velocity_mse.new_zeros(())
    weighted = config.pixel_auxiliary_mse_weight * pixel_auxiliary_mse
    return FlowLoss(
        velocity_mse + weighted,
        velocity_mse,
        pixel_auxiliary_mse,
        weighted,
        pure_semantic_count,
    )
