from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


class GaussianFlowPath(NamedTuple):
    noisy: Tensor
    clean: Tensor
    velocity: Tensor
    timestep: Tensor
    noise: Tensor


class CascadeLoss(NamedTuple):
    total: Tensor
    image_latent_velocity: Tensor
    semantic_full_velocity: Tensor
    semantic_base_velocity: Tensor
    predicted_semantic_condition_fraction: Tensor


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


def gaussian_flow_path(
    clean: Tensor,
    noise: Tensor,
    timestep: Tensor,
) -> GaussianFlowPath:
    if clean.shape != noise.shape:
        raise ValueError("clean values and noise must have the same shape")
    if timestep.shape != (clean.shape[0],):
        raise ValueError("timestep must have shape [B]")
    if torch.any((timestep < 0) | (timestep > 1)):
        raise ValueError("timesteps must be in [0,1]")
    amount = _expand(timestep, clean)
    clean_float = clean.float()
    noise_float = noise.float()
    return GaussianFlowPath(
        (1 - amount) * clean_float + amount * noise_float,
        clean_float,
        noise_float - clean_float,
        timestep.float(),
        noise_float,
    )


def x0_to_velocity(
    noisy: Tensor,
    x0_prediction: Tensor,
    timestep: Tensor,
    floor: float = 0.05,
) -> Tensor:
    if noisy.shape != x0_prediction.shape:
        raise ValueError("noisy values and x0 prediction must have the same shape")
    if floor <= 0:
        raise ValueError("velocity denominator floor must be positive")
    return (noisy.float() - x0_prediction.float()) / _expand(
        timestep.float().clamp_min(floor), noisy
    )


def select_semantic_condition(
    clean_semantic_tokens: Tensor,
    predicted_semantic_tokens: Tensor,
    generator: torch.Generator,
    probability: float = 0.25,
) -> tuple[Tensor, Tensor]:
    """Select clean or detached online predictions independently per example."""
    if clean_semantic_tokens.shape != predicted_semantic_tokens.shape:
        raise ValueError("clean and predicted semantic tokens must have the same shape")
    if not 0 <= probability <= 1:
        raise ValueError("predicted semantic condition probability must be in [0,1]")
    mask = torch.rand(
        clean_semantic_tokens.shape[0],
        device=clean_semantic_tokens.device,
        generator=generator,
    ) < probability
    expanded = _expand(mask, clean_semantic_tokens).bool()
    condition = torch.where(
        expanded,
        predicted_semantic_tokens.detach(),
        clean_semantic_tokens.detach(),
    )
    return condition, mask


def velocity_mse(
    path: GaussianFlowPath,
    prediction: Tensor,
    velocity_denominator_floor: float = 0.05,
) -> Tensor:
    return F.mse_loss(
        x0_to_velocity(
            path.noisy,
            prediction,
            path.timestep,
            velocity_denominator_floor,
        ),
        path.velocity,
    )


def cascade_losses(
    *,
    semantic_path: GaussianFlowPath,
    image_latent_path: GaussianFlowPath,
    semantic_full_prediction: Tensor,
    semantic_base_prediction: Tensor,
    image_latent_prediction: Tensor,
    predicted_semantic_mask: Tensor,
    semantic_base_loss_weight: float = 1.0,
    velocity_denominator_floor: float = 0.05,
) -> CascadeLoss:
    if semantic_base_loss_weight < 0:
        raise ValueError("semantic base loss weight cannot be negative")
    if predicted_semantic_mask.shape != (semantic_path.noisy.shape[0],):
        raise ValueError("predicted semantic mask must have shape [B]")
    semantic_full = velocity_mse(
        semantic_path,
        semantic_full_prediction,
        velocity_denominator_floor,
    )
    semantic_base = velocity_mse(
        semantic_path,
        semantic_base_prediction,
        velocity_denominator_floor,
    )
    image_latent = velocity_mse(
        image_latent_path,
        image_latent_prediction,
        velocity_denominator_floor,
    )
    total = image_latent + semantic_full + semantic_base_loss_weight * semantic_base
    return CascadeLoss(
        total,
        image_latent,
        semantic_full,
        semantic_base,
        predicted_semantic_mask.float().mean(),
    )


def latent_loss(
    path: GaussianFlowPath,
    image_latent_prediction: Tensor,
    velocity_denominator_floor: float = 0.05,
) -> CascadeLoss:
    image_latent = velocity_mse(
        path, image_latent_prediction, velocity_denominator_floor
    )
    zero = image_latent.detach().new_zeros(())
    return CascadeLoss(image_latent, image_latent, zero, zero, zero)
