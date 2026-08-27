from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
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
class JointLoss:
    total: Tensor
    full_velocity_mse: Tensor
    base_velocity_mse: Tensor
    pixel_l1: Tensor
    pixel_lpips: Tensor
    weighted_base_velocity_mse: Tensor
    weighted_pixel_l1: Tensor
    weighted_pixel_lpips: Tensor


FlowLoss = JointLoss


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


def build_lpips(device: torch.device | str) -> nn.Module:
    """Build the frozen VGG LPIPS evaluator outside the trainable DDT."""
    import lpips

    perceptual = lpips.LPIPS(net="vgg").to(device).eval()
    perceptual.requires_grad_(False)
    return perceptual


def flow_matching_loss(
    path: GaussianFlowPath,
    output: DDTOutput,
    pixel_target: Tensor,
    lpips_model: nn.Module,
    config: ObjectiveConfig,
) -> JointLoss:
    """Compute full/base semantic flow loss and raw-RGB reconstruction loss."""
    if output.rgb is None:
        raise ValueError("joint training requires RGB decoding")
    if pixel_target.ndim != 4 or pixel_target.shape != output.rgb.shape:
        raise ValueError("pixel targets and RGB predictions must have equal BCHW shapes")
    full_velocity = x0_to_velocity(
        path.noisy, output.full_x0, path.timestep, config.denominator_floor
    )
    base_velocity = x0_to_velocity(
        path.noisy, output.base_x0, path.timestep, config.denominator_floor
    )
    full_mse = F.mse_loss(full_velocity.float(), path.velocity.float())
    base_mse = F.mse_loss(base_velocity.float(), path.velocity.float())
    pixel_l1 = F.l1_loss(output.rgb.float(), pixel_target.float())
    predicted_lpips = output.rgb.float().mul(2).sub(1)
    target_lpips = pixel_target.float().mul(2).sub(1)
    pixel_lpips = lpips_model(predicted_lpips, target_lpips).float().mean()
    weighted_base = config.base_velocity_weight * base_mse
    weighted_l1 = config.pixel_l1_weight * pixel_l1
    weighted_lpips = config.pixel_lpips_weight * pixel_lpips
    return JointLoss(
        full_mse + weighted_base + weighted_l1 + weighted_lpips,
        full_mse,
        base_mse,
        pixel_l1,
        pixel_lpips,
        weighted_base,
        weighted_l1,
        weighted_lpips,
    )
