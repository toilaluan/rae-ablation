from __future__ import annotations

import math
from itertools import pairwise
from typing import NamedTuple, overload

import torch
from torch import Tensor, nn

from .objective import shift_timestep


class EulerSample(NamedTuple):
    image: Tensor
    reg: Tensor


def class_balanced_labels(sample_count: int, num_classes: int) -> Tensor:
    if sample_count < 1 or num_classes < 1:
        raise ValueError("sample count and number of classes must be positive")
    if sample_count % num_classes:
        raise ValueError("sample count must be class balanced (divisible by the number of classes)")
    return torch.arange(sample_count, dtype=torch.int64).remainder(num_classes)


def _guided(conditional: Tensor, unconditional: Tensor, scale: float) -> Tensor:
    return unconditional + scale * (conditional - unconditional)


@overload
def euler_sample(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    *,
    reg_noise: None = None,
    steps: int = 50,
    guidance_scale: float = 1.78,
    timestep_shift: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    num_classes: int | None = None,
) -> Tensor: ...


@overload
def euler_sample(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    *,
    reg_noise: Tensor,
    steps: int = 50,
    guidance_scale: float = 1.78,
    timestep_shift: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    num_classes: int | None = None,
) -> EulerSample: ...


@torch.inference_mode()
def euler_sample(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    *,
    reg_noise: Tensor | None = None,
    steps: int = 50,
    guidance_scale: float = 1.78,
    timestep_shift: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    num_classes: int | None = None,
) -> Tensor | EulerSample:
    """Integrate clean predictions from noise to data with batched standard CFG."""
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    if not math.isfinite(timestep_shift) or timestep_shift <= 0:
        raise ValueError("timestep shift must be finite and positive")
    if not math.isfinite(guidance_scale):
        raise ValueError("guidance scale must be finite")
    cfg_start, cfg_end = cfg_interval
    if not (0 <= cfg_start <= cfg_end <= 1):
        raise ValueError("CFG interval must satisfy 0 <= start <= end <= 1")
    if labels.shape != (noise.shape[0],):
        raise ValueError("labels must have shape [B]")
    if reg_noise is not None and reg_noise.shape[0] != noise.shape[0]:
        raise ValueError("REG noise batch dimension must match image noise")
    if num_classes is None:
        config = getattr(model, "config", None)
        num_classes = getattr(config, "num_classes", None)
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes is required when it cannot be read from model.config")

    x = noise.float()
    reg_state = reg_noise.float() if reg_noise is not None else None
    null_labels = torch.full_like(labels, num_classes)
    combined_labels = torch.cat((labels, null_labels))
    u = torch.linspace(0, 1, steps + 1, device=noise.device, dtype=torch.float32)
    schedule = shift_timestep(u, timestep_shift)
    for current, following in pairwise(schedule):
        timestep = current.expand(2 * noise.shape[0])
        combined_image = torch.cat((x, x)).to(noise.dtype)
        combined_reg = None
        if reg_state is not None:
            combined_reg = torch.cat((reg_state, reg_state)).to(reg_noise.dtype)
        output = model(
            combined_image,
            timestep,
            combined_labels,
            noisy_reg=combined_reg,
            return_repa=False,
        )
        conditional_image, unconditional_image = output.image.float().chunk(2)
        use_cfg = cfg_start <= float(current) <= cfg_end
        image_prediction = (
            _guided(conditional_image, unconditional_image, guidance_scale)
            if use_cfg
            else conditional_image
        )
        denominator = (1 - current).clamp_min(0.05)
        x = x + (following - current) * (image_prediction - x) / denominator

        if reg_state is not None:
            if output.reg is None:
                raise RuntimeError("REG sampling requires a model REG-token prediction")
            conditional_reg, unconditional_reg = output.reg.float().chunk(2)
            reg_prediction = (
                _guided(conditional_reg, unconditional_reg, guidance_scale)
                if use_cfg
                else conditional_reg
            )
            reg_denominator = (1 - current).clamp_min(0.05)
            reg_state = (
                reg_state
                + (following - current) * (reg_prediction - reg_state) / reg_denominator
            )
    if reg_state is None:
        return x
    return EulerSample(x, reg_state)
