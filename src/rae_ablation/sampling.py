from __future__ import annotations

from itertools import pairwise

import torch
from torch import Tensor, nn

from .flow import internal_guidance, shift_timestep, timestep_shift_alpha


def class_balanced_labels(sample_count: int, num_classes: int) -> Tensor:
    """Return deterministic labels with the same number of samples per class."""
    if sample_count < 1 or num_classes < 1:
        raise ValueError("sample count and number of classes must be positive")
    if sample_count % num_classes:
        raise ValueError(
            "sample count must be class balanced (divisible by the number of classes)"
        )
    return torch.arange(sample_count, dtype=torch.int64).remainder(num_classes)


@torch.inference_mode()
def euler_sample(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    steps: int = 50,
    guidance_scale: float = 1.78,
    guidance_start: float = 0.1,
) -> Tensor:
    """Integrate the shifted rectified-flow schedule from noise (t=1) to data (t=0)."""
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    x = noise.float()
    alpha = timestep_shift_alpha(tuple(noise.shape[1:]))
    u = torch.linspace(1, 0, steps + 1, device=noise.device, dtype=torch.float32)
    schedule = shift_timestep(u, alpha)
    for current, following in pairwise(schedule):
        timestep = current.expand(noise.shape[0])
        output = model(x.to(noise.dtype), timestep, labels)
        x_prediction = internal_guidance(
            output.early.float(),
            output.full.float(),
            timestep,
            guidance_scale,
            guidance_start,
        )
        velocity = (x - x_prediction) / current.clamp_min(0.05)
        x = x + (following - current) * velocity
    return x
