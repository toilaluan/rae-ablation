from __future__ import annotations

import math
from itertools import pairwise

import torch
from torch import Tensor, nn

from .flow import internal_guidance, shift_timestep


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
    timestep_shift: float = 1.0,
) -> Tensor:
    """Integrate rectified flow from noise (t=0) to data (t=1)."""
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    if not math.isfinite(timestep_shift) or timestep_shift <= 0:
        raise ValueError("timestep shift must be finite and positive")
    x = noise.float()
    u = torch.linspace(0, 1, steps + 1, device=noise.device, dtype=torch.float32)
    schedule = shift_timestep(u, timestep_shift)
    for current, following in pairwise(schedule):
        timestep = current.expand(noise.shape[0])
        output = model(x.to(noise.dtype), timestep, labels)
        x_prediction = internal_guidance(
            output.early.float(),
            output.full.float(),
            guidance_scale,
        )
        velocity = (x_prediction - x) / (1 - current).clamp_min(0.05)
        x = x + (following - current) * velocity
    return x
