from __future__ import annotations

import math
from itertools import pairwise
from typing import Literal

import torch
from torch import Tensor, nn

from .objective import shift_timestep

Solver = Literal["euler", "heun"]


def class_balanced_labels(sample_count: int, num_classes: int) -> Tensor:
    if sample_count < 1 or num_classes < 1 or sample_count % num_classes:
        raise ValueError("sample count must be positive and class balanced")
    return torch.arange(sample_count, dtype=torch.int64).remainder(num_classes)


def _guided(conditional: Tensor, unconditional: Tensor, scale: float) -> Tensor:
    return unconditional + scale * (conditional - unconditional)


@torch.inference_mode()
def ode_sample(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    *,
    steps: int = 50,
    guidance_scale: float = 3.0,
    timestep_shift: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    solver: Solver = "euler",
    num_classes: int | None = None,
) -> Tensor:
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be euler or heun")
    if not math.isfinite(guidance_scale):
        raise ValueError("guidance scale must be finite")
    if not math.isfinite(timestep_shift) or timestep_shift <= 0:
        raise ValueError("timestep shift must be finite and positive")
    if not 0 <= cfg_interval[0] <= cfg_interval[1] <= 1:
        raise ValueError("CFG interval must satisfy 0 <= start <= end <= 1")
    if labels.shape != (noise.shape[0],):
        raise ValueError("labels must have shape [B]")
    if num_classes is None:
        num_classes = getattr(getattr(model, "config", None), "num_classes", None)
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes is required")

    schedule = shift_timestep(
        torch.linspace(0, 1, steps + 1, device=noise.device, dtype=torch.float32),
        timestep_shift,
    )
    state = noise.float()
    null_labels = torch.full_like(labels, num_classes)

    def predict(current_state: Tensor, timestep: Tensor) -> Tensor:
        use_cfg = cfg_interval[0] <= float(timestep) <= cfg_interval[1]
        if use_cfg:
            model_state = torch.cat((current_state, current_state)).to(noise.dtype)
            model_labels = torch.cat((labels, null_labels))
            model_time = timestep.expand(2 * noise.shape[0])
        else:
            model_state = current_state.to(noise.dtype)
            model_labels = labels
            model_time = timestep.expand(noise.shape[0])
        output = model(model_state, model_time, model_labels)
        if output.x0 is None:
            raise RuntimeError("sampling model did not produce x0")
        prediction = output.x0.float()
        if use_cfg:
            conditional, unconditional = prediction.chunk(2)
            prediction = _guided(conditional, unconditional, guidance_scale)
        return prediction

    for index, (current, following) in enumerate(pairwise(schedule)):
        prediction = predict(state, current)
        step = following - current
        velocity = (prediction - state) / (1 - current).clamp_min(0.05)
        proposed = state + step * velocity
        if solver == "heun" and index < steps - 1:
            corrected = predict(proposed, following)
            corrected_velocity = (corrected - proposed) / (1 - following).clamp_min(0.05)
            state = state + 0.5 * step * (velocity + corrected_velocity)
        else:
            state = proposed
    return state
