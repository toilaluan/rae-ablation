from __future__ import annotations

import math
from itertools import pairwise
from typing import Literal

import torch
from torch import Tensor, nn

from .objective import shift_timestep, x_to_velocity

Solver = Literal["euler", "heun"]


def class_balanced_labels(sample_count: int, num_classes: int) -> Tensor:
    if sample_count < 1 or num_classes < 1 or sample_count % num_classes:
        raise ValueError("sample count must be positive and class balanced")
    return torch.arange(sample_count, dtype=torch.int64).remainder(num_classes)


def _guided_x(
    full: Tensor,
    base: Tensor,
    timestep: Tensor,
    scale: float,
    interval: tuple[float, float],
) -> Tensor:
    start, end = interval
    mask = ((timestep >= start) & (timestep <= end)).reshape(
        -1, *((1,) * (full.ndim - 1))
    )
    return torch.where(mask, base + scale * (full - base), full)


@torch.inference_mode()
def ode_sample(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    *,
    steps: int = 50,
    timestep_shift: float = 4.0,
    ig_scale: float = 1.78,
    ig_interval: tuple[float, float] = (0.1, 1.0),
    cfg_scale: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    solver: Solver = "euler",
    num_classes: int | None = None,
    velocity_denominator_floor: float = 0.05,
) -> Tensor:
    """Integrate noise (t=1) to data (t=0), guiding x before velocity conversion."""
    if steps < 1 or solver not in {"euler", "heun"}:
        raise ValueError("sampling requires positive steps and euler or heun solver")
    if not math.isfinite(timestep_shift) or timestep_shift <= 0:
        raise ValueError("timestep shift must be finite and positive")
    for name, scale in (("IG", ig_scale), ("CFG", cfg_scale)):
        if not math.isfinite(scale):
            raise ValueError(f"{name} scale must be finite")
    for name, interval in (("IG", ig_interval), ("CFG", cfg_interval)):
        if not 0 <= interval[0] < interval[1] <= 1:
            raise ValueError(f"{name} interval must satisfy 0 <= start < end <= 1")
    if labels.shape != (noise.shape[0],):
        raise ValueError("labels must have shape [B]")
    if num_classes is None:
        num_classes = getattr(getattr(model, "config", None), "num_classes", None)
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes is required")

    state = noise.float()
    schedule = shift_timestep(
        torch.linspace(1, 0, steps + 1, device=noise.device), timestep_shift
    )
    null_labels = torch.full_like(labels, num_classes)

    def predict_x(x: Tensor, timestep: Tensor) -> Tensor:
        use_cfg = cfg_scale != 1.0
        if use_cfg:
            model_x = torch.cat((x, x)).to(noise.dtype)
            model_t = timestep.expand(2 * x.shape[0])
            model_labels = torch.cat((labels, null_labels))
        else:
            model_x = x.to(noise.dtype)
            model_t = timestep.expand(x.shape[0])
            model_labels = labels
        output = model(model_x, model_t, model_labels)
        full, base = output.full_prediction.float(), output.base_prediction.float()
        if not use_cfg:
            return _guided_x(full, base, model_t, ig_scale, ig_interval)
        full_c, full_u = full.chunk(2)
        base_c, base_u = base.chunk(2)
        t = model_t[: x.shape[0]]
        cond = _guided_x(full_c, base_c, t, ig_scale, ig_interval)
        uncond = _guided_x(full_u, base_u, t, ig_scale, ig_interval)
        cfg_start, cfg_end = cfg_interval
        mask = ((t >= cfg_start) & (t <= cfg_end)).reshape(
            -1, *((1,) * (cond.ndim - 1))
        )
        return torch.where(mask, uncond + cfg_scale * (cond - uncond), cond)

    for step_index, (current, following) in enumerate(pairwise(schedule)):
        prediction = predict_x(state, current)
        velocity = x_to_velocity(
            state,
            prediction,
            current.expand(state.shape[0]),
            velocity_denominator_floor,
        )
        step_size = following - current
        proposed = state + step_size * velocity
        if solver == "heun" and step_index < steps - 1:
            corrected_x = predict_x(proposed, following)
            corrected_velocity = x_to_velocity(
                proposed,
                corrected_x,
                following.expand(state.shape[0]),
                velocity_denominator_floor,
            )
            state = state + 0.5 * step_size * (velocity + corrected_velocity)
        else:
            state = proposed
    return state


def euler_sample(*args: object, **kwargs: object) -> Tensor:
    kwargs["solver"] = "euler"
    return ode_sample(*args, **kwargs)  # type: ignore[arg-type]
