from __future__ import annotations

import math
from collections.abc import Callable
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


def _velocity_method(model: nn.Module) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
    method = getattr(model, "forward_velocity", None)
    if not callable(method) and hasattr(model, "module"):
        method = getattr(model.module, "forward_velocity", None)
    if not callable(method):
        raise TypeError("model does not provide forward_velocity")
    return method


@torch.inference_mode()
def ode_sample(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    *,
    steps: int = 50,
    timestep_shift: float = 1.0,
    cfg_scale: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    solver: Solver = "euler",
    num_classes: int | None = None,
) -> Tensor:
    """Integrate direct velocity predictions from Gaussian noise to r=1 target tokens."""
    if steps < 1 or solver not in {"euler", "heun"}:
        raise ValueError("sampling requires positive steps and euler or heun solver")
    if not math.isfinite(timestep_shift) or timestep_shift <= 0:
        raise ValueError("timestep shift must be finite and positive")
    if not math.isfinite(cfg_scale):
        raise ValueError("CFG scale must be finite")
    if not 0 <= cfg_interval[0] < cfg_interval[1] <= 1:
        raise ValueError("CFG interval must satisfy 0 <= start < end <= 1")
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
    forward_velocity = _velocity_method(model)

    def predict(x: Tensor, timestep: Tensor) -> Tensor:
        use_cfg = cfg_scale != 1.0
        ratio = torch.ones(x.shape[0], device=x.device, dtype=torch.float32)
        if not use_cfg:
            return forward_velocity(
                x.to(noise.dtype),
                timestep.expand(x.shape[0]),
                ratio,
                labels,
            ).float()
        output = forward_velocity(
            torch.cat((x, x)).to(noise.dtype),
            timestep.expand(2 * x.shape[0]),
            torch.cat((ratio, ratio)),
            torch.cat((labels, null_labels)),
        ).float()
        conditional, unconditional = output.chunk(2)
        start, end = cfg_interval
        active = ((timestep >= start) & (timestep <= end)).reshape(
            -1, *((1,) * (conditional.ndim - 1))
        )
        guided = unconditional + cfg_scale * (conditional - unconditional)
        return torch.where(active, guided, conditional)

    for step_index, (current, following) in enumerate(pairwise(schedule)):
        velocity = predict(state, current)
        step_size = following - current
        proposed = state + step_size * velocity
        if solver == "heun" and step_index < steps - 1:
            corrected = predict(proposed, following)
            state = state + 0.5 * step_size * (velocity + corrected)
        else:
            state = proposed
    return state


def euler_sample(*args: object, **kwargs: object) -> Tensor:
    kwargs["solver"] = "euler"
    return ode_sample(*args, **kwargs)  # type: ignore[arg-type]
