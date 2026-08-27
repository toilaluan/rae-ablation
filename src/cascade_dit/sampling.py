from __future__ import annotations

import math
from collections.abc import Callable
from itertools import pairwise

import torch
from torch import Tensor, nn

from .objective import shift_timestep, x0_to_velocity


def class_balanced_labels(sample_count: int, num_classes: int) -> Tensor:
    if sample_count < 1 or num_classes < 1 or sample_count % num_classes:
        raise ValueError("sample count must be positive and divisible by num_classes")
    return torch.arange(sample_count, dtype=torch.int64) % num_classes


def internal_guidance_x0(
    full_x0: Tensor,
    base_x0: Tensor,
    timestep: Tensor,
    *,
    scale: float = 1.78,
    t_min: float = 0.10,
    t_max: float = 1.0,
) -> Tensor:
    if full_x0.shape != base_x0.shape:
        raise ValueError("full and base predictions must have equal shapes")
    if timestep.shape != (full_x0.shape[0],):
        raise ValueError("internal-guidance timesteps must have shape [B]")
    if not math.isfinite(scale) or scale <= 0 or not 0 <= t_min < t_max <= 1:
        raise ValueError("invalid internal-guidance settings")
    active = ((timestep >= t_min) & (timestep <= t_max)).reshape(
        timestep.shape[0], *((1,) * (full_x0.ndim - 1))
    )
    guided = base_x0 + scale * (full_x0 - base_x0)
    return torch.where(active, guided, full_x0)


def _semantic_predictions(output: object) -> tuple[Tensor, Tensor]:
    full = getattr(output, "full_x0", None)
    base = getattr(output, "base_x0", None)
    if not isinstance(full, Tensor) or not isinstance(base, Tensor):
        raise TypeError("model output must expose tensor full_x0 and base_x0 predictions")
    return full, base


@torch.no_grad()
def sample_tokens(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    *,
    steps: int = 40,
    solver: str = "euler",
    timestep_shift: float = 1.0,
    internal_guidance_scale: float = 1.78,
    internal_guidance_t_min: float = 0.10,
    internal_guidance_t_max: float = 1.0,
    denominator_floor: float = 0.05,
    callback: Callable[[int, Tensor], None] | None = None,
) -> Tensor:
    """Integrate the internally guided semantic flow from t=1 to t=0."""
    if steps < 1 or solver not in {"euler", "heun"}:
        raise ValueError("sampling requires positive steps and the euler or heun solver")
    if labels.shape != (noise.shape[0],):
        raise ValueError("sampling labels must have shape [B]")
    if not 0 <= internal_guidance_t_min < internal_guidance_t_max <= 1:
        raise ValueError("invalid internal-guidance interval")
    base_grid = torch.linspace(1, 0, steps + 1, device=noise.device, dtype=torch.float32)
    grid = shift_timestep(base_grid, timestep_shift)
    values = noise

    def velocity(state: Tensor, scalar_time: Tensor) -> Tensor:
        time = scalar_time.expand(state.shape[0])
        full_x0, base_x0 = _semantic_predictions(
            model(state, time, labels, decode_pixels=False)
        )
        clean = internal_guidance_x0(
            full_x0,
            base_x0,
            time,
            scale=internal_guidance_scale,
            t_min=internal_guidance_t_min,
            t_max=internal_guidance_t_max,
        )
        return x0_to_velocity(state, clean, time, denominator_floor)

    for index, (current, following) in enumerate(pairwise(grid)):
        delta = following - current
        current_velocity = velocity(values, current)
        if solver == "heun" and index < steps - 1:
            proposal = values + delta * current_velocity
            next_velocity = velocity(proposal, following)
            values = values + delta * (current_velocity + next_velocity) / 2
        else:
            values = values + delta * current_velocity
        if callback is not None:
            callback(index + 1, values)
    return values


sample_semantic_tokens = sample_tokens


@torch.no_grad()
def sample_semantic_then_pixel(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    *,
    steps: int = 40,
    solver: str = "euler",
    timestep_shift: float = 1.0,
    internal_guidance_scale: float = 1.78,
    internal_guidance_t_min: float = 0.10,
    internal_guidance_t_max: float = 1.0,
    denominator_floor: float = 0.05,
) -> Tensor:
    """Integrate semantics, then run one un-guided conditional t=0 RGB decode."""
    semantic = sample_tokens(
        model,
        noise,
        labels,
        steps=steps,
        solver=solver,
        timestep_shift=timestep_shift,
        internal_guidance_scale=internal_guidance_scale,
        internal_guidance_t_min=internal_guidance_t_min,
        internal_guidance_t_max=internal_guidance_t_max,
        denominator_floor=denominator_floor,
    )
    zero = torch.zeros(noise.shape[0], device=noise.device, dtype=torch.float32)
    output = model(semantic, zero, labels, decode_pixels=True)
    pixels = getattr(output, "rgb", None)
    if not isinstance(pixels, Tensor):
        pixels = getattr(output, "pixels", None)
    if not isinstance(pixels, Tensor):
        raise TypeError("final model output must expose a tensor RGB prediction")
    return pixels
