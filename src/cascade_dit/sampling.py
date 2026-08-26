from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import torch
from torch import Tensor, nn

from .model import PredictionMode
from .objective import shift_timestep, x0_to_velocity


def class_balanced_labels(sample_count: int, num_classes: int) -> Tensor:
    if sample_count < 1 or num_classes < 1 or sample_count % num_classes:
        raise ValueError("sample count must be positive and divisible by num_classes")
    return torch.arange(sample_count, dtype=torch.int64) % num_classes


def _prediction(output: object) -> Tensor:
    value = getattr(output, "prediction", None)
    if not isinstance(value, Tensor):
        raise TypeError("model output must expose a tensor .prediction")
    return value


def _model_x0(
    model: nn.Module,
    values: Tensor,
    timestep: Tensor,
    labels: Tensor,
    pixel_ratio: Tensor,
    *,
    prediction_mode: PredictionMode,
    num_classes: int,
    guidance_scale: float,
    guidance_start: float,
    guidance_end: float,
) -> Tensor:
    progress = float(1 - timestep[0].item())
    active = guidance_scale != 1 and guidance_start <= progress <= guidance_end
    if not active:
        return _prediction(
            model(
                values,
                timestep,
                labels,
                pixel_ratio,
                prediction_mode=prediction_mode,
            )
        )
    combined_values = torch.cat((values, values))
    combined_time = torch.cat((timestep, timestep))
    unconditional = torch.full_like(labels, num_classes)
    combined_labels = torch.cat((labels, unconditional))
    combined_ratio = torch.cat((pixel_ratio, pixel_ratio))
    conditional, unconditional_prediction = _prediction(
        model(
            combined_values,
            combined_time,
            combined_labels,
            combined_ratio,
            prediction_mode=prediction_mode,
        )
    ).chunk(2)
    return unconditional_prediction + guidance_scale * (
        conditional - unconditional_prediction
    )


@torch.no_grad()
def sample_tokens(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    pixel_ratio: Tensor,
    *,
    prediction_mode: PredictionMode,
    steps: int = 40,
    solver: str = "euler",
    timestep_shift: float = 1.0,
    guidance_scale: float = 1.0,
    guidance_start: float = 0.0,
    guidance_end: float = 1.0,
    denominator_floor: float = 0.05,
    num_classes: int = 1000,
    callback: Callable[[int, Tensor], None] | None = None,
) -> Tensor:
    """Integrate one endpoint trajectory from t=1 to t=0."""
    if steps < 1 or solver not in {"euler", "heun"}:
        raise ValueError("sampling requires positive steps and the euler or heun solver")
    if prediction_mode not in {"semantic", "pixel"}:
        raise ValueError("sampling supports only semantic or pixel prediction mode")
    if not 0 <= guidance_start < guidance_end <= 1:
        raise ValueError("CFG interval must satisfy 0 <= start < end <= 1")
    if labels.shape != (noise.shape[0],):
        raise ValueError("sampling labels must have shape [B]")
    if pixel_ratio.shape != (noise.shape[0],):
        raise ValueError("sampling pixel_ratio must have shape [B]")
    base_grid = torch.linspace(1, 0, steps + 1, device=noise.device, dtype=torch.float32)
    grid = shift_timestep(base_grid, timestep_shift)
    values = noise

    def velocity(state: Tensor, scalar_time: Tensor) -> Tensor:
        time = scalar_time.expand(state.shape[0])
        clean = _model_x0(
            model,
            state,
            time,
            labels,
            pixel_ratio,
            prediction_mode=prediction_mode,
            num_classes=num_classes,
            guidance_scale=guidance_scale,
            guidance_start=guidance_start,
            guidance_end=guidance_end,
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


@torch.no_grad()
def sample_semantic_then_pixel(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    *,
    steps: int = 40,
    solver: str = "euler",
    timestep_shift: float = 1.0,
    guidance_scale: float = 1.0,
    guidance_start: float = 0.0,
    guidance_end: float = 1.0,
    denominator_floor: float = 0.05,
    num_classes: int = 1000,
) -> Tensor:
    """Sample semantic x0, then perform exactly one t=0 semantic-conditioned RGB decode."""
    ratio = torch.zeros(noise.shape[0], device=noise.device, dtype=torch.float32)
    semantic = sample_tokens(
        model,
        noise,
        labels,
        ratio,
        prediction_mode="semantic",
        steps=steps,
        solver=solver,
        timestep_shift=timestep_shift,
        guidance_scale=guidance_scale,
        guidance_start=guidance_start,
        guidance_end=guidance_end,
        denominator_floor=denominator_floor,
        num_classes=num_classes,
    )
    zero = torch.zeros(noise.shape[0], device=noise.device, dtype=torch.float32)
    return _model_x0(
        model,
        semantic,
        zero,
        labels,
        ratio,
        prediction_mode="pixel",
        num_classes=num_classes,
        guidance_scale=guidance_scale,
        guidance_start=guidance_start,
        guidance_end=guidance_end,
    )
