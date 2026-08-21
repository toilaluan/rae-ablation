from __future__ import annotations

import math
from itertools import pairwise
from typing import Literal

import torch
from torch import Tensor, nn

from .objective import SEMANTIC_STAGE, TARGET_STAGE, shift_timestep, x_to_velocity

Solver = Literal["euler", "heun"]


def class_balanced_labels(sample_count: int, num_classes: int) -> Tensor:
    if sample_count < 1 or num_classes < 1 or sample_count % num_classes:
        raise ValueError("sample count must be positive and class balanced")
    return torch.arange(sample_count, dtype=torch.int64).remainder(num_classes)


def _model_predictions(
    model: nn.Module,
    x: Tensor,
    timestep: Tensor,
    stage: Tensor,
    labels: Tensor,
) -> tuple[Tensor, Tensor]:
    owners = (model, getattr(model, "module", None))
    for owner in owners:
        method = getattr(owner, "forward_predictions", None)
        if callable(method):
            output = method(x, timestep, stage, labels)
            return output.full_prediction, output.base_prediction
    output = model(x, timestep, stage, labels)
    return output.full_prediction, output.base_prediction


def _model_full_prediction(
    model: nn.Module,
    x: Tensor,
    timestep: Tensor,
    stage: Tensor,
    labels: Tensor,
) -> Tensor:
    owners = (model, getattr(model, "module", None))
    for owner in owners:
        method = getattr(owner, "forward_full_prediction", None)
        if callable(method):
            return method(x, timestep, stage, labels)
    for owner in owners:
        method = getattr(owner, "forward_predictions", None)
        if callable(method):
            return method(x, timestep, stage, labels).full_prediction
    return model(x, timestep, stage, labels).full_prediction


def _guided_x(
    full: Tensor,
    base: Tensor,
    timestep: Tensor,
    scale: float,
    interval: tuple[float, float],
) -> Tensor:
    active = ((timestep >= interval[0]) & (timestep <= interval[1])).reshape(
        -1, *((1,) * (full.ndim - 1))
    )
    guided = base + scale * (full - base)
    return torch.where(active, guided, full)


def _leg_schedule(
    stage: int,
    steps: int,
    waypoint_time: float,
    timestep_shift: float,
    device: torch.device,
) -> Tensor:
    local = shift_timestep(
        torch.linspace(1, 0, steps + 1, device=device), timestep_shift
    )
    if stage == SEMANTIC_STAGE:
        return waypoint_time + (1 - waypoint_time) * local
    if stage == TARGET_STAGE:
        return waypoint_time * local
    raise ValueError("unknown waypoint stage")


@torch.inference_mode()
def ode_sample(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    *,
    semantic_steps: int = 10,
    target_steps: int = 40,
    waypoint_time: float = 0.5,
    timestep_shift: float = 1.0,
    ig_scale: float = 1.78,
    ig_interval: tuple[float, float] = (0.1, 1.0),
    cfg_scale: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    solver: Solver = "euler",
    num_classes: int | None = None,
    velocity_denominator_floor: float = 0.05,
) -> Tensor:
    """Integrate noise through a noisy semantic waypoint to target tokens."""
    if semantic_steps < 1 or target_steps < 1 or solver not in {"euler", "heun"}:
        raise ValueError("sampling requires positive per-stage steps and euler or heun")
    if not 0 < waypoint_time < 1:
        raise ValueError("waypoint time must be in (0,1)")
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
    null_labels = torch.full_like(labels, num_classes)
    total_steps = semantic_steps + target_steps
    completed_steps = 0

    def predict_x(x: Tensor, timestep: Tensor, stage_value: int) -> Tensor:
        if stage_value == SEMANTIC_STAGE:
            model_t = timestep.expand(x.shape[0])
            model_stage = torch.full(
                (x.shape[0],),
                SEMANTIC_STAGE,
                device=x.device,
                dtype=torch.long,
            )
            full, base = _model_predictions(
                model,
                x.to(noise.dtype),
                model_t,
                model_stage,
                labels,
            )
            return _guided_x(
                full.float(), base.float(), model_t, ig_scale, ig_interval
            )

        use_cfg = cfg_scale != 1.0
        copies = 2 if use_cfg else 1
        model_x = torch.cat((x, x)).to(noise.dtype) if use_cfg else x.to(noise.dtype)
        model_t = timestep.expand(copies * x.shape[0])
        model_stage = torch.full(
            (copies * x.shape[0],),
            TARGET_STAGE,
            device=x.device,
            dtype=torch.long,
        )
        model_labels = torch.cat((labels, null_labels)) if use_cfg else labels
        full = _model_full_prediction(
            model, model_x, model_t, model_stage, model_labels
        ).float()
        if not use_cfg:
            return full
        full_c, full_u = full.chunk(2)
        t = model_t[: x.shape[0]]
        active = ((t >= cfg_interval[0]) & (t <= cfg_interval[1])).reshape(
            -1, *((1,) * (full_c.ndim - 1))
        )
        guided = full_u + cfg_scale * (full_c - full_u)
        return torch.where(active, guided, full_c)

    segments = (
        (
            SEMANTIC_STAGE,
            _leg_schedule(
                SEMANTIC_STAGE,
                semantic_steps,
                waypoint_time,
                timestep_shift,
                noise.device,
            ),
        ),
        (
            TARGET_STAGE,
            _leg_schedule(
                TARGET_STAGE,
                target_steps,
                waypoint_time,
                timestep_shift,
                noise.device,
            ),
        ),
    )
    for stage_value, schedule in segments:
        for current, following in pairwise(schedule):
            prediction = predict_x(state, current, stage_value)
            velocity = x_to_velocity(
                state,
                prediction,
                current.expand(state.shape[0]),
                velocity_denominator_floor,
            )
            step_size = following - current
            proposed = state + step_size * velocity
            completed_steps += 1
            if solver == "heun" and completed_steps < total_steps:
                corrected_x = predict_x(proposed, following, stage_value)
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
