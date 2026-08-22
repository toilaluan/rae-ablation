from __future__ import annotations

import math
from itertools import pairwise
from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn

from .objective import shift_timestep, x0_to_velocity

Solver = Literal["euler", "heun"]


class GeneratedRepresentations(NamedTuple):
    semantic_tokens: Tensor | None
    image_latents: Tensor


def class_balanced_labels(sample_count: int, num_classes: int) -> Tensor:
    if sample_count < 1 or num_classes < 1 or sample_count % num_classes:
        raise ValueError("sample count must be positive and class balanced")
    return torch.arange(sample_count, dtype=torch.int64).remainder(num_classes)


def _owner(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


def _schedule(steps: int, timestep_shift: float, device: torch.device) -> Tensor:
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    return shift_timestep(
        torch.linspace(1, 0, steps + 1, device=device), timestep_shift
    )


def _validate_common(
    noise: Tensor,
    labels: Tensor,
    *,
    timestep_shift: float,
    solver: Solver,
) -> None:
    if labels.shape != (noise.shape[0],):
        raise ValueError("labels must have shape [B]")
    if solver not in {"euler", "heun"}:
        raise ValueError("sampling solver must be euler or heun")
    if not math.isfinite(timestep_shift) or timestep_shift <= 0:
        raise ValueError("timestep shift must be finite and positive")


def _integrate(
    state: Tensor,
    schedule: Tensor,
    predict_x0: object,
    solver: Solver,
    velocity_denominator_floor: float,
) -> Tensor:
    prediction_function = predict_x0
    for step_index, (current, following) in enumerate(pairwise(schedule), start=1):
        prediction = prediction_function(state, current)  # type: ignore[operator]
        velocity = x0_to_velocity(
            state,
            prediction,
            current.expand(state.shape[0]),
            velocity_denominator_floor,
        )
        step_size = following - current
        proposed = state + step_size * velocity
        if solver == "heun" and step_index < schedule.numel() - 1:
            corrected_prediction = prediction_function(  # type: ignore[operator]
                proposed, following
            )
            corrected_velocity = x0_to_velocity(
                proposed,
                corrected_prediction,
                following.expand(state.shape[0]),
                velocity_denominator_floor,
            )
            state = state + 0.5 * step_size * (velocity + corrected_velocity)
        else:
            state = proposed
    return state


@torch.inference_mode()
def sample_semantic_tokens(
    model: nn.Module,
    semantic_noise: Tensor,
    labels: Tensor,
    *,
    steps: int = 10,
    timestep_shift: float = 1.0,
    semantic_guidance_scale: float = 1.78,
    semantic_guidance_interval: tuple[float, float] = (0.1, 1.0),
    solver: Solver = "euler",
    velocity_denominator_floor: float = 0.05,
) -> Tensor:
    _validate_common(
        semantic_noise, labels, timestep_shift=timestep_shift, solver=solver
    )
    if not math.isfinite(semantic_guidance_scale):
        raise ValueError("semantic guidance scale must be finite")
    if not 0 <= semantic_guidance_interval[0] < semantic_guidance_interval[1] <= 1:
        raise ValueError("semantic guidance interval must satisfy 0 <= start < end <= 1")
    method = getattr(_owner(model), "forward_semantic_predictions", None)
    if not callable(method):
        raise TypeError("model does not provide semantic prediction")

    def predict(state: Tensor, timestep: Tensor) -> Tensor:
        model_timestep = timestep.expand(state.shape[0])
        predictions = method(state.to(semantic_noise.dtype), model_timestep, labels)
        full, base = predictions.full.float(), predictions.base.float()
        active = (
            (model_timestep >= semantic_guidance_interval[0])
            & (model_timestep <= semantic_guidance_interval[1])
        ).reshape(-1, 1, 1, 1)
        guided = base + semantic_guidance_scale * (full - base)
        return torch.where(active, guided, full)

    return _integrate(
        semantic_noise.float(),
        _schedule(steps, timestep_shift, semantic_noise.device),
        predict,
        solver,
        velocity_denominator_floor,
    )


@torch.inference_mode()
def sample_image_latents(
    model: nn.Module,
    image_noise: Tensor,
    labels: Tensor,
    semantic_tokens: Tensor | None = None,
    *,
    steps: int = 40,
    timestep_shift: float = 1.0,
    class_guidance_scale: float = 1.0,
    class_guidance_interval: tuple[float, float] = (0.0, 1.0),
    solver: Solver = "euler",
    num_classes: int | None = None,
    velocity_denominator_floor: float = 0.05,
) -> Tensor:
    _validate_common(image_noise, labels, timestep_shift=timestep_shift, solver=solver)
    if not math.isfinite(class_guidance_scale):
        raise ValueError("class guidance scale must be finite")
    if not 0 <= class_guidance_interval[0] < class_guidance_interval[1] <= 1:
        raise ValueError("class guidance interval must satisfy 0 <= start < end <= 1")
    owner = _owner(model)
    if num_classes is None:
        num_classes = getattr(getattr(owner, "config", None), "num_classes", None)
    if not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError("num_classes is required")
    method = getattr(owner, "forward_image_prediction", None)
    if not callable(method):
        raise TypeError("model does not provide image-latent prediction")
    if semantic_tokens is not None and semantic_tokens.shape[0] != image_noise.shape[0]:
        raise ValueError("semantic tokens and image noise batch sizes must match")
    null_labels = torch.full_like(labels, num_classes)

    def predict(state: Tensor, timestep: Tensor) -> Tensor:
        use_guidance = class_guidance_scale != 1.0
        copies = 2 if use_guidance else 1
        model_state = torch.cat((state, state)) if use_guidance else state
        model_timestep = timestep.expand(copies * state.shape[0])
        model_labels = torch.cat((labels, null_labels)) if use_guidance else labels
        if semantic_tokens is None:
            prediction = method(
                model_state.to(image_noise.dtype), model_timestep, model_labels
            ).float()
        else:
            condition = (
                torch.cat((semantic_tokens, semantic_tokens))
                if use_guidance
                else semantic_tokens
            )
            prediction = method(
                model_state.to(image_noise.dtype),
                model_timestep,
                model_labels,
                condition.to(image_noise.dtype),
            ).float()
        if not use_guidance:
            return prediction
        conditional, unconditional = prediction.chunk(2)
        active = (
            (model_timestep[: state.shape[0]] >= class_guidance_interval[0])
            & (model_timestep[: state.shape[0]] <= class_guidance_interval[1])
        ).reshape(-1, 1, 1, 1)
        guided = unconditional + class_guidance_scale * (
            conditional - unconditional
        )
        return torch.where(active, guided, conditional)

    return _integrate(
        image_noise.float(),
        _schedule(steps, timestep_shift, image_noise.device),
        predict,
        solver,
        velocity_denominator_floor,
    )


@torch.inference_mode()
def sample_cascade(
    model: nn.Module,
    semantic_noise: Tensor,
    image_noise: Tensor,
    labels: Tensor,
    *,
    semantic_steps: int = 10,
    image_steps: int = 40,
    timestep_shift: float = 1.0,
    semantic_guidance_scale: float = 1.78,
    semantic_guidance_interval: tuple[float, float] = (0.1, 1.0),
    class_guidance_scale: float = 1.0,
    class_guidance_interval: tuple[float, float] = (0.0, 1.0),
    solver: Solver = "euler",
    num_classes: int | None = None,
    velocity_denominator_floor: float = 0.05,
) -> GeneratedRepresentations:
    if semantic_noise.data_ptr() == image_noise.data_ptr():
        raise ValueError("semantic and image-latent sampling require independent noise tensors")
    semantic_tokens = sample_semantic_tokens(
        model,
        semantic_noise,
        labels,
        steps=semantic_steps,
        timestep_shift=timestep_shift,
        semantic_guidance_scale=semantic_guidance_scale,
        semantic_guidance_interval=semantic_guidance_interval,
        solver=solver,
        velocity_denominator_floor=velocity_denominator_floor,
    )
    image_latents = sample_image_latents(
        model,
        image_noise,
        labels,
        semantic_tokens,
        steps=image_steps,
        timestep_shift=timestep_shift,
        class_guidance_scale=class_guidance_scale,
        class_guidance_interval=class_guidance_interval,
        solver=solver,
        num_classes=num_classes,
        velocity_denominator_floor=velocity_denominator_floor,
    )
    return GeneratedRepresentations(semantic_tokens, image_latents)


@torch.inference_mode()
def sample_latent(
    model: nn.Module,
    image_noise: Tensor,
    labels: Tensor,
    **kwargs: object,
) -> GeneratedRepresentations:
    image_latents = sample_image_latents(
        model, image_noise, labels, None, **kwargs  # type: ignore[arg-type]
    )
    return GeneratedRepresentations(None, image_latents)
