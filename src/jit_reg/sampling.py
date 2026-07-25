from __future__ import annotations

import math
from itertools import pairwise
from typing import Literal, NamedTuple, overload

import torch
from torch import Tensor, nn

from .objective import shift_timestep

Solver = Literal["euler", "heun"]


class ODESample(NamedTuple):
    image: Tensor
    reg: Tensor


EulerSample = ODESample


def class_balanced_labels(sample_count: int, num_classes: int) -> Tensor:
    if sample_count < 1 or num_classes < 1:
        raise ValueError("sample count and number of classes must be positive")
    if sample_count % num_classes:
        raise ValueError("sample count must be class balanced (divisible by the number of classes)")
    return torch.arange(sample_count, dtype=torch.int64).remainder(num_classes)


def _guided(conditional: Tensor, unconditional: Tensor, scale: float) -> Tensor:
    return unconditional + scale * (conditional - unconditional)


@overload
def ode_sample(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    *,
    reg_noise: None = None,
    steps: int = 50,
    guidance_scale: float = 1.78,
    timestep_shift: float = 1.0,
    reg_timestep_ratio: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 0.9),
    solver: Solver = "euler",
    num_classes: int | None = None,
) -> Tensor: ...


@overload
def ode_sample(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    *,
    reg_noise: Tensor,
    steps: int = 50,
    guidance_scale: float = 1.78,
    timestep_shift: float = 1.0,
    reg_timestep_ratio: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 0.9),
    solver: Solver = "euler",
    num_classes: int | None = None,
) -> ODESample: ...


@torch.inference_mode()
def ode_sample(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    *,
    reg_noise: Tensor | None = None,
    steps: int = 50,
    guidance_scale: float = 1.78,
    timestep_shift: float = 1.0,
    reg_timestep_ratio: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 0.9),
    solver: Solver = "euler",
    num_classes: int | None = None,
) -> Tensor | ODESample:
    """Integrate clean predictions from noise to data with Euler or Heun."""
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    if not math.isfinite(timestep_shift) or timestep_shift <= 0:
        raise ValueError("timestep shift must be finite and positive")
    if not math.isfinite(reg_timestep_ratio) or reg_timestep_ratio <= 0:
        raise ValueError("REG timestep ratio must be finite and positive")
    if not math.isfinite(guidance_scale):
        raise ValueError("guidance scale must be finite")
    cfg_start, cfg_end = cfg_interval
    if not (0 <= cfg_start <= cfg_end <= 1):
        raise ValueError("CFG interval must satisfy 0 <= start <= end <= 1")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be euler or heun")
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
    image_schedule = shift_timestep(u, timestep_shift)
    reg_schedule = shift_timestep(image_schedule, reg_timestep_ratio)

    def predict_clean(
        image_state: Tensor,
        current_reg_state: Tensor | None,
        image_timestep_value: Tensor,
        reg_timestep_value: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        use_cfg = cfg_start <= float(image_timestep_value) <= cfg_end
        if use_cfg:
            model_image = torch.cat((image_state, image_state)).to(noise.dtype)
            model_labels = combined_labels
            model_timestep = image_timestep_value.expand(2 * noise.shape[0])
            model_reg_timestep = (
                reg_timestep_value.expand(2 * noise.shape[0])
                if current_reg_state is not None
                else None
            )
            model_reg = (
                torch.cat((current_reg_state, current_reg_state)).to(reg_noise.dtype)
                if current_reg_state is not None and reg_noise is not None
                else None
            )
        else:
            model_image = image_state.to(noise.dtype)
            model_labels = labels
            model_timestep = image_timestep_value.expand(noise.shape[0])
            model_reg_timestep = (
                reg_timestep_value.expand(noise.shape[0])
                if current_reg_state is not None
                else None
            )
            model_reg = (
                current_reg_state.to(reg_noise.dtype)
                if current_reg_state is not None and reg_noise is not None
                else None
            )
        model_kwargs = {"noisy_reg": model_reg, "return_repa": False}
        if model_reg_timestep is not None:
            model_kwargs["reg_timestep"] = model_reg_timestep
        output = model(model_image, model_timestep, model_labels, **model_kwargs)
        if use_cfg:
            conditional_image, unconditional_image = output.image.float().chunk(2)
            image_prediction = _guided(
                conditional_image, unconditional_image, guidance_scale
            )
        else:
            image_prediction = output.image.float()
        reg_prediction = None
        if current_reg_state is not None:
            if output.reg is None:
                raise RuntimeError("REG sampling requires a model REG-token prediction")
            if use_cfg:
                conditional_reg, unconditional_reg = output.reg.float().chunk(2)
                reg_prediction = _guided(
                    conditional_reg, unconditional_reg, guidance_scale
                )
            else:
                reg_prediction = output.reg.float()
        return image_prediction, reg_prediction

    image_pairs = pairwise(image_schedule)
    reg_pairs = pairwise(reg_schedule)
    for step_index, ((current, following), (reg_current, reg_following)) in enumerate(
        zip(image_pairs, reg_pairs, strict=True)
    ):
        image_prediction, reg_prediction = predict_clean(
            x, reg_state, current, reg_current
        )
        denominator = (1 - current).clamp_min(0.05)
        step_size = following - current
        image_velocity = (image_prediction - x) / denominator
        predicted_x = x + step_size * image_velocity

        predicted_reg = None
        reg_velocity = None
        if reg_state is not None:
            if reg_prediction is None:
                raise RuntimeError("REG sampling requires a model REG-token prediction")
            reg_denominator = (1 - reg_current).clamp_min(0.05)
            reg_step_size = reg_following - reg_current
            reg_velocity = (reg_prediction - reg_state) / reg_denominator
            predicted_reg = reg_state + reg_step_size * reg_velocity

        if solver == "heun" and step_index < steps - 1:
            corrected_image, corrected_reg = predict_clean(
                predicted_x, predicted_reg, following, reg_following
            )
            corrected_denominator = (1 - following).clamp_min(0.05)
            corrected_image_velocity = (
                corrected_image - predicted_x
            ) / corrected_denominator
            x = x + 0.5 * step_size * (image_velocity + corrected_image_velocity)
            if reg_state is not None:
                if predicted_reg is None or reg_velocity is None or corrected_reg is None:
                    raise RuntimeError(
                        "REG sampling requires a model REG-token prediction"
                    )
                corrected_reg_denominator = (1 - reg_following).clamp_min(0.05)
                corrected_reg_velocity = (
                    corrected_reg - predicted_reg
                ) / corrected_reg_denominator
                reg_state = reg_state + 0.5 * reg_step_size * (
                    reg_velocity + corrected_reg_velocity
                )
        else:
            x = predicted_x
            reg_state = predicted_reg
    if reg_state is None:
        return x
    return ODESample(x, reg_state)


def euler_sample(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    *,
    reg_noise: Tensor | None = None,
    steps: int = 50,
    guidance_scale: float = 1.78,
    timestep_shift: float = 1.0,
    reg_timestep_ratio: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 0.9),
    num_classes: int | None = None,
) -> Tensor | ODESample:
    """Compatibility wrapper for Euler integration."""
    return ode_sample(
        model,
        noise,
        labels,
        reg_noise=reg_noise,
        steps=steps,
        guidance_scale=guidance_scale,
        timestep_shift=timestep_shift,
        reg_timestep_ratio=reg_timestep_ratio,
        cfg_interval=cfg_interval,
        solver="euler",
        num_classes=num_classes,
    )


def heun_sample(
    model: nn.Module,
    noise: Tensor,
    labels: Tensor,
    *,
    reg_noise: Tensor | None = None,
    steps: int = 50,
    guidance_scale: float = 1.78,
    timestep_shift: float = 1.0,
    reg_timestep_ratio: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 0.9),
    num_classes: int | None = None,
) -> Tensor | ODESample:
    """Compatibility wrapper for Heun integration with a final Euler step."""
    return ode_sample(
        model,
        noise,
        labels,
        reg_noise=reg_noise,
        steps=steps,
        guidance_scale=guidance_scale,
        timestep_shift=timestep_shift,
        reg_timestep_ratio=reg_timestep_ratio,
        cfg_interval=cfg_interval,
        solver="heun",
        num_classes=num_classes,
    )
