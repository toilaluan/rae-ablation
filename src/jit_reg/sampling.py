from __future__ import annotations

import math
from itertools import pairwise
from typing import Literal, NamedTuple, overload

import torch
from torch import Tensor, nn

from .objective import (
    denoising_step_bounds,
    local_denoising_schedule,
    shift_timestep,
)

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
    reg_guidance_scale: float = 1.78,
    timestep_shift: float = 1.0,
    reg_timestep_shift: float = 1.0,
    img_denoising_interval: tuple[float, float] = (0.0, 1.0),
    reg_denoising_interval: tuple[float, float] = (0.0, 1.0),
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
    reg_guidance_scale: float = 1.78,
    timestep_shift: float = 1.0,
    reg_timestep_shift: float = 1.0,
    img_denoising_interval: tuple[float, float] = (0.0, 1.0),
    reg_denoising_interval: tuple[float, float] = (0.0, 1.0),
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
    reg_guidance_scale: float = 1.78,
    timestep_shift: float = 1.0,
    reg_timestep_shift: float = 1.0,
    img_denoising_interval: tuple[float, float] = (0.0, 1.0),
    reg_denoising_interval: tuple[float, float] = (0.0, 1.0),
    cfg_interval: tuple[float, float] = (0.0, 0.9),
    solver: Solver = "euler",
    num_classes: int | None = None,
) -> Tensor | ODESample:
    """Integrate clean predictions from noise to data with Euler or Heun."""
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    if not math.isfinite(timestep_shift) or timestep_shift <= 0:
        raise ValueError("timestep shift must be finite and positive")
    if not math.isfinite(reg_timestep_shift) or reg_timestep_shift <= 0:
        raise ValueError("REG timestep shift must be finite and positive")
    if not math.isfinite(guidance_scale):
        raise ValueError("guidance scale must be finite")
    if not math.isfinite(reg_guidance_scale):
        raise ValueError("REG guidance scale must be finite")
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
    image_schedule = shift_timestep(
        local_denoising_schedule(
            img_denoising_interval, steps, device=noise.device
        ),
        timestep_shift,
    )
    reg_schedule = shift_timestep(
        local_denoising_schedule(
            reg_denoising_interval, steps, device=noise.device
        ),
        reg_timestep_shift,
    )
    image_start_step, image_end_step = denoising_step_bounds(
        img_denoising_interval,
        steps,
    )
    reg_start_step, reg_end_step = denoising_step_bounds(
        reg_denoising_interval,
        steps,
    )
    model_config = getattr(model, "config", None)
    model_features = getattr(model_config, "features", None)
    model_reg_config = getattr(model_features, "reg", None)
    reg_only_forward = getattr(model, "forward_reg", None)
    use_reg_only_forward = (
        reg_noise is not None
        and callable(reg_only_forward)
        and bool(
            getattr(
                model_reg_config,
                "independent_attention",
                False,
            )
        )
    )

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
            if output.reg_early is None:
                raise RuntimeError(
                    "REG sampling requires an early REG-token prediction"
                )
            if use_cfg:
                conditional_reg = output.reg.float().chunk(2)[0]
                early_reg = output.reg_early.float().chunk(2)[0]
            else:
                conditional_reg = output.reg.float()
                early_reg = output.reg_early.float()
            reg_prediction = _guided(
                conditional_reg,
                early_reg,
                reg_guidance_scale,
            )
        return image_prediction, reg_prediction

    def predict_reg_only(
        current_reg_state: Tensor,
        image_timestep_value: Tensor,
        reg_timestep_value: Tensor,
    ) -> Tensor:
        if not use_reg_only_forward or not callable(reg_only_forward):
            raise RuntimeError(
                "REG-only prediction requires an independent-attention model"
            )
        assert reg_noise is not None
        output = reg_only_forward(
            current_reg_state.to(reg_noise.dtype),
            image_timestep_value.expand(noise.shape[0]),
            labels,
            reg_timestep=reg_timestep_value.expand(
                noise.shape[0]
            ),
        )
        return _guided(
            output.reg.float(),
            output.reg_early.float(),
            reg_guidance_scale,
        )

    image_pairs = pairwise(image_schedule)
    reg_pairs = pairwise(reg_schedule)
    for step_index, ((current, following), (reg_current, reg_following)) in enumerate(
        zip(image_pairs, reg_pairs, strict=True)
    ):
        image_active = (
            image_start_step <= step_index < image_end_step
        )
        reg_active = (
            reg_state is not None
            and reg_start_step <= step_index < reg_end_step
        )
        if use_reg_only_forward and not image_active:
            if reg_active:
                assert reg_state is not None
                reg_prediction = predict_reg_only(
                    reg_state,
                    current,
                    reg_current,
                )
                reg_denominator = (1 - reg_current).clamp_min(0.05)
                reg_step_size = reg_following - reg_current
                reg_velocity = (
                    reg_prediction - reg_state
                ) / reg_denominator
                predicted_reg = (
                    reg_state + reg_step_size * reg_velocity
                )
                if solver == "heun" and step_index < steps - 1:
                    corrected_reg = predict_reg_only(
                        predicted_reg,
                        following,
                        reg_following,
                    )
                    corrected_denominator = (
                        1 - reg_following
                    ).clamp_min(0.05)
                    corrected_velocity = (
                        corrected_reg - predicted_reg
                    ) / corrected_denominator
                    reg_state = reg_state + 0.5 * reg_step_size * (
                        reg_velocity + corrected_velocity
                    )
                else:
                    reg_state = predicted_reg
            continue
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
    reg_guidance_scale: float = 1.78,
    timestep_shift: float = 1.0,
    reg_timestep_shift: float = 1.0,
    img_denoising_interval: tuple[float, float] = (0.0, 1.0),
    reg_denoising_interval: tuple[float, float] = (0.0, 1.0),
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
        reg_guidance_scale=reg_guidance_scale,
        timestep_shift=timestep_shift,
        reg_timestep_shift=reg_timestep_shift,
        img_denoising_interval=img_denoising_interval,
        reg_denoising_interval=reg_denoising_interval,
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
    reg_guidance_scale: float = 1.78,
    timestep_shift: float = 1.0,
    reg_timestep_shift: float = 1.0,
    img_denoising_interval: tuple[float, float] = (0.0, 1.0),
    reg_denoising_interval: tuple[float, float] = (0.0, 1.0),
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
        reg_guidance_scale=reg_guidance_scale,
        timestep_shift=timestep_shift,
        reg_timestep_shift=reg_timestep_shift,
        img_denoising_interval=img_denoising_interval,
        reg_denoising_interval=reg_denoising_interval,
        cfg_interval=cfg_interval,
        solver="heun",
        num_classes=num_classes,
    )
