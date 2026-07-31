from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ExperimentConfig, SIGRegConfig
from .rng import RNGStreams


def sample_timesteps(
    batch_size: int,
    generator: torch.Generator,
    device: torch.device | str,
    *,
    loc: float = 0.0,
    scale: float = 1.0,
) -> Tensor:
    return (loc + scale * torch.randn(batch_size, generator=generator, device=device)).sigmoid()


def _expand(timestep: Tensor, like: Tensor) -> Tensor:
    result = timestep.float()
    while result.ndim < like.ndim:
        result = result.unsqueeze(-1)
    return result


def noisy_interpolate(clean: Tensor, noise: Tensor, timestep: Tensor) -> Tensor:
    if clean.shape != noise.shape:
        raise ValueError("clean data and noise shapes must match")
    value = _expand(timestep, clean)
    return value * clean.float() + (1 - value) * noise.float()


def x_to_velocity(
    noisy: Tensor, prediction: Tensor, timestep: Tensor, floor: float = 0.05
) -> Tensor:
    denominator = (1 - timestep.float()).clamp_min(floor)
    while denominator.ndim < noisy.ndim:
        denominator = denominator.unsqueeze(-1)
    return (prediction.float() - noisy.float()) / denominator


def clean_prediction_velocity_loss(
    clean: Tensor,
    noise: Tensor,
    timestep: Tensor,
    prediction: Tensor,
) -> Tensor:
    noisy = noisy_interpolate(clean, noise, timestep)
    predicted_velocity = x_to_velocity(noisy, prediction, timestep)
    target_velocity = x_to_velocity(noisy, clean, timestep)
    return F.mse_loss(predicted_velocity, target_velocity)


def shift_timestep(timestep: Tensor, alpha: float | Tensor) -> Tensor:
    return alpha * timestep / (1 + (alpha - 1) * timestep)


def denoising_step_bounds(interval: Sequence[float], steps: int) -> tuple[int, int]:
    if steps < 1 or len(interval) != 2:
        raise ValueError("a positive step count and [start, end] interval are required")
    start, end = interval
    if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= 1:
        raise ValueError("denoising interval must satisfy 0 <= start < end <= 1")
    start_step = math.floor(start * steps + 0.5)
    end_step = math.floor(end * steps + 0.5)
    if start_step >= end_step:
        raise ValueError("denoising interval collapses to zero steps after rounding")
    return start_step, end_step


def local_denoising_schedule(
    interval: Sequence[float], steps: int, *, device: torch.device | str
) -> Tensor:
    start, end = denoising_step_bounds(interval, steps)
    edges = torch.arange(steps + 1, device=device, dtype=torch.float32)
    return ((edges - start) / (end - start)).clamp(0, 1)


def dual_timestep_schedule(
    batch_size: int,
    tokens: int,
    streams: RNGStreams,
    device: torch.device,
    *,
    probability: float,
    loc: float,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    first = sample_timesteps(batch_size, streams.timesteps, device, loc=loc, scale=scale)
    second = sample_timesteps(batch_size, streams.dual_timesteps, device, loc=loc, scale=scale)
    mask = (
        torch.rand((batch_size, tokens), device=device, generator=streams.token_masks) < probability
    )
    mixed = torch.where(mask, second[:, None], first[:, None])
    cleaner = torch.maximum(first, second)
    return mixed, cleaner, mask


def token_times_to_latents(timestep: Tensor, grid_size: int, patch_size: int) -> Tensor:
    if timestep.shape[1] != grid_size**2:
        raise ValueError("token timestep count does not match the model grid")
    return (
        timestep.reshape(timestep.shape[0], grid_size, grid_size)
        .repeat_interleave(patch_size, dim=1)
        .repeat_interleave(patch_size, dim=2)[:, None]
    )


def alignment_loss(
    projected: tuple[Tensor, ...], target: tuple[Tensor, ...]
) -> tuple[Tensor, tuple[Tensor, ...]]:
    if not projected or len(projected) != len(target):
        raise ValueError("alignment requires matching non-empty feature tuples")
    per_layer = tuple(
        -F.cosine_similarity(left.float(), right.float(), dim=-1).mean()
        for left, right in zip(projected, target, strict=True)
    )
    return torch.stack(per_layer).mean(), per_layer


class SIGReg(nn.Module):
    """Random-projection Epps-Pulley regularizer for N(0, I) features."""

    def __init__(self, config: SIGRegConfig) -> None:
        super().__init__()
        self.config = config
        knots = torch.linspace(0, config.max_frequency, config.knots, dtype=torch.float32)
        spacing = config.max_frequency / max(config.knots - 1, 1)
        weights = torch.full_like(knots, 2 * spacing)
        if config.knots > 1:
            weights[[0, -1]] = spacing
        weights *= torch.exp(-0.5 * knots.square())
        self.register_buffer("knots", knots, persistent=False)
        self.register_buffer("weights", weights, persistent=False)

    def forward(
        self, groups: tuple[Tensor, ...], generator: torch.Generator
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        if not groups:
            raise ValueError("SIGReg requires at least one feature group")
        dimension = groups[0].shape[-1]
        if any(group.ndim != 3 or group.shape[-1] != dimension for group in groups):
            raise ValueError("SIGReg features must have matching [B, N, D] shapes")
        if any(group.shape[0] < 2 for group in groups):
            raise ValueError("SIGReg requires a local batch size of at least two")
        directions = torch.randn(
            dimension,
            self.config.projections,
            device=groups[0].device,
            dtype=torch.float32,
            generator=generator,
        )
        directions = F.normalize(directions, dim=0)
        losses = [group.new_zeros((), dtype=torch.float32) for group in groups]
        for start in range(0, self.config.projections, self.config.projection_chunk_size):
            direction = directions[:, start : start + self.config.projection_chunk_size]
            fraction = direction.shape[1] / self.config.projections
            for index, group in enumerate(groups):
                projected = torch.einsum("bnd,dm->nbm", group.float(), direction)
                values = projected[..., None] * self.knots
                empirical_real = values.cos().mean(dim=1)
                empirical_imag = values.sin().mean(dim=1)
                target = torch.exp(-0.5 * self.knots.square())
                statistic = (
                    ((empirical_real - target).square() + empirical_imag.square())
                    .mul(self.weights)
                    .sum(dim=-1)
                    .mul(group.shape[0])
                    .mean()
                )
                losses[index] = losses[index] + fraction * statistic
        per_group = tuple(losses)
        return torch.stack(per_group).mean(), per_group


@dataclass(frozen=True)
class SelfFlowLoss:
    total: Tensor
    mixed_velocity: Tensor
    cleaner_velocity: Tensor
    alignment: Tensor
    sigreg: Tensor
    alignment_layers: tuple[Tensor, ...]
    sigreg_groups: tuple[Tensor, ...]
    mixed_timestep_mean: Tensor
    cleaner_timestep_mean: Tensor
    feature_mean_norms: tuple[Tensor, ...]
    feature_std_means: tuple[Tensor, ...]
    feature_covariance_offdiag_rms: tuple[Tensor, ...]


@torch.no_grad()
def feature_diagnostics(
    groups: tuple[Tensor, ...],
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    mean_norms: list[Tensor] = []
    std_means: list[Tensor] = []
    offdiag_rms: list[Tensor] = []
    for group in groups:
        flattened = group.detach().float().reshape(-1, group.shape[-1])
        mean = flattened.mean(dim=0)
        centered = flattened - mean
        covariance = centered.T @ centered / flattened.shape[0]
        diagonal = covariance.diagonal()
        count = covariance.numel() - diagonal.numel()
        mean_norms.append(mean.norm())
        std_means.append(centered.square().mean(dim=0).sqrt().mean())
        offdiag_rms.append(
            ((covariance.square().sum() - diagonal.square().sum()) / max(count, 1))
            .clamp_min(0)
            .sqrt()
        )
    return tuple(mean_norms), tuple(std_means), tuple(offdiag_rms)


def self_flow_loss(
    model: nn.Module,
    clean: Tensor,
    labels: Tensor,
    streams: RNGStreams,
    config: ExperimentConfig,
    *,
    ema_model: nn.Module | None = None,
    collect_diagnostics: bool = False,
) -> SelfFlowLoss:
    objective = config.objective
    device = clean.device
    batch = clean.shape[0]
    noise = torch.randn(clean.shape, device=device, dtype=torch.float32, generator=streams.noise)
    layers = tuple(objective.align_layers)
    zero = clean.new_zeros((), dtype=torch.float32)

    if objective.uses_dual_timestep:
        mixed_model_time, cleaner_time, _mask = dual_timestep_schedule(
            batch,
            config.model.grid_size**2,
            streams,
            device,
            probability=objective.mixed_token_probability,
            loc=config.train.timestep_loc,
            scale=config.train.timestep_scale,
        )
        mixed_time = token_times_to_latents(
            mixed_model_time, config.model.grid_size, config.model.patch_size
        )
    else:
        homogeneous = sample_timesteps(
            batch,
            streams.timesteps,
            device,
            loc=config.train.timestep_loc,
            scale=config.train.timestep_scale,
        )
        mixed_model_time = homogeneous
        mixed_time = homogeneous
        cleaner_time = homogeneous

    mixed_noisy = noisy_interpolate(clean, noise, mixed_time)
    mixed_output = model(
        mixed_noisy,
        mixed_model_time,
        labels,
        capture_layers=layers,
        project_features=objective.uses_alignment,
    )
    if mixed_output.x0 is None:
        raise RuntimeError("mixed branch did not produce x0")
    mixed_velocity = clean_prediction_velocity_loss(clean, noise, mixed_time, mixed_output.x0)
    cleaner_velocity = zero
    align = zero
    sigreg = zero
    per_alignment: tuple[Tensor, ...] = ()
    per_sigreg: tuple[Tensor, ...] = ()
    diagnostic_groups: tuple[Tensor, ...] = ()

    if objective.uses_alignment:
        cleaner_noisy = noisy_interpolate(clean, noise, cleaner_time)
        if objective.kind == "ema":
            if ema_model is None:
                raise ValueError("EMA objective requires an EMA model")
            with torch.no_grad():
                cleaner_output = ema_model(
                    cleaner_noisy,
                    cleaner_time,
                    labels,
                    capture_layers=layers,
                    stop_at_layer=layers[-1],
                )
        else:
            stop = layers[-1] if objective.sigreg.clean_forward == "truncated" else None
            cleaner_output = model(
                cleaner_noisy,
                cleaner_time,
                labels,
                capture_layers=layers,
                stop_at_layer=stop,
            )
            if stop is None:
                if cleaner_output.x0 is None:
                    raise RuntimeError("full cleaner branch did not produce x0")
                cleaner_velocity = clean_prediction_velocity_loss(
                    clean, noise, cleaner_time, cleaner_output.x0
                )
        align, per_alignment = alignment_loss(mixed_output.features, cleaner_output.features)
        diagnostic_groups = mixed_output.features + cleaner_output.features
        if objective.kind == "sigreg":
            regularizer = SIGReg(objective.sigreg).to(device)
            sigreg, per_sigreg = regularizer(diagnostic_groups, streams.sigreg)

    flow = mixed_velocity
    if objective.kind == "sigreg" and objective.sigreg.clean_forward == "full":
        flow = 0.5 * (mixed_velocity + cleaner_velocity)
    total = flow + objective.alignment_weight * align
    if objective.kind == "sigreg":
        total = total + objective.sigreg.weight * sigreg
    diagnostics = (
        feature_diagnostics(diagnostic_groups)
        if collect_diagnostics and diagnostic_groups
        else ((), (), ())
    )
    return SelfFlowLoss(
        total,
        mixed_velocity,
        cleaner_velocity,
        align,
        sigreg,
        per_alignment,
        per_sigreg,
        mixed_model_time.float().mean(),
        cleaner_time.float().mean(),
        *diagnostics,
    )
