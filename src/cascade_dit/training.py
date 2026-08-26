from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import ExperimentConfig
from .data import EndpointPhase, PreparedEndpointBatch
from .model import DDTOutput
from .objective import FlowLoss, flow_matching_loss, gaussian_flow_path, sample_timesteps
from .rng import RNGStreams


@dataclass(frozen=True)
class TrainingStep:
    loss: FlowLoss
    output: DDTOutput
    noisy_tokens: Tensor
    timesteps: Tensor


def compute_training_loss(
    model: nn.Module,
    batch: PreparedEndpointBatch,
    rng: RNGStreams,
    config: ExperimentConfig,
    *,
    phase: EndpointPhase,
    timestep_shift: float | None = None,
) -> TrainingStep:
    if phase not in {"semantic", "pixel"}:
        raise ValueError("phase must be semantic or pixel")
    active_shift = (
        config.training.semantic.timestep_shift
        if phase == "semantic"
        else config.training.pixel.timestep_shift
    )
    if timestep_shift is not None:
        active_shift = timestep_shift
    timesteps = sample_timesteps(
        batch.clean.shape[0],
        rng.timesteps,
        batch.clean.device,
        loc=config.objective.timestep_loc,
        scale=config.objective.timestep_scale,
        shift=active_shift,
    )
    noise = torch.randn(
        batch.clean.shape,
        generator=rng.noise,
        device=batch.clean.device,
        dtype=batch.clean.dtype,
    )
    path = gaussian_flow_path(batch.clean, noise, timesteps)
    drop = torch.rand(
        batch.labels.shape,
        generator=rng.label_dropout,
        device=batch.labels.device,
    ) < config.model.class_dropout_probability
    output = model(
        path.noisy,
        timesteps,
        batch.labels,
        batch.pixel_ratio,
        prediction_mode="mixed" if phase == "semantic" else "pixel",
        force_class_drop=drop,
    )
    if not isinstance(output, DDTOutput):
        raise TypeError("DDT training requires a DDTOutput")
    loss = flow_matching_loss(
        path,
        output,
        batch.pixel,
        batch.pixel_ratio,
        config.objective,
    )
    return TrainingStep(loss, output, path.noisy, timesteps)
