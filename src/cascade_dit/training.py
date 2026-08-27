from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import ExperimentConfig
from .data import PreparedJointBatch
from .model import DDTOutput
from .objective import JointLoss, flow_matching_loss, gaussian_flow_path, sample_timesteps
from .rng import RNGStreams


@dataclass(frozen=True)
class TrainingStep:
    loss: JointLoss
    output: DDTOutput
    noisy_tokens: Tensor
    timesteps: Tensor


def compute_training_loss(
    model: nn.Module,
    batch: PreparedJointBatch,
    rng: RNGStreams,
    config: ExperimentConfig,
    lpips_model: nn.Module,
    *,
    timestep_shift: float | None = None,
) -> TrainingStep:
    active_shift = config.training.timestep_shift
    if timestep_shift is not None:
        active_shift = timestep_shift
    timesteps = sample_timesteps(
        batch.semantic.shape[0],
        rng.timesteps,
        batch.semantic.device,
        loc=config.objective.timestep_loc,
        scale=config.objective.timestep_scale,
        shift=active_shift,
    )
    noise = torch.randn(
        batch.semantic.shape,
        generator=rng.noise,
        device=batch.semantic.device,
        dtype=batch.semantic.dtype,
    )
    path = gaussian_flow_path(batch.semantic, noise, timesteps)
    drop = torch.rand(
        batch.labels.shape,
        generator=rng.label_dropout,
        device=batch.labels.device,
    ) < config.model.class_dropout_probability
    output = model(
        path.noisy,
        timesteps,
        batch.labels,
        decode_pixels=True,
        force_class_drop=drop,
    )
    if not isinstance(output, DDTOutput):
        raise TypeError("DDT training requires a DDTOutput")
    loss = flow_matching_loss(
        path,
        output,
        batch.pixels,
        lpips_model,
        config.objective,
    )
    return TrainingStep(loss, output, path.noisy, timesteps)
