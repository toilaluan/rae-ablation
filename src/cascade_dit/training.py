from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import ExperimentConfig
from .data import PreparedJointBatch
from .model import DDT, DDTOutput, PixelFeatureBundle
from .objective import (
    GaussianFlowPath,
    JointLoss,
    flow_matching_loss,
    gaussian_flow_path,
    sample_timesteps,
)
from .rng import RNGStreams


@dataclass(frozen=True)
class TrainingStep:
    loss: JointLoss
    output: DDTOutput
    noisy_tokens: Tensor
    timesteps: Tensor


@dataclass(frozen=True)
class PreparedTrainingPath:
    path: GaussianFlowPath
    force_class_drop: Tensor


class ScheduledTrainingModel(nn.Module):
    """DDP-visible execution modes for sparse and repeated pixel updates."""

    def __init__(self, model: DDT) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        noisy_tokens: Tensor | None = None,
        timestep: Tensor | None = None,
        labels: Tensor | None = None,
        *,
        mode: str,
        force_class_drop: Tensor | None = None,
        pixel_features: PixelFeatureBundle | None = None,
        detach_flow: bool = False,
    ) -> DDTOutput | tuple[DDTOutput, PixelFeatureBundle] | Tensor:
        if mode == "decoder":
            if pixel_features is None:
                raise ValueError("decoder-only execution requires cached pixel features")
            return self.model.decode_pixel_features(pixel_features)
        if mode not in {"semantic", "joint"}:
            raise ValueError("training execution mode must be semantic, joint, or decoder")
        if noisy_tokens is None or timestep is None or labels is None:
            raise ValueError("semantic and joint execution require flow inputs")
        semantic, features = self.model.forward_semantic(
            noisy_tokens,
            timestep,
            labels,
            capture_pixel_features=mode == "joint",
            force_class_drop=force_class_drop,
        )
        if mode == "semantic":
            return semantic
        assert features is not None
        pixels = self.model.decode_pixel_features(features, detach_flow=detach_flow)
        return (
            DDTOutput(semantic.full_x0, semantic.base_x0, pixels),
            features.detached(),
        )


def prepare_training_path(
    batch: PreparedJointBatch,
    rng: RNGStreams,
    config: ExperimentConfig,
    *,
    timestep_shift: float | None = None,
) -> PreparedTrainingPath:
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
    return PreparedTrainingPath(path, drop)


def compute_training_loss(
    model: nn.Module,
    batch: PreparedJointBatch,
    rng: RNGStreams,
    config: ExperimentConfig,
    lpips_model: nn.Module,
    *,
    timestep_shift: float | None = None,
) -> TrainingStep:
    prepared = prepare_training_path(
        batch,
        rng,
        config,
        timestep_shift=timestep_shift,
    )
    path = prepared.path
    output = model(
        path.noisy,
        path.timestep,
        batch.labels,
        decode_pixels=True,
        force_class_drop=prepared.force_class_drop,
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
    return TrainingStep(loss, output, path.noisy, path.timestep)
