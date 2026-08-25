from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import ExperimentConfig
from .objective import FlowLoss, clean_prediction_loss, gaussian_flow_path, sample_timesteps
from .rng import RNGStreams


@dataclass(frozen=True)
class TrainingStep:
    loss: FlowLoss
    clean_prediction: Tensor
    noisy_tokens: Tensor
    timesteps: Tensor


def compute_training_loss(
    model: nn.Module,
    clean_tokens: Tensor,
    labels: Tensor,
    representation_weights: Tensor,
    rng: RNGStreams,
    config: ExperimentConfig,
    *,
    source_prediction: Tensor | None = None,
    source_model: nn.Module | None = None,
) -> TrainingStep:
    timesteps = sample_timesteps(
        clean_tokens.shape[0],
        rng.timesteps,
        clean_tokens.device,
        loc=config.objective.timestep_loc,
        scale=config.objective.timestep_scale,
        shift=config.objective.timestep_shift,
    )
    noise = torch.randn(
        clean_tokens.shape,
        generator=rng.noise,
        device=clean_tokens.device,
        dtype=clean_tokens.dtype,
    )
    path = gaussian_flow_path(clean_tokens, noise, timesteps)
    drop = torch.rand(
        labels.shape, generator=rng.label_dropout, device=labels.device
    ) < config.model.class_dropout_probability
    if source_model is not None:
        with torch.no_grad():
            source_output = source_model(
                path.noisy,
                timesteps,
                labels,
                representation_weights,
                force_class_drop=drop,
            )
            source_prediction = (
                source_output if isinstance(source_output, Tensor) else source_output.prediction
            )
    output = model(
        path.noisy,
        timesteps,
        labels,
        representation_weights,
        force_class_drop=drop,
    )
    prediction = output if isinstance(output, Tensor) else output.prediction
    loss = clean_prediction_loss(
        path,
        prediction,
        config.objective,
        source_prediction=source_prediction,
        source_consistency_weight=config.curriculum.source_consistency_weight,
    )
    return TrainingStep(loss, prediction, path.noisy, timesteps)


def train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    clean_tokens: Tensor,
    labels: Tensor,
    representation_weights: Tensor,
    rng: RNGStreams,
    config: ExperimentConfig,
) -> TrainingStep:
    optimizer.zero_grad(set_to_none=True)
    result = compute_training_loss(
        model, clean_tokens, labels, representation_weights, rng, config
    )
    result.loss.total.backward()
    torch.nn.utils.clip_grad_norm_(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        config.optimizer.gradient_clip,
    )
    optimizer.step()
    return result
