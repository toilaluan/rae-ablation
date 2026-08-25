from __future__ import annotations

import copy

import pytest
import torch

from cascade_dit.checkpoint import EMA
from cascade_dit.config import (
    DDTAttentionConfig,
    DDTPathConfig,
    ExperimentConfig,
    ModelConfig,
    RepresentationSourceConfig,
    RepresentationStageConfig,
)
from cascade_dit.curriculum import configure_trainability, stage_at_epoch
from cascade_dit.model import DDT
from cascade_dit.optim import build_optimizer, parameter_count
from cascade_dit.rng import RNGStreams
from cascade_dit.training import train_one_step


def _config(levels: int) -> ExperimentConfig:
    config = ExperimentConfig()
    dino = [12, 8, 4, 2][: levels - 1]
    config.representations.stages = [
        RepresentationStageConfig(
            f"dino{layer}", 1, RepresentationSourceConfig("dino", layer)
        )
        for layer in dino
    ] + [RepresentationStageConfig("rgb", 1, RepresentationSourceConfig("rgb"))]
    config.model = ModelConfig(
        encoder=DDTPathConfig(16, 1, 1),
        decoder=DDTPathConfig(32, 1, 2),
        attention=DDTAttentionConfig(rope=False),
    )
    config.model.conditioning.num_classes = 4
    config.optimizer.global_batch_size = 1
    config.validate()
    return config


@pytest.mark.parametrize("levels", [3, 4])
def test_tiny_dynamic_curriculum_and_matched_baseline(levels: int) -> None:
    config = _config(levels)
    model = DDT(config.model, levels)
    configure_trainability(model)
    optimizer = build_optimizer(model, config.optimizer)
    optimizer_identity = id(optimizer)
    ema = EMA(model)
    streams = RNGStreams.create(8)
    targets = torch.randn(1, 64, 768)
    labels = torch.tensor([2])
    for epoch in range(config.total_epochs):
        position = stage_at_epoch(epoch, config)
        weights = torch.zeros(1, levels)
        weights[:, position.index] = 1
        result = train_one_step(
            model, optimizer, targets, labels, weights, streams, config
        )
        assert torch.isfinite(result.loss.total)
        ema.update(model)
    assert id(optimizer) == optimizer_identity

    baseline = DDT(config.model, levels)
    configure_trainability(baseline)
    assert parameter_count(model) == parameter_count(baseline)
    baseline_weights = torch.zeros(1, levels)
    baseline_weights[:, -1] = 1
    baseline_result = train_one_step(
        baseline,
        build_optimizer(baseline, config.optimizer),
        targets,
        labels,
        baseline_weights,
        RNGStreams.create(8),
        config,
    )
    assert torch.isfinite(baseline_result.loss.total)


def test_source_model_copy_stays_independent() -> None:
    config = _config(3)
    first = DDT(config.model, 3)
    second = copy.deepcopy(first)
    with torch.no_grad():
        next(second.parameters()).add_(1)
    assert not torch.equal(next(first.parameters()), next(second.parameters()))
