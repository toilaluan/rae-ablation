from __future__ import annotations

import copy

import torch

from cascade_dit.checkpoint import EMA
from cascade_dit.config import (
    CurriculumConfig,
    DetailHeadsConfig,
    ExperimentConfig,
    ModelConfig,
)
from cascade_dit.curriculum import configure_trainability, transition_to_stage
from cascade_dit.model import LightningDiT
from cascade_dit.optim import build_optimizer, parameter_count
from cascade_dit.rng import RNGStreams
from cascade_dit.training import train_one_step


def _config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.model = ModelConfig(
        hidden_size=16,
        depth=4,
        num_heads=1,
        mlp_ratio=1,
        num_classes=4,
        rope=False,
        class_dropout_probability=0,
    )
    config.detail_heads = DetailHeadsConfig(pixel_enabled=False)
    config.curriculum = CurriculumConfig(
        epochs=[1, 1, 1, 1],
        fully_train_blocks=[0, 3],
        lora_blocks=[1, 2],
        lora_rank=2,
        lora_alpha=2,
        lora_dropout=0,
    )
    return config


def test_tiny_four_stage_and_matched_baseline_runs() -> None:
    config = _config()
    model = LightningDiT(config.model, config.detail_heads)
    ema = EMA(model)
    projectors = {
        "dino12": torch.eye(768),
        "dino8": torch.eye(768).roll(1, 0),
        "dino4": torch.eye(768).roll(2, 0),
        "rgb": torch.eye(768).roll(3, 0),
    }
    model.set_stage("dino12", projectors["dino12"])
    configure_trainability(model, 0, config.curriculum)
    streams = RNGStreams.create(8)
    targets = torch.randn(1, 64, 768)
    labels = torch.tensor([2])
    for stage_index, stage in enumerate(config.curriculum.stages):
        if stage_index:
            assert transition_to_stage(
                model, ema, stage_index, projectors, config.curriculum
            )
        optimizer = build_optimizer(model, config.optimizer, transitioned=stage_index > 0)
        result = train_one_step(
            model, optimizer, targets @ projectors[stage], labels, streams, config, stage=stage
        )
        assert torch.isfinite(result.loss.total)
        ema.update(model)
    ema_model = ema.to_model(model)
    model.merge_lora()
    ema_model.merge_lora()
    model.fold_projector()
    ema_model.fold_projector()
    baseline = LightningDiT(config.model, config.detail_heads)
    configure_trainability(baseline, 3, config.curriculum, pixel_baseline=True)
    assert parameter_count(model) == parameter_count(baseline)
    baseline.set_stage("rgb", torch.eye(768))
    baseline_optimizer = build_optimizer(baseline, config.optimizer)
    baseline_result = train_one_step(
        baseline,
        baseline_optimizer,
        targets,
        labels,
        RNGStreams.create(8),
        config,
        stage="rgb",
    )
    assert torch.isfinite(baseline_result.loss.total)


def test_source_model_copy_stays_independent() -> None:
    config = _config()
    first = LightningDiT(config.model, config.detail_heads)
    second = copy.deepcopy(first)
    with torch.no_grad():
        next(second.parameters()).add_(1)
    assert not torch.equal(next(first.parameters()), next(second.parameters()))
