from __future__ import annotations

import json

import pytest
import torch

from cascade_dit.checkpoint import EMA, CheckpointManager, ResumeState
from cascade_dit.checkpoint_bundle import load_checkpoint_bundle
from cascade_dit.config import DDTAttentionConfig, DDTPathConfig, ExperimentConfig, ModelConfig
from cascade_dit.model import DDT
from cascade_dit.rng import RNGStreams


def _config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.model = ModelConfig(
        encoder=DDTPathConfig(16, 1, 1),
        decoder=DDTPathConfig(32, 1, 2),
        attention=DDTAttentionConfig(rope=False),
    )
    config.model.conditioning.num_classes = 4
    return config


def _model(config: ExperimentConfig) -> DDT:
    return DDT(config.model, config.num_representation_levels)


def _state(config: ExperimentConfig, *, complete: bool = False) -> ResumeState:
    stage_index = config.num_representation_levels - 1 if complete else 0
    return ResumeState(
        schema_version=4,
        completed_epoch=config.total_epochs if complete else 20,
        global_step=7,
        stage_index=stage_index,
        stage=config.stage_names[stage_index],
        epoch_within_stage=20,
        transition_state="stable",
        training_complete=complete,
        images_seen=1024,
        estimated_flops=123,
        wall_clock_seconds=4.5,
        optimizer=None,
        scheduler=None,
        rng_streams=RNGStreams.create(4).state_dict(),
        data_fingerprint="data",
        representation_fingerprint="representations",
    )


def test_checkpoint_roundtrip_and_old_schema_rejection(tmp_path) -> None:
    config = _config()
    model = _model(config)
    ema = EMA(model)
    manager = CheckpointManager(tmp_path)
    saved = manager.save(model=model, ema=ema, state=_state(config), config=config)
    with torch.no_grad():
        next(model.parameters()).zero_()
    restored = manager.load(saved, model, ema, expected_data_fingerprint="data")
    assert restored.completed_epoch == 20
    old = tmp_path / "old"
    old.mkdir()
    (old / "metadata.json").write_text(json.dumps({"schema_version": 3}))
    with pytest.raises(ValueError, match="incompatible"):
        manager.load(old, model, ema)


def test_boundary_save_is_idempotent_and_requires_complete_rgb(tmp_path) -> None:
    config = _config()
    model = _model(config)
    ema = EMA(model)
    manager = CheckpointManager(tmp_path)
    before = ResumeState(**(vars(_state(config)) | {"transition_state": "before"}))
    first = manager.save_boundary(
        target_stage_index=1,
        position="before",
        model=model,
        ema=ema,
        state=before,
        config=config,
    )
    second = manager.save_boundary(
        target_stage_index=1,
        position="before",
        model=model,
        ema=ema,
        state=before,
        config=config,
    )
    assert first == second
    with pytest.raises(ValueError, match="completed RGB"):
        load_checkpoint_bundle(first, require_rgb_complete=True)


def test_completed_rgb_checkpoint_is_sampling_eligible(tmp_path) -> None:
    config = _config()
    model = _model(config)
    ema = EMA(model)
    manager = CheckpointManager(tmp_path)
    saved = manager.save(
        model=model,
        ema=ema,
        state=_state(config, complete=True),
        config=config,
    )
    assert load_checkpoint_bundle(saved, require_rgb_complete=True).weights.is_file()
