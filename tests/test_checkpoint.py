from __future__ import annotations

import json

import pytest
import torch

from cascade_dit.checkpoint import EMA, CheckpointManager, ResumeState
from cascade_dit.checkpoint_bundle import load_checkpoint_bundle
from cascade_dit.config import DetailHeadsConfig, ExperimentConfig, ModelConfig
from cascade_dit.model import LightningDiT
from cascade_dit.rng import RNGStreams


def _model() -> LightningDiT:
    return LightningDiT(
        ModelConfig(hidden_size=16, depth=1, num_heads=1, num_classes=4, rope=False),
        DetailHeadsConfig(pixel_enabled=False),
    )


def _state(model: LightningDiT, *, complete: bool = False) -> ResumeState:
    optimizer = torch.optim.AdamW(model.parameters())
    return ResumeState(
        schema_version=3,
        completed_epoch=80 if complete else 20,
        global_step=7,
        stage_index=3 if complete else 0,
        stage="rgb" if complete else "dino12",
        epoch_within_stage=20,
        transition_state="stable",
        training_complete=complete,
        images_seen=1024,
        estimated_flops=123,
        wall_clock_seconds=4.5,
        optimizer=optimizer.state_dict(),
        scheduler=None,
        rng_streams=RNGStreams.create(4).state_dict(),
        data_fingerprint="data",
        representation_fingerprint="representations",
    )


def test_checkpoint_roundtrip_and_old_schema_rejection(tmp_path) -> None:
    model = _model()
    ema = EMA(model)
    manager = CheckpointManager(tmp_path)
    saved = manager.save(model=model, ema=ema, state=_state(model), config=ExperimentConfig())
    with torch.no_grad():
        next(model.parameters()).zero_()
    restored = manager.load(saved, model, ema, expected_data_fingerprint="data")
    assert restored.completed_epoch == 20
    old = tmp_path / "old"
    old.mkdir()
    (old / "metadata.json").write_text(json.dumps({"schema_version": 2}))
    with pytest.raises(ValueError, match="incompatible"):
        manager.load(old, model, ema)


def test_boundary_save_is_idempotent_and_sampling_requires_complete_rgb(tmp_path) -> None:
    model = _model()
    ema = EMA(model)
    manager = CheckpointManager(tmp_path)
    before = _state(model)
    before = ResumeState(**(vars(before) | {"transition_state": "before"}))
    first = manager.save_boundary(
        target_stage_index=1,
        position="before",
        model=model,
        ema=ema,
        state=before,
        config=ExperimentConfig(),
    )
    second = manager.save_boundary(
        target_stage_index=1,
        position="before",
        model=model,
        ema=ema,
        state=before,
        config=ExperimentConfig(),
    )
    assert first == second
    with pytest.raises(ValueError, match="completed RGB"):
        load_checkpoint_bundle(first, require_rgb_complete=True)


def test_completed_rgb_checkpoint_is_sampling_eligible(tmp_path) -> None:
    model = _model()
    model.set_stage("rgb", torch.eye(768))
    ema = EMA(model)
    manager = CheckpointManager(tmp_path)
    saved = manager.save(
        model=model, ema=ema, state=_state(model, complete=True), config=ExperimentConfig()
    )
    assert load_checkpoint_bundle(saved, require_rgb_complete=True).weights.is_file()
