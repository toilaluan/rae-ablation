from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file

from cascade_dit.checkpoint import (
    EMA,
    TRAINING_STATE_SCHEMA_VERSION,
    CheckpointManager,
    ResumeState,
)
from cascade_dit.checkpoint_bundle import load_checkpoint_bundle
from cascade_dit.config import (
    DDTAttentionConfig,
    DDTDecoderConfig,
    DDTPathConfig,
    ExperimentConfig,
    ModelConfig,
)
from cascade_dit.model import build_model
from cascade_dit.rng import RNGStreams


def _config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.model = ModelConfig(
        encoder=DDTPathConfig(16, 1, 1),
        decoder=DDTDecoderConfig(32, 2, 1, 2),
        attention=DDTAttentionConfig(rope=False),
    )
    config.model.conditioning.num_classes = 4
    config.validation.periodic_samples = 4
    config.evaluation.num_samples = 4
    return config


def _state(
    config: ExperimentConfig,
    *,
    phase: str = "semantic",
    complete: bool = False,
) -> ResumeState:
    completed = config.total_epochs if complete else 1
    epoch_in_phase = config.training.pixel.epochs if complete else 1
    return ResumeState(
        schema_version=TRAINING_STATE_SCHEMA_VERSION,
        completed_epoch=completed,
        global_step=7,
        phase="pixel" if complete else phase,  # type: ignore[arg-type]
        epoch_within_phase=epoch_in_phase,
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


def test_checkpoint_roundtrip_plain_state_and_old_schema_rejection(tmp_path) -> None:
    config = _config()
    model = build_model(config.model)
    ema = EMA(model)
    manager = CheckpointManager(tmp_path)
    saved = manager.save(model=model, ema=ema, state=_state(config), config=config)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        next(model.parameters()).zero_()
    restored = manager.load(saved, model, ema, expected_data_fingerprint="data")
    assert restored.phase == "semantic"
    assert all(torch.equal(model.state_dict()[name], value) for name, value in original.items())
    serialized = load_file(saved / "model.safetensors")
    assert set(serialized) == set(model.state_dict())
    assert not any("adapter" in name or ".base." in name for name in serialized)

    old = tmp_path / "old"
    old.mkdir()
    (old / "metadata.json").write_text(json.dumps({"schema_version": 4}))
    with pytest.raises(ValueError, match="old checkpoints"):
        manager.load(old, model, ema)


def test_pixel_boundary_is_idempotent_and_not_sampling_eligible(tmp_path) -> None:
    config = _config()
    model = build_model(config.model)
    ema = EMA(model)
    manager = CheckpointManager(tmp_path)
    state = ResumeState(
        **(
            vars(_state(config))
            | {
                "completed_epoch": config.training.semantic.epochs,
                "phase": "pixel",
                "epoch_within_phase": 0,
            }
        )
    )
    first = manager.save_phase_boundary(
        model=model, ema=ema, state=state, config=config
    )
    second = manager.save_phase_boundary(
        model=model, ema=ema, state=state, config=config
    )
    assert first == second
    with pytest.raises(ValueError, match="completed pixel"):
        load_checkpoint_bundle(first, require_rgb_complete=True)


def test_completed_pixel_checkpoint_is_sampling_eligible(tmp_path) -> None:
    config = _config()
    model = build_model(config.model)
    manager = CheckpointManager(tmp_path)
    saved = manager.save(
        model=model,
        ema=EMA(model),
        state=_state(config, complete=True),
        config=config,
    )
    bundle = load_checkpoint_bundle(saved, require_rgb_complete=True)
    assert bundle.weights.is_file()
    assert bundle.metadata["phase"] == "pixel"
