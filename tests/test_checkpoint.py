from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from cascade_dit.checkpoint import (
    EMA,
    TRAINING_STATE_SCHEMA_VERSION,
    CheckpointManager,
    ResumeState,
)
from cascade_dit.checkpoint_bundle import load_checkpoint_bundle
from cascade_dit.config import (
    DDTAttentionConfig,
    DDTPathConfig,
    ExperimentConfig,
    ModelConfig,
    PixelDecoderConfig,
    SemanticDecoderConfig,
)
from cascade_dit.model import DDT, build_model
from cascade_dit.rng import RNGStreams


class FastDetailer(nn.Module):
    def forward(self, coarse: torch.Tensor, _context: torch.Tensor) -> torch.Tensor:
        return coarse


def _config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.training.epochs = 2
    config.training.decoder.frozen_epochs = 1
    config.model = ModelConfig(
        base_prediction_depth=1,
        encoder=DDTPathConfig(16, 1, 1),
        semantic_decoder=SemanticDecoderConfig(32, 1, 2),
        pixel_decoder=PixelDecoderConfig(32, 3, 2, [1], [1]),
        attention=DDTAttentionConfig(rope=False),
    )
    config.model.conditioning.num_classes = 4
    config.validation.periodic_samples = 4
    config.evaluation.num_samples = 4
    return config


def _model(config: ExperimentConfig) -> DDT:
    model = build_model(config.model)
    model.set_pixel_detailer(FastDetailer())
    return model


def _state(config: ExperimentConfig, *, complete: bool = False) -> ResumeState:
    return ResumeState(
        schema_version=TRAINING_STATE_SCHEMA_VERSION,
        completed_epoch=config.total_epochs if complete else 1,
        global_step=7,
        training_complete=complete,
        images_seen=1024,
        estimated_flops=123,
        wall_clock_seconds=4.5,
        optimizer=None,
        scheduler=None,
        rng_streams=RNGStreams.create(4).state_dict(),
        data_fingerprint="data",
        representation_fingerprint="representations",
        pixel_decoder_fingerprint="pixel-decoder-sha256",
        decoder_phase="joint",
    )


def test_checkpoint_roundtrip_plain_state_and_old_schema_rejection(tmp_path) -> None:
    config = _config()
    model = _model(config)
    ema = EMA(model)
    manager = CheckpointManager(tmp_path)
    saved = manager.save(model=model, ema=ema, state=_state(config), config=config)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        next(model.parameters()).zero_()
    restored = manager.load(saved, model, ema, expected_data_fingerprint="data")
    assert restored.completed_epoch == 1 and not restored.training_complete
    assert all(torch.equal(model.state_dict()[name], value) for name, value in original.items())
    serialized = load_file(saved / "model.safetensors")
    assert set(serialized) == set(model.state_dict())
    assert not any(name.startswith("lpips") for name in serialized)

    old = tmp_path / "old"
    old.mkdir()
    (old / "metadata.json").write_text(json.dumps({"schema_version": 5}))
    with pytest.raises(ValueError, match="old checkpoints"):
        manager.load(old, model, ema)


def test_checkpoint_metadata_records_decoder_phase_and_any_checkpoint_can_be_sampled(
    tmp_path,
) -> None:
    config = _config()
    model = _model(config)
    manager = CheckpointManager(tmp_path)
    saved = manager.save(
        model=model,
        ema=EMA(model),
        state=_state(config, complete=False),
        config=config,
    )
    bundle = load_checkpoint_bundle(saved)
    assert bundle.weights.is_file()
    assert bundle.metadata["decoder_phase"] == "joint"
    assert bundle.metadata["pixel_decoder_fingerprint"] == "pixel-decoder-sha256"


def test_completed_joint_checkpoint_records_single_schedule_completion(tmp_path) -> None:
    config = _config()
    model = _model(config)
    saved = CheckpointManager(tmp_path).save(
        model=model,
        ema=EMA(model),
        state=_state(config, complete=True),
        config=config,
    )
    bundle = load_checkpoint_bundle(saved)
    assert bundle.metadata["training_complete"] is True
    assert bundle.metadata["completed_epoch"] == config.training.epochs


def test_ema_can_update_only_decoder_parameters() -> None:
    config = _config()
    model = _model(config)
    ema = EMA(model, decay=0.5)
    before = {name: value.clone() for name, value in ema.state_dict().items()}
    pixel_names = model.pixel_parameter_names()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    ema.update(model, pixel_names)
    after = ema.state_dict()
    assert any(not torch.equal(after[name], before[name]) for name in pixel_names)
    assert all(
        torch.equal(after[name], before[name])
        for name in model.flow_parameter_names()
    )
