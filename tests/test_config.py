from __future__ import annotations

import json

import pytest

import cascade_dit
from cascade_dit.config import (
    CONFIG_SCHEMA_VERSION,
    ExperimentConfig,
    load_checkpoint_config,
    load_config,
)


def test_checked_in_joint_schema_config() -> None:
    assert set(cascade_dit.__all__) == {"DDT", "DDTOutput", "build_model", "load_config"}
    config = load_config("configs/experiment.yaml")
    assert config.schema_version == CONFIG_SCHEMA_VERSION == 10
    assert config.total_epochs == config.training.epochs == 200
    assert config.training.timestep_shift == 4
    assert config.model.base_prediction_depth == 4
    assert config.model.semantic_decoder.width == 2048
    assert config.model.semantic_decoder.depth == 2
    assert config.model.pixel_decoder.depth == 8
    assert config.model.pixel_decoder.encoder_features == [2, 4, 6, 8, 12]
    assert config.model.pixel_decoder.decoder_features == [1, 2]
    assert config.model.pixel_decoder.fusion.kind == "layer_attention"
    assert vars(config.model.pixel_decoder.fusion.layer_attention) == {
        "width": 1024,
        "depth": 2,
        "heads": 16,
        "mlp_ratio": 4.0,
    }
    assert vars(config.training.decoder) == {
        "pretrained_checkpoint": "artifacts/pixel-decoder/checkpoints/epoch-0050",
        "pretrained_weights": "ema",
        "frozen_epochs": 20,
        "update_interval": 1,
        "updates_per_active_step": 1,
        "backpropagate_to_flow": True,
        "samples_per_batch": 64,
    }
    assert config.pixel_pretraining.epochs == 50
    assert config.pixel_pretraining.semantic_input_mode == "cached"
    assert config.pixel_pretraining.data.image_column == "image"
    assert config.pixel_pretraining.optimizer.scheduler.warmup_epochs == 5
    assert vars(config.sampling.internal_guidance) == {
        "scale": 1.78,
        "t_min": 0.1,
        "t_max": 1.0,
    }
    assert config.sampling.final_decode_timestep == 0.5
    assert vars(config.validation.reconstruction) == {
        "split": "validation",
        "cache_directory": "data/reconstruction-validation",
        "manifest": "manifest.json",
        "samples_per_class": 1,
        "batch_size": 32,
        "preview_samples": 8,
    }
    assert not hasattr(config.training, "mode")
    assert not hasattr(config.model, "decoder")


@pytest.mark.parametrize("version", [5, 6, 7, 8, 9])
def test_old_config_schemas_are_rejected(tmp_path, version: int) -> None:
    path = tmp_path / f"v{version}.json"
    path.write_text(json.dumps({"schema_version": version}))
    with pytest.raises(ValueError, match="intentionally not migrated"):
        load_config(path)


@pytest.mark.parametrize(
    "field",
    [
        "training.mode=progressive",
        "training.pixel.epochs=1",
        "model.decoder.depth=2",
        "objective.pixel_auxiliary_mse_weight=1",
        "sampling.guidance_scale=3",
        "checkpoint.save_phase_boundary=true",
    ],
)
def test_removed_fields_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match="schema-v10"):
        load_config("configs/experiment.yaml", [field])


def test_schema_v9_checkpoint_config_preserves_legacy_decode_timestep(tmp_path) -> None:
    values = ExperimentConfig().as_dict()
    values["schema_version"] = 9
    values["sampling"].pop("final_decode_timestep")
    values["validation"].pop("reconstruction")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))

    migrated = load_checkpoint_config(path)

    assert migrated.schema_version == 10
    assert migrated.sampling.final_decode_timestep == 0.0
    assert migrated.validation.reconstruction.samples_per_class == 1


def test_feature_base_and_weight_validation() -> None:
    config = ExperimentConfig()
    config.model.base_prediction_depth = 13
    with pytest.raises(ValueError, match="base_prediction_depth"):
        config.validate()

    config = ExperimentConfig()
    config.model.pixel_decoder.encoder_features = [4, 4]
    with pytest.raises(ValueError, match="unique"):
        config.validate()

    config = ExperimentConfig()
    config.model.pixel_decoder.decoder_features = [3]
    with pytest.raises(ValueError, match="out-of-range"):
        config.validate()

    config = ExperimentConfig()
    config.objective.pixel_lpips_weight = -1
    with pytest.raises(ValueError, match="non-negative"):
        config.validate()

    config = ExperimentConfig()
    config.model.pixel_decoder.fusion.kind = "unknown"
    with pytest.raises(ValueError, match=r"fusion\.kind"):
        config.validate()

    config = ExperimentConfig()
    config.training.decoder.update_interval = 0
    with pytest.raises(ValueError, match="decoder optimization"):
        config.validate()

    config = ExperimentConfig()
    config.training.decoder.samples_per_batch = 0
    with pytest.raises(ValueError, match="samples_per_batch"):
        config.validate()

    config = ExperimentConfig()
    config.training.decoder.pretrained_weights = "best"
    with pytest.raises(ValueError, match="pretrained_weights"):
        config.validate()

    config = ExperimentConfig()
    config.pixel_pretraining.semantic_input_mode = "unknown"
    with pytest.raises(ValueError, match="semantic_input_mode"):
        config.validate()


def test_internal_guidance_interval_validation() -> None:
    config = ExperimentConfig()
    config.sampling.internal_guidance.t_min = 1
    with pytest.raises(ValueError, match="internal-guidance interval"):
        config.validate()

    config = ExperimentConfig()
    config.sampling.final_decode_timestep = float("nan")
    with pytest.raises(ValueError, match="final_decode_timestep"):
        config.validate()
