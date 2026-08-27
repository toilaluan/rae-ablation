from __future__ import annotations

import json

import pytest

import cascade_dit
from cascade_dit.config import CONFIG_SCHEMA_VERSION, ExperimentConfig, load_config


def test_checked_in_joint_schema_config() -> None:
    assert set(cascade_dit.__all__) == {"DDT", "DDTOutput", "build_model", "load_config"}
    config = load_config("configs/experiment.yaml")
    assert config.schema_version == CONFIG_SCHEMA_VERSION == 8
    assert config.total_epochs == config.training.epochs == 200
    assert config.training.timestep_shift == 4
    assert config.model.base_prediction_depth == 8
    assert config.model.semantic_decoder.width == 2048
    assert config.model.semantic_decoder.depth == 2
    assert config.model.pixel_decoder.depth == 3
    assert config.model.pixel_decoder.encoder_features == [4, 8, 12]
    assert config.model.pixel_decoder.decoder_features == [1, 2]
    assert config.model.pixel_decoder.fusion.kind == "concatenate"
    assert vars(config.model.pixel_decoder.fusion.layer_attention) == {
        "width": 768,
        "depth": 1,
        "heads": 12,
        "mlp_ratio": 2.0,
    }
    assert vars(config.training.decoder) == {
        "update_interval": 1,
        "updates_per_active_step": 1,
        "backpropagate_to_flow": True,
    }
    assert vars(config.sampling.internal_guidance) == {
        "scale": 1.78,
        "t_min": 0.1,
        "t_max": 1.0,
    }
    assert not hasattr(config.training, "mode")
    assert not hasattr(config.model, "decoder")


@pytest.mark.parametrize("version", [5, 6, 7])
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
    with pytest.raises(ValueError, match="schema-v8"):
        load_config("configs/experiment.yaml", [field])


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


def test_internal_guidance_interval_validation() -> None:
    config = ExperimentConfig()
    config.sampling.internal_guidance.t_min = 1
    with pytest.raises(ValueError, match="internal-guidance interval"):
        config.validate()
