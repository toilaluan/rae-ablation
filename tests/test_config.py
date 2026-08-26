from __future__ import annotations

import json

import pytest

import cascade_dit
from cascade_dit.config import CONFIG_SCHEMA_VERSION, ExperimentConfig, load_config


def test_checked_in_schema_v7_config_and_both_modes() -> None:
    assert set(cascade_dit.__all__) == {"DDT", "DDTOutput", "build_model", "load_config"}
    config = load_config("configs/experiment.yaml")
    assert config.schema_version == CONFIG_SCHEMA_VERSION == 7
    assert config.training.mode == "progressive"
    assert config.total_epochs == 170
    assert config.phase_for_epoch(49) == ("semantic", 49)
    assert config.phase_for_epoch(50) == ("pixel", 0)
    assert config.representations.semantic.name == "dino12"
    assert config.model.decoder.semantic_depth == 2

    baseline = load_config(
        "configs/experiment.yaml",
        ["training.mode=pixel_only", "model.encoder.depth=6"],
    )
    assert baseline.total_epochs == 120
    assert baseline.phase_for_epoch(0) == ("pixel", 0)
    assert baseline.model.encoder.depth == 6


@pytest.mark.parametrize("version", [5, 6])
def test_old_config_schemas_are_rejected(tmp_path, version: int) -> None:
    path = tmp_path / f"v{version}.json"
    path.write_text(json.dumps({"schema_version": version}))
    with pytest.raises(ValueError, match="not migrated"):
        load_config(path)


@pytest.mark.parametrize("semantic_depth", [0, 3, 4])
def test_decoder_split_is_validated(semantic_depth: int) -> None:
    config = ExperimentConfig()
    config.model.decoder.depth = 3
    config.model.decoder.semantic_depth = semantic_depth
    with pytest.raises(ValueError, match="semantic_depth"):
        config.validate()


def test_ratio_probability_and_auxiliary_weight_validation() -> None:
    config = ExperimentConfig()
    config.training.semantic.mixed_probability = 1.01
    with pytest.raises(ValueError, match="mixed_probability"):
        config.validate()

    config = ExperimentConfig()
    config.training.semantic.mix_ratio.scale = 0
    with pytest.raises(ValueError, match=r"mix_ratio\.scale"):
        config.validate()

    config = ExperimentConfig()
    config.training.semantic.mix_ratio.distribution = "uniform"
    with pytest.raises(ValueError, match="logit_normal"):
        config.validate()

    config = ExperimentConfig()
    config.objective.pixel_auxiliary_mse_weight = -1
    with pytest.raises(ValueError, match="non-negative"):
        config.validate()


def test_invalid_mode_and_epoch_are_rejected() -> None:
    config = ExperimentConfig()
    config.training.mode = "baseline"
    with pytest.raises(ValueError, match="progressive or pixel_only"):
        config.validate()
    with pytest.raises(ValueError, match="outside"):
        ExperimentConfig().phase_for_epoch(-1)
