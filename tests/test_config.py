from __future__ import annotations

import json

import pytest

from cascade_dit.config import (
    CONFIG_SCHEMA_VERSION,
    ExperimentConfig,
    RepresentationSourceConfig,
    RepresentationStageConfig,
    load_config,
)


def test_checked_in_nested_config_and_baseline_override() -> None:
    config = load_config("configs/experiment.yaml")
    assert config.total_epochs == 80
    assert config.stage_names == ["dino12", "dino8", "dino4", "rgb"]
    assert config.dino_layers == [12, 8, 4]
    assert (config.model.encoder.width, config.model.encoder.depth) == (768, 12)
    assert (config.model.decoder.width, config.model.decoder.depth) == (2048, 2)
    assert config.curriculum.mixing.pure_current_probability == pytest.approx(0.85)
    baseline = load_config(
        "configs/experiment.yaml",
        ["experiment.mode=pixel_baseline", "model.encoder.depth=6"],
    )
    assert baseline.experiment.mode == "pixel_baseline"
    assert baseline.model.encoder.depth == 6


def test_dynamic_stage_list_controls_R_width_and_epochs() -> None:
    config = ExperimentConfig()
    config.representations.stages = [
        RepresentationStageConfig(
            "semantic", 3, RepresentationSourceConfig("dino", 11)
        ),
        RepresentationStageConfig(
            "detail", 4, RepresentationSourceConfig("dino", 2)
        ),
        RepresentationStageConfig("pixels", 5, RepresentationSourceConfig("rgb")),
    ]
    config.validate()
    assert config.num_representation_levels == 3
    assert config.total_epochs == 12
    assert config.dino_layers == [11, 2]


def test_schema_stage_and_probability_validation(tmp_path) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema_version": 3, "alignment": {}}))
    with pytest.raises(ValueError, match="Alignment-era"):
        load_config(path)

    config = ExperimentConfig()
    config.curriculum.mixing.adjacent.probability = 0.98
    with pytest.raises(ValueError, match="sum above one"):
        config.validate()

    config = ExperimentConfig()
    config.representations.stages[-1].source.kind = "dino"
    with pytest.raises(ValueError, match="final stage"):
        config.validate()
    assert ExperimentConfig().schema_version == CONFIG_SCHEMA_VERSION
