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
    assert config.total_epochs == 50
    assert config.stage_names == ["dino12", "dino8", "rgb"]
    assert config.dino_layers == [12, 8]
    assert (config.model.encoder.width, config.model.encoder.depth) == (1152, 12)
    assert (config.model.decoder.width, config.model.decoder.depth) == (2048, 2)
    assert config.curriculum.mixing.pure_current_probability == pytest.approx(0.85)
    assert config.optimizer.scheduler.kind == "cosine"
    baseline = load_config(
        "configs/experiment.yaml",
        ["experiment.mode=pixel_baseline", "model.encoder.depth=6"],
    )
    assert baseline.experiment.mode == "pixel_baseline"
    assert baseline.model.encoder.depth == 6
    constant = load_config(
        "configs/experiment.yaml", ["optimizer.scheduler.kind=constant"]
    )
    assert constant.optimizer.scheduler.kind == "constant"


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
    path.write_text(json.dumps({"schema_version": 4}))
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


def test_stage_timestep_shifts_and_evaluation_sample_counts_are_validated() -> None:
    config = ExperimentConfig()
    config.representations.stages[1].timestep_shift = 0
    with pytest.raises(ValueError, match="timestep shifts"):
        config.validate()

    config = ExperimentConfig()
    config.validation.periodic_samples = 1_500
    with pytest.raises(ValueError, match="class balanced"):
        config.validate()

    config.validation.periodic_samples = 51_000
    with pytest.raises(ValueError, match="cannot exceed"):
        config.validate()

    config = ExperimentConfig()
    config.optimizer.scheduler.kind = "linear"
    with pytest.raises(ValueError, match="constant or cosine"):
        config.validate()
