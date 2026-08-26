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
    assert config.total_epochs == 175
    assert config.stage_names == ["dino12", "dino8", "rgb"]
    assert config.dino_layers == [12, 8]
    assert (config.model.encoder.width, config.model.encoder.depth) == (768, 12)
    assert (config.model.decoder.width, config.model.decoder.depth) == (2048, 2)
    assert config.curriculum.mixing.pure_current_probability == pytest.approx(0.9)
    assert config.optimizer.scheduler.kind == "constant"
    baseline = load_config(
        "configs/experiment.yaml",
        [
            "pixel_training.use_representation_guiding=false",
            "model.encoder.depth=6",
        ],
    )
    assert not baseline.pixel_training.use_representation_guiding
    assert baseline.total_epochs == 120
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
    ]
    config.pixel_training.name = "pixels"
    config.pixel_training.epochs = 5
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
    config.representations.stages[-1].source.kind = "rgb"
    with pytest.raises(ValueError, match="must be DINO"):
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


def test_schema_v5_moves_rgb_and_mode_under_pixel_training(tmp_path) -> None:
    current = load_config("configs/experiment.yaml").as_dict()
    pixel = current.pop("pixel_training")
    representations = pixel["representations"]
    representations["stages"].append(
        {
            "name": pixel["name"],
            "epochs": pixel["epochs"],
            "source": {"kind": "rgb", "layer": None},
            "timestep_shift": pixel["timestep_shift"],
        }
    )
    current["schema_version"] = 5
    current["experiment"]["mode"] = "pixel_baseline"
    current["representations"] = representations
    path = tmp_path / "v5.json"
    path.write_text(json.dumps(current))

    migrated = load_config(path)

    assert migrated.schema_version == 6
    assert migrated.pixel_training.name == "rgb"
    assert not migrated.pixel_training.use_representation_guiding
    assert migrated.stage_names == ["dino12", "dino8", "rgb"]


def test_pixel_only_validation_ignores_dino_preparation_settings() -> None:
    config = ExperimentConfig()
    config.pixel_training.use_representation_guiding = False
    config.encoder.repository = "unused"
    config.representations.dtype = "unused"
    config.representations.require_complete = False
    config.representations.stages = []

    config.validate()

    assert config.total_epochs == config.pixel_training.epochs
    assert config.num_representation_levels == 1
