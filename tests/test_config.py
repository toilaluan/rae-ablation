from __future__ import annotations

import json

import pytest

from cascade_dit.config import CONFIG_SCHEMA_VERSION, ExperimentConfig, load_config


def test_checked_in_config_is_complete_and_selects_baseline_by_override() -> None:
    config = load_config("configs/experiment.yaml")
    assert config.total_epochs == 80
    assert (config.model.depth, config.model.hidden_size, config.model.num_heads) == (12, 768, 12)
    assert config.curriculum.stages == ["dino12", "dino8", "dino4", "rgb"]
    baseline = load_config(
        "configs/experiment.yaml", ["experiment.mode=pixel_baseline"]
    )
    assert baseline.experiment.mode == "pixel_baseline"


def test_old_ddt_mage_schema_is_rejected_without_migration(tmp_path) -> None:
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema_version": 2, "image_latents": {}, "train": {}}))
    with pytest.raises(ValueError, match="DDT/Mage"):
        load_config(path)


def test_block_lists_are_validated() -> None:
    config = ExperimentConfig()
    config.curriculum.lora_blocks = [2, 2]
    with pytest.raises(ValueError, match="disjoint"):
        config.validate()
    config = ExperimentConfig()
    config.curriculum.fully_train_blocks = [12]
    with pytest.raises(ValueError, match="outside"):
        config.validate()
    assert config.schema_version == CONFIG_SCHEMA_VERSION
