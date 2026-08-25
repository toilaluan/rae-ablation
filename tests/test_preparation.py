from __future__ import annotations

from copy import deepcopy

import pytest

from cascade_dit.config import (
    ExperimentConfig,
    RepresentationSourceConfig,
    RepresentationStageConfig,
)
from cascade_dit.data import data_fingerprint
from cascade_dit.preparation import (
    _initialize_preparation_state,
    preparation_fingerprint,
    validate_manifest_compatibility,
)
from cascade_dit.representations import RepresentationManifest
from cascade_dit.vision import encoder_fingerprint


def _config_with_layer_8() -> ExperimentConfig:
    config = ExperimentConfig()
    config.representations.stages = [
        RepresentationStageConfig(
            "dino12", 1, RepresentationSourceConfig("dino", 12)
        ),
        RepresentationStageConfig(
            "dino8", 1, RepresentationSourceConfig("dino", 8)
        ),
        RepresentationStageConfig("rgb", 1, RepresentationSourceConfig("rgb")),
    ]
    return config


def _manifest(config: ExperimentConfig) -> RepresentationManifest:
    stages = [
        {"name": stage.name, "source": vars(stage.source)} for stage in config.stages
    ]
    return RepresentationManifest(
        dataset_size=2,
        dataset_fingerprint=data_fingerprint(config.data, config.data.image_size),
        encoder_fingerprint=encoder_fingerprint(config.encoder, config.dino_layers),
        preparation_fingerprint=preparation_fingerprint(config),
        stages=stages,
        statistics={},
        cache_files={},
        completion_files={},
        complete=True,
    )


def test_cached_manifest_can_reuse_an_ordered_subset_without_layer_8() -> None:
    cached_config = _config_with_layer_8()
    active_config = deepcopy(cached_config)
    active_config.representations.stages.pop(1)

    validate_manifest_compatibility(active_config, _manifest(cached_config))


def test_cached_manifest_still_rejects_an_uncached_layer() -> None:
    cached_config = _config_with_layer_8()
    active_config = deepcopy(cached_config)
    active_config.representations.stages[1] = RepresentationStageConfig(
        "dino4", 1, RepresentationSourceConfig("dino", 4)
    )

    with pytest.raises(ValueError, match="does not include"):
        validate_manifest_compatibility(active_config, _manifest(cached_config))


def test_in_progress_cache_can_resume_without_layer_8(tmp_path) -> None:
    cached_config = _config_with_layer_8()
    _initialize_preparation_state(tmp_path, cached_config, dataset_size=2)
    active_config = deepcopy(cached_config)
    active_config.representations.stages.pop(1)

    _initialize_preparation_state(tmp_path, active_config, dataset_size=2)
