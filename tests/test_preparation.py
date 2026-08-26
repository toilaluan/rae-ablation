from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from safetensors.torch import save_file

from cascade_dit.config import ExperimentConfig
from cascade_dit.data import data_fingerprint
from cascade_dit.preparation import (
    preparation_fingerprint,
    prepare_representations,
    validate_manifest_compatibility,
)
from cascade_dit.representations import LayerMemmap, RepresentationManifest
from cascade_dit.vision import encoder_fingerprint

CACHED_STAGES = [
    {"name": "dino12", "source": {"kind": "dino", "layer": 12}},
    {"name": "dino8", "source": {"kind": "dino", "layer": 8}},
    {"name": "rgb", "source": {"kind": "rgb", "layer": None}},
]


def _manifest(config: ExperimentConfig) -> RepresentationManifest:
    statistics = {
        name: {"mean": [0.0] * 768, "std": [1.0] * 768, "count": 128}
        for name in ("dino12", "dino8", "rgb")
    }
    return RepresentationManifest(
        dataset_size=2,
        dataset_fingerprint=data_fingerprint(config.data, config.data.image_size),
        encoder_fingerprint=encoder_fingerprint(config.encoder, [12, 8]),
        preparation_fingerprint=preparation_fingerprint(config, CACHED_STAGES),
        stages=CACHED_STAGES,
        statistics=statistics,
        cache_files={"dino12": "dino12.bf16", "dino8": "dino8.bf16"},
        completion_files={
            "dino12": "dino12.completion.u8",
            "dino8": "dino8.completion.u8",
        },
        complete=True,
    )


def test_existing_manifest_with_extra_dino_layer_is_reused(tmp_path) -> None:
    config = ExperimentConfig()
    config.representations.directory = str(tmp_path)
    manifest = _manifest(config)
    for name in ("dino12", "dino8"):
        cache = LayerMemmap.initialize(tmp_path, name, 2)
        cache.write(torch.tensor([0, 1]), torch.zeros(2, 64, 768))
    save_file(
        {
            f"statistics.{name}.{statistic}": torch.zeros(768)
            if statistic == "mean"
            else torch.ones(768)
            for name in ("dino12", "dino8", "rgb")
            for statistic in ("mean", "std")
        },
        tmp_path / "statistics.safetensors",
    )
    manifest.save(tmp_path / "manifest.json")

    loaded = prepare_representations(config, rank=0, world_size=1, device=torch.device("cpu"))

    assert loaded.stage_names == ["dino12", "dino8", "rgb"]
    validate_manifest_compatibility(config, loaded)


def test_manifest_rejects_a_different_selected_semantic_endpoint() -> None:
    cached_config = ExperimentConfig()
    active_config = deepcopy(cached_config)
    active_config.representations.semantic.name = "dino4"
    active_config.representations.semantic.layer = 4
    with pytest.raises(ValueError, match="selected semantic"):
        validate_manifest_compatibility(active_config, _manifest(cached_config))
