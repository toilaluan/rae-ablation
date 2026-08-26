from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from cascade_dit.representations import (
    ChannelStatisticsAccumulator,
    LayerMemmap,
    RepresentationManifest,
    deterministic_subset_indices,
    patchify_rgb,
    required_representation_bytes,
    unpatchify_rgb,
)


def test_rgb_patchification_roundtrips_in_row_major_order() -> None:
    images = torch.arange(2 * 3 * 128 * 128, dtype=torch.float32).reshape(
        2, 3, 128, 128
    )
    tokens = patchify_rgb(images)
    assert tokens.shape == (2, 64, 768)
    assert torch.equal(unpatchify_rgb(tokens), images)


def test_streaming_channel_statistics_normalize_to_unit_variance() -> None:
    values = torch.randn(5, 7, 4, generator=torch.Generator().manual_seed(3))
    accumulator = ChannelStatisticsAccumulator(4)
    accumulator.update(values[:2])
    accumulator.update(values[2:])
    statistics = accumulator.finalize()
    normalized = statistics.normalize(values)
    assert torch.allclose(normalized.mean((0, 1)), torch.zeros(4), atol=1e-6)
    assert torch.allclose(normalized.square().mean((0, 1)), torch.ones(4), atol=1e-5)


def test_dynamic_layer_cache_is_resumable_and_sized() -> None:
    assert required_representation_bytes(2, dino_levels=2) == 2 * 2 * (64 * 768 * 2 + 1)


def test_layer_cache_is_resumable_and_deterministic(tmp_path) -> None:
    cache = LayerMemmap.initialize(tmp_path, "deep", 2)
    first = torch.randn(1, 64, 768, generator=torch.Generator().manual_seed(9))
    cache.write(torch.tensor([1]), first)
    resumed = LayerMemmap.initialize(tmp_path, "deep", 2)
    assert resumed.pending_indices().tolist() == [0]
    assert torch.equal(resumed.lookup(torch.tensor([1])), first.bfloat16())
    resumed.write(torch.tensor([0]), torch.zeros(1, 64, 768))
    assert resumed.pending_indices().size == 0


def test_normalization_only_manifest_and_old_schema_rejection(tmp_path) -> None:
    cache = LayerMemmap.initialize(tmp_path, "deep", 2)
    cache.write(torch.tensor([0, 1]), torch.zeros(2, 64, 768))
    statistics = {
        "deep": {"mean": [0.0] * 768, "std": [1.0] * 768, "count": 128},
        "rgb": {"mean": [0.0] * 768, "std": [1.0] * 768, "count": 128},
    }
    save_file(
        {
            "statistics.deep.mean": torch.zeros(768),
            "statistics.deep.std": torch.ones(768),
            "statistics.rgb.mean": torch.zeros(768),
            "statistics.rgb.std": torch.ones(768),
        },
        tmp_path / "statistics.safetensors",
    )
    manifest = RepresentationManifest(
        dataset_size=2,
        dataset_fingerprint="data",
        encoder_fingerprint="encoder",
        preparation_fingerprint="preparation",
        stages=[
            {"name": "deep", "source": {"kind": "dino", "layer": 12}},
            {"name": "rgb", "source": {"kind": "rgb", "layer": None}},
        ],
        statistics=statistics,
        cache_files={"deep": "deep.bf16"},
        completion_files={"deep": "deep.completion.u8"},
        complete=True,
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = RepresentationManifest.load(path)
    assert loaded.stage_names == ["deep", "rgb"]

    path.write_text(json.dumps({"schema_version": 2}))
    with pytest.raises(ValueError, match="alignment-era"):
        RepresentationManifest.load(path)
    path.write_text("not-json")
    with pytest.raises(ValueError, match="corrupt"):
        RepresentationManifest.load(path)


def test_subset_selection_is_deterministic() -> None:
    assert torch.equal(
        deterministic_subset_indices(100, 12, 4),
        deterministic_subset_indices(100, 12, 4),
    )


def test_rgb_only_manifest_validation_ignores_dino_statistics_and_cache(tmp_path) -> None:
    save_file(
        {
            "statistics.rgb.mean": torch.zeros(768),
            "statistics.rgb.std": torch.ones(768),
        },
        tmp_path / "statistics.safetensors",
    )
    manifest = RepresentationManifest(
        dataset_size=2,
        dataset_fingerprint="data",
        encoder_fingerprint="encoder",
        preparation_fingerprint="preparation",
        stages=[
            {"name": "deep", "source": {"kind": "dino", "layer": 12}},
            {"name": "rgb", "source": {"kind": "rgb", "layer": None}},
        ],
        statistics={
            "rgb": {"mean": [0.0] * 768, "std": [1.0] * 768, "count": 128}
        },
        cache_files={},
        completion_files={},
        complete=True,
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)

    loaded = RepresentationManifest.load(path, required_stages=["rgb"])

    assert loaded.stage_names[-1] == "rgb"
