from __future__ import annotations

import json

import pytest
import torch

from cascade_dit.representations import (
    ChannelStatisticsAccumulator,
    LayerMemmap,
    RepresentationManifest,
    deterministic_subset_indices,
    patchify_rgb,
    unpatchify_rgb,
)


def test_rgb_patchification_roundtrips_in_row_major_order() -> None:
    images = torch.arange(2 * 3 * 128 * 128, dtype=torch.float32).reshape(2, 3, 128, 128)
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


def test_layer_cache_is_resumable_and_deterministic(tmp_path) -> None:
    cache = LayerMemmap.initialize(tmp_path, "dino12", 2)
    first = torch.randn(1, 64, 768, generator=torch.Generator().manual_seed(9))
    cache.write(torch.tensor([1]), first)
    resumed = LayerMemmap.initialize(tmp_path, "dino12", 2)
    assert resumed.pending_indices().tolist() == [0]
    assert torch.equal(resumed.lookup(torch.tensor([1])), first.bfloat16())
    resumed.write(torch.tensor([0]), torch.zeros(1, 64, 768))
    resumed.finalize()


def test_subset_selection_and_corrupt_manifest_rejection(tmp_path) -> None:
    assert torch.equal(
        deterministic_subset_indices(100, 12, 4),
        deterministic_subset_indices(100, 12, 4),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 999}))
    with pytest.raises(ValueError, match="schema"):
        RepresentationManifest.load(path)
    path.write_text("not-json")
    with pytest.raises(ValueError, match="corrupt"):
        RepresentationManifest.load(path)
