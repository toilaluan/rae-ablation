from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from jit_reg.latent_statistics import (
    EndpointLatentNormalizer,
    LatentStatisticsAccumulator,
    latent_statistics_sha256,
    load_latent_statistics,
    save_latent_statistics,
)


def test_all_endpoint_statistics_and_round_trip(tmp_path) -> None:
    target = torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[6.0]]]])
    semantic = torch.tensor([[[[5.0]], [[10.0]]], [[[7.0]], [[14.0]]]])
    pixel = torch.tensor([[[[0.0]], [[4.0]]], [[[2.0]], [[8.0]]]])
    accumulator = LatentStatisticsAccumulator(2)
    accumulator.update(target, semantic, pixel)
    statistics = accumulator.finalize(
        vision_fingerprint="vision-id",
        data_fingerprint="data-id",
    )

    assert statistics.patch_count == 2
    assert torch.allclose(statistics.target_mean, torch.tensor([2.0, 4.0]))
    assert torch.allclose(statistics.target_std, torch.tensor([1.0, 2.0]))
    assert torch.allclose(statistics.semantic_mean, torch.tensor([6.0, 12.0]))
    assert torch.allclose(statistics.pixel_mean, torch.tensor([1.0, 6.0]))

    path = tmp_path / "statistics.safetensors"
    save_latent_statistics(statistics, path)
    loaded = load_latent_statistics(
        path,
        expected_channels=2,
        expected_vision_fingerprint="vision-id",
        expected_data_fingerprint="data-id",
    )
    assert torch.equal(loaded.pixel_std, statistics.pixel_std)
    assert len(latent_statistics_sha256(path)) == 64

    normalizer = EndpointLatentNormalizer(loaded)
    assert torch.allclose(
        normalizer.denormalize_target(normalizer.normalize_target(target)), target
    )
    assert torch.allclose(
        normalizer.denormalize_semantic(normalizer.normalize_semantic(semantic)),
        semantic,
    )
    assert torch.allclose(
        normalizer.denormalize_pixel(normalizer.normalize_pixel(pixel)), pixel
    )


def test_legacy_latent_statistics_schema_is_rejected(tmp_path) -> None:
    path = tmp_path / "legacy.safetensors"
    vector = torch.ones(2)
    save_file(
        {
            name: vector.clone()
            for name in (
                "target_mean",
                "target_std",
                "semantic_mean",
                "semantic_std",
                "pixel_mean",
                "pixel_std",
            )
        },
        path,
        metadata={"schema_version": "3"},
    )
    with pytest.raises(ValueError, match="incompatible schema"):
        load_latent_statistics(path, expected_channels=2)
