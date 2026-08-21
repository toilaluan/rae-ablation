from __future__ import annotations

import torch

from jit_reg.latent_statistics import (
    EndpointLatentNormalizer,
    LatentStatisticsAccumulator,
    latent_statistics_sha256,
    load_latent_statistics,
    save_latent_statistics,
)


def test_separate_endpoint_statistics_and_round_trip(tmp_path) -> None:
    target = torch.tensor([[[[1.0, 3.0]], [[2.0, 6.0]]]])
    semantic = torch.tensor([[[[5.0, 7.0]], [[10.0, 14.0]]]])
    accumulator = LatentStatisticsAccumulator(2)
    accumulator.update(target, semantic)
    statistics = accumulator.finalize(
        vision_fingerprint="vision-id",
        data_fingerprint="data-id",
    )

    assert statistics.count == 2
    assert torch.allclose(statistics.target_mean, torch.tensor([2.0, 4.0]))
    assert torch.allclose(statistics.target_std, torch.tensor([1.0, 2.0]))
    assert torch.allclose(statistics.semantic_mean, torch.tensor([6.0, 12.0]))
    assert torch.allclose(statistics.semantic_std, torch.tensor([1.0, 2.0]))

    path = tmp_path / "statistics.safetensors"
    save_latent_statistics(statistics, path)
    loaded = load_latent_statistics(
        path,
        expected_channels=2,
        expected_vision_fingerprint="vision-id",
        expected_data_fingerprint="data-id",
    )
    assert torch.equal(loaded.target_mean, statistics.target_mean)
    assert torch.equal(loaded.target_std, statistics.target_std)
    assert torch.equal(loaded.semantic_mean, statistics.semantic_mean)
    assert torch.equal(loaded.semantic_std, statistics.semantic_std)
    assert len(latent_statistics_sha256(path)) == 64

    normalizer = EndpointLatentNormalizer(loaded)
    assert torch.allclose(
        normalizer.normalize_target(target),
        (target - statistics.target_mean.reshape(1, 2, 1, 1))
        / statistics.target_std.reshape(1, 2, 1, 1),
    )
    assert torch.allclose(
        normalizer.normalize_semantic(semantic),
        (semantic - statistics.semantic_mean.reshape(1, 2, 1, 1))
        / statistics.semantic_std.reshape(1, 2, 1, 1),
    )
