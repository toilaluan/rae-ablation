from __future__ import annotations

import torch

from jit_reg.latent_statistics import (
    LatentStatisticsAccumulator,
    SharedLatentNormalizer,
    latent_statistics_sha256,
    load_latent_statistics,
    save_latent_statistics,
)


def test_shared_statistics_pool_endpoints_and_round_trip(tmp_path) -> None:
    target = torch.tensor([[[[1.0, 3.0]], [[2.0, 6.0]]]])
    semantic = torch.tensor([[[[5.0, 7.0]], [[10.0, 14.0]]]])
    accumulator = LatentStatisticsAccumulator(2)
    accumulator.update(target, semantic)
    statistics = accumulator.finalize(
        vision_fingerprint="vision-id",
        data_fingerprint="data-id",
    )

    assert statistics.count == 4
    assert torch.allclose(statistics.mean, torch.tensor([4.0, 8.0]))
    assert torch.allclose(statistics.std, torch.tensor([5.0, 20.0]).sqrt())

    path = tmp_path / "statistics.safetensors"
    save_latent_statistics(statistics, path)
    loaded = load_latent_statistics(
        path,
        expected_channels=2,
        expected_vision_fingerprint="vision-id",
        expected_data_fingerprint="data-id",
    )
    assert torch.equal(loaded.mean, statistics.mean)
    assert torch.equal(loaded.std, statistics.std)
    assert len(latent_statistics_sha256(path)) == 64

    normalizer = SharedLatentNormalizer(loaded)
    assert torch.allclose(
        normalizer(target),
        (target - statistics.mean.reshape(1, 2, 1, 1))
        / statistics.std.reshape(1, 2, 1, 1),
    )
