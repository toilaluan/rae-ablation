from __future__ import annotations

import torch

from cascade_dit.semantic_statistics import (
    SemanticStatisticsAccumulator,
    SemanticTokenNormalizer,
    load_semantic_statistics,
    save_semantic_statistics,
    semantic_statistics_sha256,
)


def test_semantic_statistics_roundtrip_and_normalization(tmp_path) -> None:
    accumulator = SemanticStatisticsAccumulator(2)
    values = torch.tensor([[[[1.0, 3.0]], [[5.0, 7.0]]]])
    accumulator.update(values)
    statistics = accumulator.finalize(
        semantic_encoder_fingerprint="encoder", data_fingerprint="data"
    )
    path = tmp_path / "semantic.safetensors"
    save_semantic_statistics(statistics, path)
    loaded = load_semantic_statistics(
        path,
        expected_channels=2,
        expected_semantic_encoder_fingerprint="encoder",
        expected_data_fingerprint="data",
    )
    assert loaded.channels == 2
    assert len(semantic_statistics_sha256(path)) == 64
    normalized = SemanticTokenNormalizer(loaded).normalize(values)
    assert torch.allclose(normalized.mean((0, 2, 3)), torch.zeros(2), atol=1e-6)
