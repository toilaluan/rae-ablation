from __future__ import annotations

import torch

from cascade_dit.alignment import (
    StreamingProcrustes,
    alignment_mse,
    cumulative_projectors,
    orthogonality_error,
)


def _orthogonal(dimension: int, seed: int) -> torch.Tensor:
    values = torch.randn(dimension, dimension, generator=torch.Generator().manual_seed(seed))
    return torch.linalg.qr(values).Q


def test_streaming_procrustes_recovers_known_full_rank_alignment() -> None:
    expected = _orthogonal(12, 3)
    source = torch.randn(500, 12, generator=torch.Generator().manual_seed(8))
    target = source @ expected
    fit = StreamingProcrustes(12)
    fit.update(source[:200], target[:200])
    fit.update(source[200:], target[200:])
    actual = fit.solve()
    assert torch.allclose(actual, expected, atol=2e-5)
    assert alignment_mse(source, target, actual) < 1e-10
    assert orthogonality_error(actual) < 1e-5


def test_cumulative_projectors_preserve_orthogonality_and_order() -> None:
    pairwise = {
        "dino12->dino8": _orthogonal(8, 1),
        "dino8->dino4": _orthogonal(8, 2),
        "dino4->rgb": _orthogonal(8, 3),
    }
    cumulative = cumulative_projectors(pairwise)
    assert torch.allclose(
        cumulative["rgb"],
        pairwise["dino12->dino8"] @ pairwise["dino8->dino4"] @ pairwise["dino4->rgb"],
    )
    assert all(orthogonality_error(matrix) < 2e-5 for matrix in cumulative.values())
