from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise as adjacent_pairs

import torch
from torch import Tensor, nn


def orthogonality_error(matrix: Tensor) -> float:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("orthogonal matrix must be square")
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    return float(torch.linalg.matrix_norm(matrix.T @ matrix - identity).item())


class StreamingProcrustes:
    """Streaming full-rank orthogonal Procrustes for row-vector token pairs."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("alignment dimension must be positive")
        self.dimension = dimension
        self.cross_covariance = torch.zeros(dimension, dimension, dtype=torch.float64)
        self.count = 0

    def update(self, source: Tensor, target: Tensor) -> None:
        if source.shape != target.shape or source.shape[-1] != self.dimension:
            raise ValueError(f"paired values must have equal [...,{self.dimension}] shapes")
        left = source.detach().reshape(-1, self.dimension).to("cpu", torch.float64)
        right = target.detach().reshape(-1, self.dimension).to("cpu", torch.float64)
        self.cross_covariance.addmm_(left.T, right)
        self.count += left.shape[0]

    def merge(self, other: StreamingProcrustes) -> None:
        if other.dimension != self.dimension:
            raise ValueError("alignment dimensions differ")
        self.cross_covariance += other.cross_covariance
        self.count += other.count

    def solve(
        self,
        *,
        identity: bool = False,
        driver: str | None = None,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        if self.count < 1:
            raise ValueError("cannot fit Procrustes without paired tokens")
        if identity:
            return torch.eye(self.dimension)
        target = torch.device(device)
        covariance = self.cross_covariance.to(target)
        options = {"driver": driver} if target.type == "cuda" and driver is not None else {}
        u, _, vh = torch.linalg.svd(covariance, full_matrices=False, **options)
        return (u @ vh).float().cpu()


def alignment_mse(source: Tensor, target: Tensor, matrix: Tensor) -> float:
    if source.shape != target.shape or source.shape[-1] != matrix.shape[0]:
        raise ValueError("held-out values and alignment matrix are incompatible")
    return float((source.float() @ matrix.float() - target.float()).square().mean().item())


def cumulative_projectors(
    pairwise: Mapping[str, Tensor],
    stages: Iterable[str] = ("dino12", "dino8", "dino4", "rgb"),
) -> dict[str, Tensor]:
    names = tuple(stages)
    if len(names) < 1:
        raise ValueError("at least one stage is required")
    first = next(iter(pairwise.values()), None)
    if first is None:
        raise ValueError("pairwise alignments cannot be empty")
    cumulative = torch.eye(first.shape[0], dtype=first.dtype, device=first.device)
    result = {names[0]: cumulative.clone()}
    for source, target in adjacent_pairs(names):
        key = f"{source}->{target}"
        if key not in pairwise:
            raise ValueError(f"missing pairwise alignment {key}")
        matrix = pairwise[key]
        if matrix.shape != cumulative.shape:
            raise ValueError(f"alignment {key} has an incompatible shape")
        cumulative = cumulative @ matrix
        result[target] = cumulative.clone()
    return result


@dataclass(frozen=True)
class AlignmentDiagnostics:
    heldout_mse: float
    orthogonality_error: float
    count: int
    parameterization: str = "full-rank orthogonal initialization alignment"


class OrthogonalProjector(nn.Module):
    """Keeps a stage transform outside the nonlinear canonical denoiser."""

    def __init__(self, matrix: Tensor) -> None:
        super().__init__()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("projector must be square")
        self.register_buffer("matrix", matrix.detach().float().clone())

    def to_canonical(self, stage_tokens: Tensor) -> Tensor:
        return stage_tokens @ self.matrix.T.to(stage_tokens)

    def from_canonical(self, canonical_tokens: Tensor) -> Tensor:
        return canonical_tokens @ self.matrix.to(canonical_tokens)
