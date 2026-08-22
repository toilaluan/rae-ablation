from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn

SEMANTIC_STATISTICS_SCHEMA_VERSION = 1


def semantic_statistics_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SemanticStatistics:
    mean: Tensor
    std: Tensor
    count: int
    semantic_encoder_fingerprint: str
    data_fingerprint: str

    @property
    def channels(self) -> int:
        return int(self.mean.numel())

    def validate(self) -> None:
        if self.mean.ndim != 1 or self.std.shape != self.mean.shape:
            raise ValueError("semantic statistics must be matching channel vectors")
        if self.count < 2:
            raise ValueError("semantic statistics require at least two observations")
        if not self.semantic_encoder_fingerprint or not self.data_fingerprint:
            raise ValueError("semantic statistics provenance cannot be empty")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("semantic statistics contain non-finite values")
        if not torch.all(self.std > 0):
            raise ValueError("semantic standard deviations must be positive")


class SemanticTokenNormalizer(nn.Module):
    def __init__(self, statistics: SemanticStatistics) -> None:
        super().__init__()
        statistics.validate()
        self.register_buffer("mean", statistics.mean.float().reshape(1, -1, 1, 1))
        self.register_buffer("std", statistics.std.float().reshape(1, -1, 1, 1))

    def normalize(self, semantic_tokens: Tensor) -> Tensor:
        if semantic_tokens.ndim != 4 or semantic_tokens.shape[1] != self.mean.shape[1]:
            raise ValueError("semantic tokens have an incompatible channel dimension")
        result = (semantic_tokens.float() - self.mean) / self.std
        return result.to(semantic_tokens.dtype)


class SemanticStatisticsAccumulator:
    def __init__(self, channels: int, device: torch.device | str = "cpu") -> None:
        if channels < 1:
            raise ValueError("statistics channel count must be positive")
        self.sum = torch.zeros(channels, dtype=torch.float64, device=device)
        self.sum_of_squares = torch.zeros_like(self.sum)
        self.count = torch.zeros((), dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, semantic_tokens: Tensor) -> None:
        if semantic_tokens.ndim != 4 or semantic_tokens.shape[1] != self.sum.numel():
            raise ValueError("semantic token tensor has an incompatible shape")
        dimensions = (0, 2, 3)
        self.sum += semantic_tokens.sum(dimensions, dtype=torch.float64)
        self.sum_of_squares += semantic_tokens.float().square().sum(
            dimensions, dtype=torch.float64
        )
        self.count += semantic_tokens.shape[0] * semantic_tokens.shape[2] * semantic_tokens.shape[3]

    @torch.no_grad()
    def all_reduce(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.sum)
            torch.distributed.all_reduce(self.sum_of_squares)
            torch.distributed.all_reduce(self.count)

    def finalize(
        self,
        *,
        semantic_encoder_fingerprint: str,
        data_fingerprint: str,
        epsilon: float = 1e-6,
    ) -> SemanticStatistics:
        if self.count < 2:
            raise ValueError("not enough observations for semantic statistics")
        count = self.count.to(torch.float64)
        mean = self.sum / count
        variance = (self.sum_of_squares / count - mean.square()).clamp_min(0)
        statistics = SemanticStatistics(
            mean.float().cpu(),
            variance.sqrt().clamp_min(epsilon).float().cpu(),
            int(self.count.item()),
            semantic_encoder_fingerprint,
            data_fingerprint,
        )
        statistics.validate()
        return statistics


def save_semantic_statistics(
    statistics: SemanticStatistics, path: str | Path
) -> None:
    statistics.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".partial-{os.getpid()}")
    save_file(
        {"mean": statistics.mean.contiguous(), "std": statistics.std.contiguous()},
        str(temporary),
        metadata={
            "schema_version": str(SEMANTIC_STATISTICS_SCHEMA_VERSION),
            "count": str(statistics.count),
            "semantic_encoder_fingerprint": statistics.semantic_encoder_fingerprint,
            "data_fingerprint": statistics.data_fingerprint,
            "aggregation": "semantic-only-per-channel-population",
        },
    )
    os.replace(temporary, destination)


def load_semantic_statistics(
    path: str | Path,
    *,
    expected_channels: int,
    expected_semantic_encoder_fingerprint: str | None = None,
    expected_data_fingerprint: str | None = None,
) -> SemanticStatistics:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"semantic statistics file is missing: {source}")
    with safe_open(source, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        if metadata.get("schema_version") != str(SEMANTIC_STATISTICS_SCHEMA_VERSION):
            raise ValueError(
                "semantic statistics use an incompatible schema; regenerate them"
            )
        statistics = SemanticStatistics(
            handle.get_tensor("mean"),
            handle.get_tensor("std"),
            int(metadata["count"]),
            metadata.get("semantic_encoder_fingerprint", ""),
            metadata.get("data_fingerprint", ""),
        )
    statistics.validate()
    if statistics.channels != expected_channels:
        raise ValueError(
            f"semantic statistics have {statistics.channels} channels, expected {expected_channels}"
        )
    if (
        expected_semantic_encoder_fingerprint is not None
        and statistics.semantic_encoder_fingerprint != expected_semantic_encoder_fingerprint
    ):
        raise ValueError("semantic statistics encoder provenance mismatch")
    if (
        expected_data_fingerprint is not None
        and statistics.data_fingerprint != expected_data_fingerprint
    ):
        raise ValueError("semantic statistics dataset/preprocessing provenance mismatch")
    return statistics
