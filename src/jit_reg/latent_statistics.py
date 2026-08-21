from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn

STATISTICS_SCHEMA_VERSION = 2


def latent_statistics_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LatentStatistics:
    target_mean: Tensor
    target_std: Tensor
    semantic_mean: Tensor
    semantic_std: Tensor
    count: int
    vision_fingerprint: str
    data_fingerprint: str

    @property
    def channels(self) -> int:
        return int(self.target_mean.numel())

    def validate(self) -> None:
        tensors = (
            self.target_mean,
            self.target_std,
            self.semantic_mean,
            self.semantic_std,
        )
        if self.target_mean.ndim != 1 or any(
            value.shape != self.target_mean.shape for value in tensors[1:]
        ):
            raise ValueError(
                "target and semantic latent statistics must be matching vectors"
            )
        if self.count < 2:
            raise ValueError(
                "latent statistics must contain at least two observations per endpoint"
            )
        if not self.vision_fingerprint or not self.data_fingerprint:
            raise ValueError("latent statistics provenance cannot be empty")
        if not all(torch.isfinite(value).all() for value in tensors):
            raise ValueError("latent statistics contain non-finite values")
        if not torch.all(self.target_std > 0) or not torch.all(self.semantic_std > 0):
            raise ValueError("latent statistics standard deviations must be positive")


class EndpointLatentNormalizer(nn.Module):
    def __init__(self, statistics: LatentStatistics) -> None:
        super().__init__()
        statistics.validate()
        for name in ("target_mean", "target_std", "semantic_mean", "semantic_std"):
            self.register_buffer(
                name,
                getattr(statistics, name).float().reshape(1, -1, 1, 1),
            )

    @staticmethod
    def _normalize(tokens: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        if tokens.ndim != 4 or tokens.shape[1] != mean.shape[1]:
            raise ValueError("latent tensor has an incompatible channel dimension")
        normalized = (tokens.float() - mean) / std
        return normalized.to(tokens.dtype)

    def normalize_target(self, tokens: Tensor) -> Tensor:
        return self._normalize(tokens, self.target_mean, self.target_std)

    def normalize_semantic(self, tokens: Tensor) -> Tensor:
        return self._normalize(tokens, self.semantic_mean, self.semantic_std)


class LatentStatisticsAccumulator:
    def __init__(self, channels: int, device: torch.device | str = "cpu") -> None:
        if channels < 1:
            raise ValueError("statistics channel count must be positive")
        shape = (2, channels)
        self.sum = torch.zeros(shape, dtype=torch.float64, device=device)
        self.sum_of_squares = torch.zeros_like(self.sum)
        self.count = torch.zeros(2, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, target: Tensor, semantic: Tensor) -> None:
        for index, values in enumerate((target, semantic)):
            if values.ndim != 4 or values.shape[1] != self.sum.shape[1]:
                raise ValueError("endpoint tensor has an incompatible shape")
            dimensions = (0, 2, 3)
            self.sum[index] += values.sum(dimensions, dtype=torch.float64)
            self.sum_of_squares[index] += values.float().square().sum(
                dimensions, dtype=torch.float64
            )
            self.count[index] += values.shape[0] * values.shape[2] * values.shape[3]

    @torch.no_grad()
    def all_reduce(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.sum)
            torch.distributed.all_reduce(self.sum_of_squares)
            torch.distributed.all_reduce(self.count)

    def finalize(
        self,
        *,
        vision_fingerprint: str,
        data_fingerprint: str,
        epsilon: float = 1e-6,
    ) -> LatentStatistics:
        counts = self.count.to(torch.float64).unsqueeze(1)
        if torch.any(self.count < 2):
            raise ValueError(
                "not enough observations for separate target and semantic statistics"
            )
        if self.count[0] != self.count[1]:
            raise ValueError("target and semantic statistics must use equal observation counts")
        mean = self.sum / counts
        variance = (self.sum_of_squares / counts - mean.square()).clamp_min(0)
        statistics = LatentStatistics(
            target_mean=mean[0].float().cpu(),
            target_std=variance[0].sqrt().clamp_min(epsilon).float().cpu(),
            semantic_mean=mean[1].float().cpu(),
            semantic_std=variance[1].sqrt().clamp_min(epsilon).float().cpu(),
            count=int(self.count[0].item()),
            vision_fingerprint=vision_fingerprint,
            data_fingerprint=data_fingerprint,
        )
        statistics.validate()
        return statistics


def save_latent_statistics(statistics: LatentStatistics, path: str | Path) -> None:
    statistics.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".partial-{os.getpid()}")
    save_file(
        {
            "target_mean": statistics.target_mean.contiguous(),
            "target_std": statistics.target_std.contiguous(),
            "semantic_mean": statistics.semantic_mean.contiguous(),
            "semantic_std": statistics.semantic_std.contiguous(),
        },
        str(temporary),
        metadata={
            "schema_version": str(STATISTICS_SCHEMA_VERSION),
            "count": str(statistics.count),
            "vision_fingerprint": statistics.vision_fingerprint,
            "data_fingerprint": statistics.data_fingerprint,
            "aggregation": "separate-target-semantic-per-channel-population",
        },
    )
    os.replace(temporary, destination)


def load_latent_statistics(
    path: str | Path,
    *,
    expected_channels: int,
    expected_vision_fingerprint: str | None = None,
    expected_data_fingerprint: str | None = None,
) -> LatentStatistics:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"latent statistics file is missing: {source}")
    with safe_open(source, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        if metadata.get("schema_version") != str(STATISTICS_SCHEMA_VERSION):
            raise ValueError(
                "latent statistics use an incompatible schema; regenerate separate "
                "target and semantic statistics"
            )
        statistics = LatentStatistics(
            target_mean=handle.get_tensor("target_mean"),
            target_std=handle.get_tensor("target_std"),
            semantic_mean=handle.get_tensor("semantic_mean"),
            semantic_std=handle.get_tensor("semantic_std"),
            count=int(metadata["count"]),
            vision_fingerprint=metadata.get("vision_fingerprint", ""),
            data_fingerprint=metadata.get("data_fingerprint", ""),
        )
    statistics.validate()
    if statistics.channels != expected_channels:
        raise ValueError(
            f"latent statistics have {statistics.channels} channels, expected {expected_channels}"
        )
    if (
        expected_vision_fingerprint is not None
        and statistics.vision_fingerprint != expected_vision_fingerprint
    ):
        raise ValueError("latent statistics DINO provenance mismatch")
    if (
        expected_data_fingerprint is not None
        and statistics.data_fingerprint != expected_data_fingerprint
    ):
        raise ValueError("latent statistics dataset/preprocessing provenance mismatch")
    return statistics
