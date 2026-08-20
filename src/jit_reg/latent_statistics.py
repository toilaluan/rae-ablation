from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn

STATISTICS_SCHEMA_VERSION = 1


def latent_statistics_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LatentStatistics:
    mean: Tensor
    std: Tensor
    count: int
    vision_fingerprint: str
    data_fingerprint: str

    @property
    def channels(self) -> int:
        return int(self.mean.numel())

    def validate(self) -> None:
        if self.mean.ndim != 1 or self.std.shape != self.mean.shape:
            raise ValueError("latent statistics mean and std must be matching vectors")
        if self.count < 2:
            raise ValueError("latent statistics must contain at least two observations")
        if not self.vision_fingerprint or not self.data_fingerprint:
            raise ValueError("latent statistics provenance cannot be empty")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("latent statistics contain non-finite values")
        if not torch.all(self.std > 0):
            raise ValueError("latent statistics standard deviations must be positive")


class SharedLatentNormalizer(nn.Module):
    def __init__(self, statistics: LatentStatistics) -> None:
        super().__init__()
        statistics.validate()
        self.register_buffer("mean", statistics.mean.float().reshape(1, -1, 1, 1))
        self.register_buffer("std", statistics.std.float().reshape(1, -1, 1, 1))

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 4 or tokens.shape[1] != self.mean.shape[1]:
            raise ValueError("latent tensor has an incompatible channel dimension")
        normalized = (tokens.float() - self.mean) / self.std
        return normalized.to(tokens.dtype)


class LatentStatisticsAccumulator:
    def __init__(self, channels: int, device: torch.device | str = "cpu") -> None:
        if channels < 1:
            raise ValueError("statistics channel count must be positive")
        self.sum = torch.zeros(channels, dtype=torch.float64, device=device)
        self.sum_of_squares = torch.zeros_like(self.sum)
        self.count = torch.zeros((), dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, *endpoints: Tensor) -> None:
        for values in endpoints:
            if values.ndim != 4 or values.shape[1] != self.sum.numel():
                raise ValueError("endpoint tensor has an incompatible shape")
            dimensions = (0, 2, 3)
            self.sum += values.sum(dimensions, dtype=torch.float64)
            self.sum_of_squares += values.float().square().sum(
                dimensions, dtype=torch.float64
            )
            self.count += values.shape[0] * values.shape[2] * values.shape[3]

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
        count = int(self.count.item())
        if count < 2:
            raise ValueError("not enough endpoint observations to compute statistics")
        mean = self.sum / count
        variance = (self.sum_of_squares / count - mean.square()).clamp_min(0)
        statistics = LatentStatistics(
            mean=mean.float().cpu(),
            std=variance.sqrt().clamp_min(epsilon).float().cpu(),
            count=count,
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
        {"mean": statistics.mean.contiguous(), "std": statistics.std.contiguous()},
        str(temporary),
        metadata={
            "schema_version": str(STATISTICS_SCHEMA_VERSION),
            "count": str(statistics.count),
            "vision_fingerprint": statistics.vision_fingerprint,
            "data_fingerprint": statistics.data_fingerprint,
            "aggregation": "shared-target-semantic-per-channel-population",
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
            raise ValueError("latent statistics use an incompatible schema")
        statistics = LatentStatistics(
            mean=handle.get_tensor("mean"),
            std=handle.get_tensor("std"),
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
