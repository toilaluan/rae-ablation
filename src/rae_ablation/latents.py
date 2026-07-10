from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass
class LatentStatistics:
    mean: Tensor
    std: Tensor
    count: int

    def normalize(self, latent: Tensor) -> Tensor:
        return (latent - self.mean.to(latent.device, latent.dtype)) / self.std.to(
            latent.device, latent.dtype
        )

    def denormalize(self, latent: Tensor) -> Tensor:
        return latent * self.std.to(latent.device, latent.dtype) + self.mean.to(
            latent.device, latent.dtype
        )

    def save(self, path: str | Path) -> None:
        from safetensors.torch import save_file

        save_file(
            {"mean": self.mean.cpu().float(), "std": self.std.cpu().float()},
            str(path),
            metadata={"count": str(self.count), "population": "true"},
        )

    @classmethod
    def load(cls, path: str | Path) -> LatentStatistics:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            count = int(handle.metadata()["count"])
            return cls(handle.get_tensor("mean"), handle.get_tensor("std"), count)


class ChannelMoments:
    """Float64 population moments over samples and spatial positions."""

    def __init__(self, channels: int) -> None:
        self.total = torch.zeros(channels, dtype=torch.float64)
        self.square_total = torch.zeros(channels, dtype=torch.float64)
        self.count = 0

    def update(self, latent: Tensor) -> None:
        if latent.ndim != 4 or latent.shape[1] != self.total.numel():
            raise ValueError("latent shape does not match accumulator channels")
        values = latent.detach().cpu().to(torch.float64)
        self.total += values.sum(dim=(0, 2, 3))
        self.square_total += values.square().sum(dim=(0, 2, 3))
        self.count += values.shape[0] * values.shape[2] * values.shape[3]

    def distributed_reduce(self, device: torch.device | str) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        packed = torch.cat(
            (
                self.total,
                self.square_total,
                torch.tensor([self.count], dtype=torch.float64),
            )
        ).to(device)
        torch.distributed.all_reduce(packed)
        channels = self.total.numel()
        self.total = packed[:channels].cpu()
        self.square_total = packed[channels : 2 * channels].cpu()
        self.count = int(packed[-1].item())

    def finalize(self) -> LatentStatistics:
        if self.count == 0:
            raise ValueError("cannot finalize empty moments")
        mean = self.total / self.count
        variance = (self.square_total / self.count - mean.square()).clamp_min(0)
        std = variance.sqrt().clamp_min(1e-6)
        return LatentStatistics(mean.float()[:, None, None], std.float()[:, None, None], self.count)
