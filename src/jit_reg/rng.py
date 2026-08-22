from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch


@dataclass
class RNGStreams:
    shuffle: torch.Generator
    initialization: torch.Generator
    timesteps: torch.Generator
    noise: torch.Generator

    @classmethod
    def create(cls, seed: int, device: str | torch.device = "cpu") -> RNGStreams:
        offsets = (0, 10_000, 20_000, 30_000)
        shuffle = torch.Generator(device="cpu").manual_seed(seed + offsets[0])
        initialization = torch.Generator(device="cpu").manual_seed(seed + offsets[1])
        timesteps = torch.Generator(device=device).manual_seed(seed + offsets[2])
        noise = torch.Generator(device=device).manual_seed(seed + offsets[3])
        return cls(shuffle, initialization, timesteps, noise)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "shuffle": self.shuffle.get_state(),
            "initialization": self.initialization.get_state(),
            "timesteps": self.timesteps.get_state(),
            "noise": self.noise.get_state(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = set(self.state_dict())
        if set(state) != expected:
            raise ValueError(
                "checkpoint RNG streams are incompatible with schema v18; "
                f"expected {sorted(expected)}"
            )
        for name, value in state.items():
            getattr(self, name).set_state(value)


def indexed_standard_normal(
    indices: torch.Tensor,
    sample_shape: tuple[int, ...],
    seed: int,
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate per-index noise independent of batching and distributed layout."""
    if indices.ndim != 1 or torch.any(indices < 0):
        raise ValueError("noise indices must be a non-negative one-dimensional tensor")
    if seed < 0 or any(dimension < 1 for dimension in sample_shape):
        raise ValueError("noise seed must be non-negative and sample dimensions positive")
    samples = []
    for index in indices.detach().cpu().tolist():
        generator = torch.Generator(device="cpu").manual_seed(seed + int(index))
        samples.append(
            torch.randn(sample_shape, generator=generator, dtype=torch.float32)
        )
    return torch.stack(samples).to(device=device, dtype=dtype)


@contextmanager
def preserve_global_rng() -> Iterator[None]:
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def preserving_global_rng(function: Any) -> Any:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with preserve_global_rng():
            return function(*args, **kwargs)

    return wrapped
