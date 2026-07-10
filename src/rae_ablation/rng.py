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
        for name, value in state.items():
            getattr(self, name).set_state(value)


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
