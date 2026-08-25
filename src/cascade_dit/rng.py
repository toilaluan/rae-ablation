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
    label_dropout: torch.Generator
    sampling: torch.Generator

    @classmethod
    def create(cls, seed: int, device: str | torch.device = "cpu") -> RNGStreams:
        offsets = range(0, 60_000, 10_000)
        generators = [
            torch.Generator(device="cpu" if index < 2 else device).manual_seed(seed + offset)
            for index, offset in enumerate(offsets)
        ]
        return cls(*generators)

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            name: getattr(self, name).get_state()
            for name in ("shuffle", "initialization", "timesteps", "noise", "label_dropout", "sampling")
        }
        state["global_cpu"] = torch.random.get_rng_state()
        state["global_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = set(self.state_dict())
        if set(state) != expected:
            raise ValueError("checkpoint RNG streams are incompatible with schema v4")
        for name in expected - {"global_cpu", "global_cuda"}:
            getattr(self, name).set_state(state[name])
        torch.random.set_rng_state(state["global_cpu"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["global_cuda"])


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
