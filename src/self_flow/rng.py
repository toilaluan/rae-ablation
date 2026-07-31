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
    dual_timesteps: torch.Generator
    token_masks: torch.Generator
    noise: torch.Generator
    label_dropout: torch.Generator
    sigreg: torch.Generator

    @classmethod
    def create(cls, seed: int, device: str | torch.device = "cpu") -> RNGStreams:
        offsets = (0, 10_000, 20_000, 25_000, 27_000, 30_000, 50_000, 60_000)
        shuffle = torch.Generator(device="cpu").manual_seed(seed + offsets[0])
        initialization = torch.Generator(device="cpu").manual_seed(seed + offsets[1])
        timesteps = torch.Generator(device=device).manual_seed(seed + offsets[2])
        dual_timesteps = torch.Generator(device=device).manual_seed(seed + offsets[3])
        token_masks = torch.Generator(device=device).manual_seed(seed + offsets[4])
        noise = torch.Generator(device=device).manual_seed(seed + offsets[5])
        label_dropout = torch.Generator(device=device).manual_seed(seed + offsets[6])
        sigreg = torch.Generator(device=device).manual_seed(seed + offsets[7])
        return cls(
            shuffle,
            initialization,
            timesteps,
            dual_timesteps,
            token_masks,
            noise,
            label_dropout,
            sigreg,
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "shuffle": self.shuffle.get_state(),
            "initialization": self.initialization.get_state(),
            "timesteps": self.timesteps.get_state(),
            "dual_timesteps": self.dual_timesteps.get_state(),
            "token_masks": self.token_masks.get_state(),
            "noise": self.noise.get_state(),
            "label_dropout": self.label_dropout.get_state(),
            "sigreg": self.sigreg.get_state(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = set(self.state_dict())
        if set(state) != expected:
            raise ValueError(
                "checkpoint RNG streams are incompatible with schema v11; "
                f"expected {sorted(expected)}"
            )
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
