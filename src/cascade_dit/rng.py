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
    semantic_timesteps: torch.Generator
    semantic_noise: torch.Generator
    image_timesteps: torch.Generator
    image_noise: torch.Generator
    semantic_reuse_mask: torch.Generator
    label_dropout: torch.Generator

    @classmethod
    def create(cls, seed: int, device: str | torch.device = "cpu") -> RNGStreams:
        offsets = tuple(range(0, 80_000, 10_000))
        shuffle = torch.Generator(device="cpu").manual_seed(seed + offsets[0])
        initialization = torch.Generator(device="cpu").manual_seed(seed + offsets[1])
        semantic_timesteps = torch.Generator(device=device).manual_seed(seed + offsets[2])
        semantic_noise = torch.Generator(device=device).manual_seed(seed + offsets[3])
        image_timesteps = torch.Generator(device=device).manual_seed(seed + offsets[4])
        image_noise = torch.Generator(device=device).manual_seed(seed + offsets[5])
        semantic_reuse_mask = torch.Generator(device=device).manual_seed(seed + offsets[6])
        label_dropout = torch.Generator(device=device).manual_seed(seed + offsets[7])
        return cls(
            shuffle,
            initialization,
            semantic_timesteps,
            semantic_noise,
            image_timesteps,
            image_noise,
            semantic_reuse_mask,
            label_dropout,
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "shuffle": self.shuffle.get_state(),
            "initialization": self.initialization.get_state(),
            "semantic_timesteps": self.semantic_timesteps.get_state(),
            "semantic_noise": self.semantic_noise.get_state(),
            "image_timesteps": self.image_timesteps.get_state(),
            "image_noise": self.image_noise.get_state(),
            "semantic_reuse_mask": self.semantic_reuse_mask.get_state(),
            "label_dropout": self.label_dropout.get_state(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = set(self.state_dict())
        if set(state) != expected:
            raise ValueError(
                "checkpoint RNG streams are incompatible with CascadeDiT schema; "
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
