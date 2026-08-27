from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RNGStreams:
    initialization: torch.Generator
    timesteps: torch.Generator
    noise: torch.Generator
    label_dropout: torch.Generator

    @classmethod
    def create(cls, seed: int, device: str | torch.device = "cpu") -> RNGStreams:
        return cls(
            torch.Generator(device="cpu").manual_seed(seed),
            torch.Generator(device=device).manual_seed(seed + 10_000),
            torch.Generator(device=device).manual_seed(seed + 20_000),
            torch.Generator(device=device).manual_seed(seed + 30_000),
        )

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            name: getattr(self, name).get_state()
            for name in ("initialization", "timesteps", "noise", "label_dropout")
        }
        state["global_cpu"] = torch.random.get_rng_state()
        state["global_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = set(self.state_dict())
        if set(state) != expected:
            raise ValueError("checkpoint RNG streams are incompatible with schema v6")
        for name in expected - {"global_cpu", "global_cuda"}:
            getattr(self, name).set_state(state[name])
        torch.random.set_rng_state(state["global_cpu"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["global_cuda"])
