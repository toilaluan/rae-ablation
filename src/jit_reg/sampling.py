from __future__ import annotations

import math
from itertools import pairwise
from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn

from .objective import shift_timestep, x_to_velocity

Solver = Literal["euler", "heun"]


class EndpointSample(NamedTuple):
    endpoint: Tensor
    nfe: int


def _prediction(
    model: nn.Module, x: Tensor, timestep: Tensor, semantic: Tensor
) -> Tensor:
    return model(x, timestep, semantic)


def _schedule(steps: int, timestep_shift: float, device: torch.device) -> Tensor:
    return shift_timestep(torch.linspace(1, 0, steps + 1, device=device), timestep_shift)


def _validate_condition(semantic: Tensor, noise: Tensor) -> None:
    if semantic.shape != noise.shape:
        raise ValueError("semantic and initial noise tensors must have matching shapes")


def one_step_sample(
    model: nn.Module, semantic: Tensor, noise: Tensor
) -> EndpointSample:
    _validate_condition(semantic, noise)
    timestep = torch.ones(semantic.shape[0], device=semantic.device, dtype=torch.float32)
    endpoint = _prediction(model, noise, timestep, semantic)
    return EndpointSample(endpoint, 1)


def flow_sample(
    model: nn.Module,
    semantic: Tensor,
    noise: Tensor,
    *,
    steps: int = 40,
    timestep_shift: float = 4.0,
    solver: Solver = "euler",
) -> EndpointSample:
    """Integrate standard-normal noise into a semantic-conditioned endpoint."""
    if steps < 1 or solver not in {"euler", "heun"}:
        raise ValueError("sampling requires positive steps and euler or heun")
    if not math.isfinite(timestep_shift) or timestep_shift <= 0:
        raise ValueError("timestep shift must be finite and positive")
    _validate_condition(semantic, noise)
    x = noise
    nfe = 0
    schedule = _schedule(steps, timestep_shift, semantic.device)
    for current, following in pairwise(schedule):
        timestep = current.expand(semantic.shape[0])
        endpoint = _prediction(model, x, timestep, semantic)
        nfe += 1
        velocity = x_to_velocity(x, endpoint, timestep).to(x.dtype)
        delta = following - current
        proposal = x + delta * velocity
        if solver == "heun" and following > 0:
            next_timestep = following.expand(semantic.shape[0])
            corrected_endpoint = _prediction(model, proposal, next_timestep, semantic)
            nfe += 1
            corrected_velocity = x_to_velocity(
                proposal, corrected_endpoint, next_timestep
            ).to(x.dtype)
            x = x + delta * 0.5 * (velocity + corrected_velocity)
        else:
            x = proposal
    return EndpointSample(x, nfe)


ode_sample = flow_sample


@torch.inference_mode()
def sample(*args: object, **kwargs: object) -> EndpointSample:
    return flow_sample(*args, **kwargs)  # type: ignore[arg-type]
