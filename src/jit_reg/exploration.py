from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from .objective import per_sample_repa_cosine_loss, per_sample_velocity_mse


class ExplorationResult(NamedTuple):
    noisy: Tensor
    noise: Tensor
    indices: Tensor
    selected_score: Tensor
    candidate_zero_score: Tensor


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _gather_candidate(values: Tensor, indices: Tensor) -> Tensor:
    if values.ndim < 2 or values.shape[0] != indices.shape[0]:
        raise ValueError("candidate tensors must have shape [B, K, ...]")
    view = indices.reshape(indices.shape[0], 1, *([1] * (values.ndim - 2)))
    return values.gather(1, view.expand(-1, 1, *values.shape[2:])).squeeze(1)


@torch.no_grad()
def select_candidates(
    model: nn.Module,
    noisy_candidates: Tensor,
    noise_candidates: Tensor,
    clean: Tensor,
    timestep: Tensor,
    labels: Tensor,
    *,
    selector: str,
    repa_target: Tensor | None = None,
    candidate_chunk_size: int | None = None,
    velocity_denominator_floor: float = 0.05,
) -> ExplorationResult:
    if noisy_candidates.shape != noise_candidates.shape or noisy_candidates.ndim != 5:
        raise ValueError("noisy and noise candidates must have equal [B, K, C, H, W] shapes")
    batch, k = noisy_candidates.shape[:2]
    if clean.shape != noisy_candidates.shape[:1] + noisy_candidates.shape[2:]:
        raise ValueError("clean targets must have shape [B, C, H, W]")
    if timestep.shape != (batch,) or labels.shape != (batch,):
        raise ValueError("timestep and labels must have shape [B]")
    if selector not in {"final_velocity", "repa"}:
        raise ValueError("selector must be final_velocity or repa")
    if selector == "repa" and repa_target is None:
        raise ValueError("REPA selection requires a target")
    chunk = candidate_chunk_size or k
    if chunk < 1 or chunk > k or k % chunk:
        raise ValueError("candidate chunk size must divide K")

    best_score = torch.full((batch,), torch.inf, device=clean.device)
    best_index = torch.zeros(batch, dtype=torch.long, device=clean.device)
    candidate_zero_score: Tensor | None = None
    base_model = _unwrap(model)
    for start in range(0, k, chunk):
        stop = start + chunk
        noisy = noisy_candidates[:, start:stop].flatten(0, 1)
        noise = noise_candidates[:, start:stop].flatten(0, 1)
        repeated_timestep = timestep.repeat_interleave(chunk)
        repeated_labels = labels.repeat_interleave(chunk)
        if selector == "final_velocity":
            output = model(noisy, repeated_timestep, repeated_labels, return_repa=False)
            repeated_clean = clean.repeat_interleave(chunk, dim=0)
            score = per_sample_velocity_mse(
                repeated_clean,
                noise,
                noisy,
                repeated_timestep,
                output.x_prediction,
                floor=velocity_denominator_floor,
            ).reshape(batch, chunk)
        else:
            forward_repa = getattr(base_model, "forward_repa", None)
            if not callable(forward_repa):
                raise TypeError("model does not provide forward_repa")
            prediction = forward_repa(noisy, repeated_timestep, repeated_labels)
            assert repa_target is not None
            score = per_sample_repa_cosine_loss(
                prediction,
                repa_target.repeat_interleave(chunk, dim=0),
            ).reshape(batch, chunk)
        if start == 0:
            candidate_zero_score = score[:, 0]
        chunk_score, relative_index = score.min(dim=1)
        replace = chunk_score < best_score
        best_score = torch.where(replace, chunk_score, best_score)
        best_index = torch.where(replace, relative_index + start, best_index)

    assert candidate_zero_score is not None
    return ExplorationResult(
        _gather_candidate(noisy_candidates, best_index),
        _gather_candidate(noise_candidates, best_index),
        best_index,
        best_score,
        candidate_zero_score,
    )
