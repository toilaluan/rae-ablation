from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .objective import per_sample_velocity_mse


class ExplorationResult(NamedTuple):
    noisy: Tensor
    noise: Tensor
    indices: Tensor
    early_candidate_zero_score: Tensor | None
    early_best_score: Tensor | None
    full_first_finalist_score: Tensor | None
    full_selected_score: Tensor | None


def _gather_candidates(values: Tensor, indices: Tensor) -> Tensor:
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch[:, None], indices] if indices.ndim == 2 else values[batch, indices]


def _model_method(model: nn.Module, name: str) -> Callable[..., Tensor]:
    method = getattr(model, name, None)
    if not callable(method) and hasattr(model, "module"):
        method = getattr(model.module, name, None)
    if not callable(method):
        raise TypeError(f"model does not provide {name}")
    return method


@torch.no_grad()
def _score_chunks(
    noisy: Tensor,
    noise: Tensor,
    clean: Tensor,
    timestep: Tensor,
    labels: Tensor,
    forward: Callable[[Tensor, Tensor, Tensor], Tensor],
    chunk_size: int | None,
    floor: float,
) -> Tensor:
    batch, candidates = noisy.shape[:2]
    chunk = chunk_size or candidates
    if chunk < 1 or candidates % chunk:
        raise ValueError("candidate chunk size must divide its stage candidate count")
    scores: list[Tensor] = []
    for start in range(0, candidates, chunk):
        stop = start + chunk
        flat_noisy = noisy[:, start:stop].flatten(0, 1)
        flat_noise = noise[:, start:stop].flatten(0, 1)
        repeated_t = timestep.repeat_interleave(chunk)
        repeated_labels = labels.repeat_interleave(chunk)
        repeated_clean = clean.repeat_interleave(chunk, dim=0)
        prediction = forward(flat_noisy, repeated_t, repeated_labels)
        scores.append(
            per_sample_velocity_mse(
                repeated_clean,
                flat_noise,
                flat_noisy,
                repeated_t,
                prediction,
                floor=floor,
            ).reshape(batch, chunk)
        )
    return torch.cat(scores, dim=1)


def select_candidates(
    model: nn.Module,
    noisy_candidates: Tensor,
    noise_candidates: Tensor,
    clean: Tensor,
    timestep: Tensor,
    labels: Tensor,
    *,
    k_early: int,
    k_full: int,
    early_candidate_chunk_size: int | None = None,
    full_candidate_chunk_size: int | None = None,
    velocity_denominator_floor: float = 0.05,
) -> ExplorationResult:
    """Use the DDT base prefix globally, then its full path on the finalists."""
    if k_early < 1 or k_full < 1:
        raise ValueError("k_early and k_full must be positive")
    k_total = k_early * k_full
    if noisy_candidates.shape != noise_candidates.shape:
        raise ValueError("noisy and noise candidates must have equal shapes")
    if noisy_candidates.ndim != clean.ndim + 1:
        raise ValueError("candidates must have shape [B, K, ...]")
    batch = clean.shape[0]
    if noisy_candidates.shape[:2] != (batch, k_total):
        raise ValueError("candidate count must equal k_early * k_full")
    if timestep.shape != (batch,) or labels.shape != (batch,):
        raise ValueError("timestep and labels must have shape [B]")

    original = torch.arange(k_total, device=clean.device).expand(batch, -1)
    early_zero: Tensor | None = None
    early_best: Tensor | None = None
    if k_early > 1:
        early_scores = _score_chunks(
            noisy_candidates,
            noise_candidates,
            clean,
            timestep,
            labels,
            _model_method(model, "forward_base_prediction"),
            early_candidate_chunk_size,
            velocity_denominator_floor,
        )
        early_zero = early_scores[:, 0]
        order = torch.argsort(early_scores, dim=1, stable=True)
        finalist_indices = order[:, :k_full]
        early_best = early_scores.gather(1, finalist_indices)[:, 0]
        finalist_noisy = _gather_candidates(noisy_candidates, finalist_indices)
        finalist_noise = _gather_candidates(noise_candidates, finalist_indices)
    else:
        finalist_indices = original
        finalist_noisy = noisy_candidates
        finalist_noise = noise_candidates

    full_first: Tensor | None = None
    full_selected: Tensor | None = None
    if k_full > 1:
        full_scores = _score_chunks(
            finalist_noisy,
            finalist_noise,
            clean,
            timestep,
            labels,
            _model_method(model, "forward_full_prediction"),
            full_candidate_chunk_size,
            velocity_denominator_floor,
        )
        full_first = full_scores[:, 0]
        full_selected = full_scores.min(dim=1).values
        sentinel = torch.full_like(finalist_indices, k_total)
        tied_original = torch.where(
            full_scores == full_selected[:, None], finalist_indices, sentinel
        )
        selected_original = tied_original.min(dim=1).values
        relative = (finalist_indices == selected_original[:, None]).long().argmax(dim=1)
    else:
        relative = torch.zeros(batch, dtype=torch.long, device=clean.device)
    selected_indices = finalist_indices.gather(1, relative[:, None]).squeeze(1)
    return ExplorationResult(
        _gather_candidates(noisy_candidates, selected_indices),
        _gather_candidates(noise_candidates, selected_indices),
        selected_indices,
        early_zero,
        early_best,
        full_first,
        full_selected,
    )
