from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .objective import per_sample_repa_cosine_loss, per_sample_velocity_mse


class ExplorationResult(NamedTuple):
    noisy: Tensor
    noise: Tensor
    indices: Tensor
    early_candidate_zero_score: Tensor | None
    early_best_score: Tensor | None
    full_first_finalist_score: Tensor | None
    full_selected_score: Tensor | None


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _gather_candidates(values: Tensor, indices: Tensor) -> Tensor:
    if values.ndim < 2 or values.shape[0] != indices.shape[0]:
        raise ValueError("candidate tensors must have shape [B, K, ...]")
    squeeze = indices.ndim == 1
    if squeeze:
        indices = indices[:, None]
    if indices.ndim != 2:
        raise ValueError("candidate indices must have shape [B] or [B, M]")
    view = indices.reshape(indices.shape[0], indices.shape[1], *([1] * (values.ndim - 2)))
    gathered = values.gather(
        1,
        view.expand(-1, -1, *values.shape[2:]),
    )
    return gathered.squeeze(1) if squeeze else gathered


def _validate_chunk_size(chunk_size: int | None, candidates: int, stage: str) -> int:
    chunk = chunk_size or candidates
    if chunk < 1 or chunk > candidates or candidates % chunk:
        raise ValueError(f"{stage} candidate chunk size must divide {candidates}")
    return chunk


def _score_candidate_chunks(
    noisy_candidates: Tensor,
    noise_candidates: Tensor,
    timestep: Tensor,
    labels: Tensor,
    *,
    chunk_size: int | None,
    score_chunk: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    stage: str,
) -> Tensor:
    batch, candidates = noisy_candidates.shape[:2]
    chunk = _validate_chunk_size(chunk_size, candidates, stage)
    scores: list[Tensor] = []
    for start in range(0, candidates, chunk):
        stop = start + chunk
        noisy = noisy_candidates[:, start:stop].flatten(0, 1)
        noise = noise_candidates[:, start:stop].flatten(0, 1)
        repeated_timestep = timestep.repeat_interleave(chunk)
        repeated_labels = labels.repeat_interleave(chunk)
        score = score_chunk(noisy, noise, repeated_timestep, repeated_labels)
        if score.shape != (batch * chunk,):
            raise ValueError("candidate scorer must return one score per candidate")
        scores.append(score.reshape(batch, chunk))
    return torch.cat(scores, dim=1)


@torch.no_grad()
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
    early_selector: str,
    repa_target: Tensor | None = None,
    early_candidate_chunk_size: int | None = None,
    full_candidate_chunk_size: int | None = None,
    velocity_denominator_floor: float = 0.05,
) -> ExplorationResult:
    if noisy_candidates.shape != noise_candidates.shape or noisy_candidates.ndim != 5:
        raise ValueError("noisy and noise candidates must have equal [B, K, C, H, W] shapes")
    batch, candidates = noisy_candidates.shape[:2]
    if clean.shape != noisy_candidates.shape[:1] + noisy_candidates.shape[2:]:
        raise ValueError("clean targets must have shape [B, C, H, W]")
    if timestep.shape != (batch,) or labels.shape != (batch,):
        raise ValueError("timestep and labels must have shape [B]")
    if k_early < 1 or k_full < 1 or candidates != k_early * k_full:
        raise ValueError("candidate count must equal k_early * k_full")
    if early_selector not in {"none", "early_prediction", "repa"}:
        raise ValueError("early_selector must be none, early_prediction, or repa")
    if (k_early == 1) != (early_selector == "none"):
        raise ValueError("early selection is required exactly when k_early > 1")
    if early_selector == "repa" and repa_target is None:
        raise ValueError("REPA selection requires a target")

    base_model = _unwrap(model)
    candidate_indices = torch.arange(candidates, device=clean.device).expand(batch, -1)
    early_candidate_zero_score: Tensor | None = None
    early_best_score: Tensor | None = None

    if k_early > 1:
        if early_selector == "early_prediction":
            forward_early = getattr(base_model, "forward_early_prediction", None)
            if not callable(forward_early):
                raise TypeError("model does not provide forward_early_prediction")

            def score_early(
                noisy: Tensor,
                noise: Tensor,
                repeated_timestep: Tensor,
                repeated_labels: Tensor,
            ) -> Tensor:
                prediction = forward_early(noisy, repeated_timestep, repeated_labels)
                repeated_clean = clean.repeat_interleave(noisy.shape[0] // batch, dim=0)
                return per_sample_velocity_mse(
                    repeated_clean,
                    noise,
                    noisy,
                    repeated_timestep,
                    prediction,
                    floor=velocity_denominator_floor,
                )

        else:
            forward_repa = getattr(base_model, "forward_repa", None)
            if not callable(forward_repa):
                raise TypeError("model does not provide forward_repa")
            assert repa_target is not None

            def score_early(
                noisy: Tensor,
                noise: Tensor,
                repeated_timestep: Tensor,
                repeated_labels: Tensor,
            ) -> Tensor:
                del noise
                prediction = forward_repa(noisy, repeated_timestep, repeated_labels)
                repeated_target = repa_target.repeat_interleave(
                    noisy.shape[0] // batch,
                    dim=0,
                )
                return per_sample_repa_cosine_loss(prediction, repeated_target)

        early_scores = _score_candidate_chunks(
            noisy_candidates,
            noise_candidates,
            timestep,
            labels,
            chunk_size=early_candidate_chunk_size,
            score_chunk=score_early,
            stage="early",
        )
        early_candidate_zero_score = early_scores[:, 0]
        early_order = torch.argsort(early_scores, dim=1, stable=True)
        finalist_indices = early_order[:, :k_full]
        early_best_score = early_scores.gather(1, finalist_indices)[:, 0]
        finalist_noisy = _gather_candidates(noisy_candidates, finalist_indices)
        finalist_noise = _gather_candidates(noise_candidates, finalist_indices)
    else:
        finalist_indices = candidate_indices
        finalist_noisy = noisy_candidates
        finalist_noise = noise_candidates

    full_first_finalist_score: Tensor | None = None
    full_selected_score: Tensor | None = None
    if k_full > 1:

        def score_full(
            noisy: Tensor,
            noise: Tensor,
            repeated_timestep: Tensor,
            repeated_labels: Tensor,
        ) -> Tensor:
            output = model(noisy, repeated_timestep, repeated_labels, return_repa=False)
            repeated_clean = clean.repeat_interleave(noisy.shape[0] // batch, dim=0)
            return per_sample_velocity_mse(
                repeated_clean,
                noise,
                noisy,
                repeated_timestep,
                output.x_prediction,
                floor=velocity_denominator_floor,
            )

        full_scores = _score_candidate_chunks(
            finalist_noisy,
            finalist_noise,
            timestep,
            labels,
            chunk_size=full_candidate_chunk_size,
            score_chunk=score_full,
            stage="full",
        )
        full_first_finalist_score = full_scores[:, 0]
        full_selected_score = full_scores.min(dim=1).values
        tied_original_indices = torch.where(
            full_scores == full_selected_score[:, None],
            finalist_indices,
            candidates,
        )
        selected_original_index = tied_original_indices.min(dim=1).values
        relative_index = (finalist_indices == selected_original_index[:, None]).to(
            torch.int64
        ).argmax(dim=1)
    else:
        relative_index = torch.zeros(batch, dtype=torch.long, device=clean.device)

    selected_indices = finalist_indices.gather(1, relative_index[:, None]).squeeze(1)
    return ExplorationResult(
        _gather_candidates(noisy_candidates, selected_indices),
        _gather_candidates(noise_candidates, selected_indices),
        selected_indices,
        early_candidate_zero_score,
        early_best_score,
        full_first_finalist_score,
        full_selected_score,
    )
