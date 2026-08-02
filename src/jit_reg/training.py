from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate

from .checkpoint import EMA, CheckpointManager, should_checkpoint
from .config import ExperimentConfig
from .exploration import select_candidates
from .flops import model_flops_per_step
from .model.tokens import classifier_free_labels
from .objective import ddt_losses, noisy_interpolate, sample_timesteps
from .optim import CompositeOptimizer, learning_rate_at_epoch, set_learning_rate
from .rng import RNGStreams
from .token_cache import TokenCache
from .validation import should_validate
from .wandb_utils import RunLogger


class TrainBatch(NamedTuple):
    tokens: Tensor
    labels: Tensor
    indices: Tensor


def prepare_train_batch(
    batch: TrainBatch,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> TrainBatch:
    return TrainBatch(
        batch.tokens.to(device, non_blocking=non_blocking),
        batch.labels.to(device, dtype=torch.long, non_blocking=non_blocking),
        batch.indices.to(device, dtype=torch.long, non_blocking=non_blocking),
    )


class _PreparedBatchIterator(Iterator[TrainBatch]):
    def __init__(self, source: Iterator[TrainBatch], device: torch.device) -> None:
        self.source = source
        self.device = device
        self.data_wait_seconds = 0.0

    def __iter__(self) -> _PreparedBatchIterator:
        return self

    def __next__(self) -> TrainBatch:
        started = time.perf_counter()
        batch = next(self.source)
        self.data_wait_seconds += time.perf_counter() - started
        return prepare_train_batch(batch, self.device)

    def close(self) -> None:
        return None


class _TrainBatchCollator:
    def __init__(self, cache: TokenCache) -> None:
        self.cache = cache

    def __call__(self, samples: list[Any]) -> TrainBatch:
        _pixels, label_values, index_values = zip(*samples, strict=True)
        labels = default_collate(label_values)
        indices = default_collate(index_values)
        return TrainBatch(self.cache.lookup(indices), labels, indices)


def distributed_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def local_batch_size(global_batch_size: int, world_size: int) -> int:
    if world_size < 1 or global_batch_size < 1 or global_batch_size % world_size:
        raise ValueError("train.global_batch_size must be divisible by distributed world size")
    return global_batch_size // world_size


def make_train_loader(
    dataset: torch.utils.data.Dataset[Any],
    config: ExperimentConfig,
    streams: RNGStreams,
    token_cache: TokenCache,
) -> tuple[DataLoader[Any], DistributedSampler[Any] | None]:
    rank, world_size = distributed_info()
    batch_size = local_batch_size(config.train.global_batch_size, world_size)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.train.seed,
            drop_last=True,
        )
        if world_size > 1
        else None
    )
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            generator=streams.shuffle,
            num_workers=config.data.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.data.workers > 0,
            prefetch_factor=config.data.prefetch_factor if config.data.workers > 0 else None,
            collate_fn=_TrainBatchCollator(token_cache),
            drop_last=True,
        ),
        sampler,
    )


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    last_completed_validation_epoch: int | None = None


ValidationCallback = Callable[[int, bool], dict[str, float]]


def train(
    *,
    canonical_model: nn.Module,
    execution_model: nn.Module,
    loader: DataLoader[Any],
    sampler: DistributedSampler[Any] | None,
    optimizer: CompositeOptimizer,
    ema: EMA,
    streams: RNGStreams,
    config: ExperimentConfig,
    checkpoint_manager: CheckpointManager,
    data_fingerprint: str,
    validation_plan_hash: str,
    validation_plan_path: str | Path,
    logger: RunLogger,
    token_cache: TokenCache,
    state: TrainState | None = None,
    validation_callback: ValidationCallback | None = None,
) -> TrainState:
    state = state or TrainState()
    rank, world_size = distributed_info()
    device = next(canonical_model.parameters()).device
    if loader.batch_size != local_batch_size(config.train.global_batch_size, world_size):
        raise ValueError("train loader batch size does not match the configured global batch")
    if token_cache is None:
        raise ValueError("RAE-DDT training requires a complete token cache")

    step_flops = model_flops_per_step(config)
    optimizer.zero_grad()
    for epoch_index in range(state.epoch, config.train.epochs):
        logger.event(f"epoch {epoch_index + 1}/{config.train.epochs} started")
        if sampler is not None:
            sampler.set_epoch(epoch_index)
        execution_model.train()
        epoch_started = time.perf_counter()
        iterator = _PreparedBatchIterator(iter(loader), device)
        previous_wait = iterator.data_wait_seconds
        for batch_index in range(len(loader)):
            step_started = time.perf_counter()
            clean, labels, _indices = next(iterator)
            data_wait = iterator.data_wait_seconds - previous_wait
            previous_wait = iterator.data_wait_seconds
            clean = clean.float()
            timesteps = sample_timesteps(
                clean.shape[0],
                streams.timesteps,
                device,
                loc=config.train.timestep_loc,
                scale=config.train.timestep_scale,
                shift=config.train.timestep_shift,
            )
            model_labels = classifier_free_labels(
                labels,
                config.model.num_classes,
                config.model.cfg_dropout_prob,
                streams.label_dropout,
            )
            exploration = config.exploration
            k_total = exploration.k_total
            diagnostics = None
            if k_total == 1:
                selected_noise = torch.randn(
                    clean.shape, device=device, dtype=torch.float32, generator=streams.noise
                )
                selected_noisy = noisy_interpolate(clean, selected_noise, timesteps)
                selected_indices = torch.zeros(clean.shape[0], device=device, dtype=torch.long)
            else:
                noise_candidates = torch.randn(
                    (clean.shape[0], k_total, *clean.shape[1:]),
                    device=device,
                    dtype=torch.float32,
                    generator=streams.noise,
                )
                t = timesteps.reshape(clean.shape[0], 1, 1, 1, 1)
                noisy_candidates = (1 - t) * clean[:, None] + t * noise_candidates
                diagnostics = select_candidates(
                    execution_model,
                    noisy_candidates,
                    noise_candidates,
                    clean,
                    timesteps,
                    model_labels,
                    k_early=exploration.k_early,
                    k_full=exploration.k_full,
                    early_candidate_chunk_size=exploration.early_candidate_chunk_size,
                    full_candidate_chunk_size=exploration.full_candidate_chunk_size,
                    velocity_denominator_floor=config.train.velocity_denominator_floor,
                )
                selected_noisy, selected_noise = diagnostics.noisy, diagnostics.noise
                selected_indices = diagnostics.indices

            output = execution_model(selected_noisy, timesteps, model_labels)
            losses = ddt_losses(
                clean,
                selected_noise,
                selected_noisy,
                timesteps,
                output.full_prediction,
                output.base_prediction,
                base_loss_weight=config.train.base_loss_weight,
                velocity_denominator_floor=config.train.velocity_denominator_floor,
            )
            losses.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(canonical_model.parameters(), config.train.grad_clip)
            progress = epoch_index + (batch_index + 1) / len(loader)
            lr = learning_rate_at_epoch(progress, config.train.optimizer.lr, config.train.optimizer.min_lr)
            set_learning_rate(optimizer, lr)
            optimizer.step()
            optimizer.zero_grad()
            ema.update(canonical_model)
            state.global_step += 1
            elapsed = max(time.perf_counter() - step_started, 1e-9)
            metrics = {
                "train/loss": losses.total.item(),
                "train/full_velocity_loss": losses.full_velocity.item(),
                "train/base_velocity_loss": losses.base_velocity.item(),
                "train/timestep_mean": timesteps.mean().item(),
                "train/lr": lr,
                "train/gradient_norm": float(grad_norm),
                "train/epoch": progress,
                "train/global_step": state.global_step,
                "progress/images_seen": state.global_step * config.train.global_batch_size,
                "compute/model_flops_per_step": step_flops,
                "compute/cumulative_model_flops": state.global_step * step_flops,
                "exploration/k_total": k_total,
                "exploration/k_early": exploration.k_early,
                "exploration/k_full": exploration.k_full,
                "exploration/selected_index_mean": selected_indices.float().mean().item(),
                "throughput/images_per_second": config.train.global_batch_size / elapsed,
                "throughput/tokens_per_second": config.train.global_batch_size * 64 / elapsed,
                "throughput/step_seconds": elapsed,
                "throughput/data_wait_seconds": data_wait,
                "system/peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
            }
            if diagnostics is not None:
                if diagnostics.early_candidate_zero_score is not None:
                    metrics["exploration/early/candidate_zero_score"] = diagnostics.early_candidate_zero_score.mean().item()
                    metrics["exploration/early/best_score"] = diagnostics.early_best_score.mean().item()  # type: ignore[union-attr]
                    metrics["exploration/early/score_improvement"] = (
                        diagnostics.early_candidate_zero_score
                        - diagnostics.early_best_score  # type: ignore[operator]
                    ).mean().item()
                if diagnostics.full_first_finalist_score is not None:
                    metrics["exploration/full/first_finalist_score"] = diagnostics.full_first_finalist_score.mean().item()
                    metrics["exploration/full/selected_score"] = diagnostics.full_selected_score.mean().item()  # type: ignore[union-attr]
                    metrics["exploration/full/score_improvement"] = (
                        diagnostics.full_first_finalist_score
                        - diagnostics.full_selected_score  # type: ignore[operator]
                    ).mean().item()
            logger.log_training(metrics, step=state.global_step)
            if config.train.max_steps is not None and state.global_step >= config.train.max_steps:
                break
        iterator.close()

        state.epoch = epoch_index + 1
        final_epoch = state.epoch == config.train.epochs or (
            config.train.max_steps is not None and state.global_step >= config.train.max_steps
        )
        validation_metrics: dict[str, float] = {}
        if validation_callback is not None and should_validate(
            state.epoch,
            config.train.epochs if not final_epoch else state.epoch,
            config.validation.interval,
            state.last_completed_validation_epoch,
            config.validation.validate_now,
        ):
            validation_metrics = validation_callback(state.epoch, final_epoch)
            state.last_completed_validation_epoch = state.epoch
            config.validation.validate_now = False
            logger.log(validation_metrics, step=state.global_step)

        if rank == 0 and (
            should_checkpoint(state.epoch, config.train.epochs, config.checkpoint.interval)
            or final_epoch
        ):
            checkpoint_manager.save(
                completed_epoch=state.epoch,
                global_step=state.global_step,
                model=canonical_model,
                ema=ema,
                optimizer_state=optimizer.state_dict(),
                rng_streams=streams.state_dict(),
                data_fingerprint=data_fingerprint,
                validation_plan_hash=validation_plan_hash,
                validation_plan_path=validation_plan_path,
                last_completed_validation_epoch=state.last_completed_validation_epoch,
                wandb_run_id=logger.run_id,
                resolved_config=config.as_dict(),
                token_cache_fingerprint=token_cache.fingerprint,
                fid=None,
            )
        logger.log({"train/epoch_seconds": time.perf_counter() - epoch_started}, step=state.global_step)
        if final_epoch:
            break
    return state
