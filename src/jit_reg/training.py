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
from .endpoint_cache import EndpointCache
from .flops import model_flops_per_step
from .model.tokens import classifier_free_labels
from .objective import (
    SEMANTIC_STAGE,
    diffusion_losses,
    sample_stages,
    sample_timesteps,
    waypoint_path,
)
from .optim import CompositeOptimizer, learning_rate_at_epoch, set_learning_rate
from .rng import RNGStreams
from .validation import should_validate
from .vision import DINOEndpointEncoder, vision_fingerprint
from .wandb_utils import RunLogger


class TrainBatch(NamedTuple):
    pixels: Tensor | None
    target: Tensor | None
    semantic: Tensor | None
    labels: Tensor
    indices: Tensor


def prepare_train_batch(
    batch: TrainBatch,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> TrainBatch:
    return TrainBatch(
        batch.pixels.to(device, non_blocking=non_blocking)
        if batch.pixels is not None
        else None,
        batch.target.to(device, non_blocking=non_blocking)
        if batch.target is not None
        else None,
        batch.semantic.to(device, non_blocking=non_blocking)
        if batch.semantic is not None
        else None,
        batch.labels.to(device, dtype=torch.long, non_blocking=non_blocking),
        batch.indices.to(device, dtype=torch.long, non_blocking=non_blocking),
    )


class _PreparedBatchIterator(Iterator[TrainBatch]):
    """Double-buffer the single uint8 image transfer used by online DINO extraction."""

    def __init__(self, source: Iterator[TrainBatch], device: torch.device) -> None:
        self.source = source
        self.device = device
        self.data_wait_seconds = 0.0
        self._next_batch: TrainBatch | None = None
        self._exhausted = False
        self._stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._preload()

    def __iter__(self) -> _PreparedBatchIterator:
        return self

    def __next__(self) -> TrainBatch:
        batch = self.consume()
        self.preload()
        return batch

    def consume(self) -> TrainBatch:
        if self._next_batch is None:
            raise StopIteration
        batch, self._next_batch = self._next_batch, None
        if self._stream is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_stream(self._stream)
            for tensor in batch:
                if tensor is not None:
                    tensor.record_stream(current)
        return batch

    def preload(self) -> None:
        if self._next_batch is None:
            self._preload()

    def _preload(self) -> None:
        if self._exhausted:
            return
        started = time.perf_counter()
        try:
            batch = next(self.source)
        except StopIteration:
            self._exhausted = True
            return
        self.data_wait_seconds += time.perf_counter() - started
        if self._stream is None:
            self._next_batch = prepare_train_batch(batch, self.device)
        else:
            with torch.cuda.stream(self._stream):
                self._next_batch = prepare_train_batch(
                    batch, self.device, non_blocking=True
                )

    def close(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()
        self._next_batch = None


def _collate(samples: list[Any]) -> TrainBatch:
    pixels, labels, indices = zip(*samples, strict=True)
    if any(value is None for value in pixels):
        raise ValueError("online feature training requires RGB pixels")
    return TrainBatch(
        default_collate(pixels),  # type: ignore[arg-type]
        None,
        None,
        default_collate(labels),
        default_collate(indices),
    )


class _EndpointCacheCollator:
    def __init__(self, cache: EndpointCache) -> None:
        self.cache = cache

    def __call__(self, samples: list[Any]) -> TrainBatch:
        _pixels, label_values, index_values = zip(*samples, strict=True)
        labels = default_collate(label_values)
        indices = default_collate(index_values)
        target, semantic = self.cache.lookup(indices)
        return TrainBatch(None, target, semantic, labels, indices)


def distributed_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def local_batch_size(global_batch_size: int, world_size: int) -> int:
    if world_size < 1 or global_batch_size < 1 or global_batch_size % world_size:
        raise ValueError("global batch size must be divisible by distributed world size")
    return global_batch_size // world_size


def make_train_loader(
    dataset: torch.utils.data.Dataset[Any],
    config: ExperimentConfig,
    streams: RNGStreams,
    endpoint_cache: EndpointCache | None = None,
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
            collate_fn=(
                _EndpointCacheCollator(endpoint_cache)
                if endpoint_cache is not None
                else _collate
            ),
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
    feature_encoder: DINOEndpointEncoder | None,
    loader: DataLoader[Any],
    sampler: DistributedSampler[Any] | None,
    optimizer: CompositeOptimizer,
    ema: EMA,
    streams: RNGStreams,
    config: ExperimentConfig,
    checkpoint_manager: CheckpointManager,
    data_fingerprint: str,
    decoder_sha256: str,
    latent_statistics_sha256: str,
    endpoint_cache_fingerprint: str,
    validation_plan_hash: str,
    validation_plan_path: str | Path,
    logger: RunLogger,
    state: TrainState | None = None,
    validation_callback: ValidationCallback | None = None,
) -> TrainState:
    state = state or TrainState()
    rank, world_size = distributed_info()
    device = next(canonical_model.parameters()).device
    if loader.batch_size != local_batch_size(config.train.global_batch_size, world_size):
        raise ValueError("train loader batch size does not match configured global batch")
    step_flops = model_flops_per_step(config)
    feature_id = vision_fingerprint(config.vision)
    optimizer.zero_grad()
    for epoch_index in range(state.epoch, config.train.epochs):
        logger.event(f"epoch {epoch_index + 1}/{config.train.epochs} started")
        if sampler is not None:
            sampler.set_epoch(epoch_index)
        execution_model.train()
        if feature_encoder is not None:
            feature_encoder.eval()
        epoch_started = time.perf_counter()
        iterator = _PreparedBatchIterator(iter(loader), device)
        previous_wait = 0.0
        validation_metrics: dict[str, float] = {}
        for batch_index in range(len(loader)):
            step_started = epoch_started if batch_index == 0 else time.perf_counter()
            pixels, cached_target, cached_semantic, labels, _indices = iterator.consume()
            if batch_index + 1 < len(loader):
                iterator.preload()
            feature_started = time.perf_counter()
            feature_start_event: torch.cuda.Event | None = None
            feature_end_event: torch.cuda.Event | None = None
            if device.type == "cuda":
                feature_start_event = torch.cuda.Event(enable_timing=True)
                feature_end_event = torch.cuda.Event(enable_timing=True)
                feature_start_event.record()
            if cached_target is None or cached_semantic is None:
                if pixels is None or feature_encoder is None:
                    raise RuntimeError("online endpoint extraction is unavailable")
                target_tokens, semantic_tokens = feature_encoder.encode_pair(pixels)
            else:
                target_tokens = cached_target
                semantic_tokens = cached_semantic
            feature_seconds_cpu = time.perf_counter() - feature_started
            if feature_end_event is not None:
                feature_end_event.record()
            local_timesteps = sample_timesteps(
                target_tokens.shape[0],
                streams.timesteps,
                device,
                loc=config.train.timestep_loc,
                scale=config.train.timestep_scale,
                shift=config.train.timestep_shift,
            )
            stages = sample_stages(
                target_tokens.shape[0],
                streams.stage,
                device,
                config.waypoint.semantic_stage_probability,
            )
            noise = torch.randn(
                target_tokens.shape,
                device=device,
                dtype=torch.float32,
                generator=streams.noise,
            )
            path = waypoint_path(
                target_tokens,
                semantic_tokens,
                noise,
                local_timesteps,
                stages,
                config.waypoint,
            )
            model_labels = classifier_free_labels(
                labels,
                config.model.num_classes,
                config.model.cfg_dropout_prob,
                streams.label_dropout,
            )
            output = execution_model(path.noisy, path.timestep, path.stage, model_labels)
            losses = diffusion_losses(
                path.noisy,
                path.velocity,
                path.timestep,
                path.stage,
                output.full_prediction,
                output.base_prediction,
                output.repa_projection,
                semantic_tokens,
                base_loss_weight=config.train.base_loss_weight,
                repa_coefficient=config.repa.coefficient,
                velocity_denominator_floor=config.train.velocity_denominator_floor,
            )
            losses.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                canonical_model.parameters(), config.train.grad_clip
            )
            progress = epoch_index + (batch_index + 1) / len(loader)
            lr = learning_rate_at_epoch(
                progress, config.train.optimizer.lr, config.train.optimizer.min_lr
            )
            set_learning_rate(optimizer, lr)
            optimizer.step()
            optimizer.zero_grad()
            ema.update(canonical_model)
            state.global_step += 1
            reached_limit = (
                config.train.max_steps is not None
                and state.global_step >= config.train.max_steps
            )
            data_wait = iterator.data_wait_seconds - previous_wait
            previous_wait = iterator.data_wait_seconds
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            feature_seconds = (
                feature_start_event.elapsed_time(feature_end_event) / 1000
                if feature_start_event is not None and feature_end_event is not None
                else feature_seconds_cpu
            )
            elapsed = max(time.perf_counter() - step_started, 1e-9)
            semantic_fraction = (stages == SEMANTIC_STAGE).float().mean()
            metrics = {
                "train/loss": losses.total.item(),
                "train/full_velocity_loss": losses.full_velocity.item(),
                "train/base_semantic_velocity_loss": losses.base_semantic_velocity.item(),
                "train/repa_loss": losses.repa.item(),
                "train/repa_cosine_similarity": losses.cosine_similarity.item(),
                "train/local_timestep_mean": local_timesteps.mean().item(),
                "train/global_timestep_mean": path.timestep.mean().item(),
                "train/semantic_stage_fraction": semantic_fraction.item(),
                "train/target_stage_fraction": (1 - semantic_fraction).item(),
                "train/lr": lr,
                "train/gradient_norm": float(grad_norm),
                "train/epoch": progress,
                "train/global_step": state.global_step,
                "progress/images_seen": state.global_step * config.train.global_batch_size,
                "compute/model_flops_per_step": step_flops,
                "compute/cumulative_model_flops": state.global_step * step_flops,
                "throughput/images_per_second": config.train.global_batch_size / elapsed,
                "throughput/tokens_per_second": (
                    config.train.global_batch_size * config.vision.grid_size**2 / elapsed
                ),
                "throughput/step_seconds": elapsed,
                "throughput/feature_encoder_seconds": feature_seconds,
                "throughput/data_wait_seconds": data_wait,
                "system/peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
            }
            logger.log_training(metrics, step=state.global_step)
            if reached_limit:
                break
        iterator.close()
        state.epoch = epoch_index + 1
        final_epoch = state.epoch == config.train.epochs or (
            config.train.max_steps is not None and state.global_step >= config.train.max_steps
        )
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
            fid_values = [
                value for key, value in validation_metrics.items() if key.startswith("validation/FID-")
            ]
            path = checkpoint_manager.save(
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
                feature_fingerprint=feature_id,
                decoder_sha256=decoder_sha256,
                latent_statistics_sha256=latent_statistics_sha256,
                endpoint_cache_fingerprint=endpoint_cache_fingerprint,
                fid=fid_values[0] if fid_values and config.checkpoint.save_best_fid else None,
            )
            if config.wandb.checkpoint_artifacts:
                logger.log_checkpoint(str(path), f"diffusion-epoch-{state.epoch:04d}")
        logger.log(
            {"train/epoch_seconds": time.perf_counter() - epoch_started},
            step=state.global_step,
        )
        if final_epoch:
            break
    return state
