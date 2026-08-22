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
from .data import patchify_rgb, prepare_pixels
from .endpoint_cache import EndpointCache
from .latent_statistics import EndpointLatentNormalizer
from .objective import flow_losses, flow_path, sample_timesteps
from .optim import CompositeOptimizer, learning_rate_at_epoch, set_learning_rate
from .rng import RNGStreams
from .validation import should_validate
from .vision import DINOEndpointEncoder, vision_fingerprint
from .wandb_utils import RunLogger


class TrainBatch(NamedTuple):
    pixels: Tensor | None
    target: Tensor | None
    semantic: Tensor | None
    indices: Tensor


def prepare_train_batch(
    batch: TrainBatch,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> TrainBatch:
    return TrainBatch(
        *(
            value.to(device, non_blocking=non_blocking) if value is not None else None
            for value in batch[:-1]
        ),
        batch.indices.to(device, dtype=torch.long, non_blocking=non_blocking),
    )


class _PreparedBatchIterator(Iterator[TrainBatch]):
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
    pixels, _labels, indices = zip(*samples, strict=True)
    if any(value is None for value in pixels):
        raise ValueError("online endpoint training requires RGB pixels")
    return TrainBatch(
        default_collate(pixels),  # type: ignore[arg-type]
        None,
        None,
        default_collate(indices),
    )


class _EndpointCacheCollator:
    def __init__(self, cache: EndpointCache) -> None:
        self.cache = cache

    def __call__(self, samples: list[Any]) -> TrainBatch:
        pixel_values, _labels, index_values = zip(*samples, strict=True)
        indices = default_collate(index_values)
        target, semantic = self.cache.lookup(indices)
        pixels = (
            None
            if all(value is None for value in pixel_values)
            else default_collate(pixel_values)
        )
        return TrainBatch(pixels, target, semantic, indices)


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
    train = config.flow.train
    batch_size = local_batch_size(train.global_batch_size, world_size)
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=train.seed,
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
            prefetch_factor=(
                config.data.prefetch_factor if config.data.workers > 0 else None
            ),
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
    normalizer: EndpointLatentNormalizer,
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
    flow = config.flow
    train_config = flow.train
    if loader.batch_size != local_batch_size(train_config.global_batch_size, world_size):
        raise ValueError("train loader batch size does not match configured global batch")
    optimizer.zero_grad()
    for epoch_index in range(state.epoch, train_config.epochs):
        logger.event(
            f"{flow.endpoint} flow epoch {epoch_index + 1}/{train_config.epochs} started"
        )
        if sampler is not None:
            sampler.set_epoch(epoch_index)
        execution_model.train()
        if feature_encoder is not None:
            feature_encoder.eval()
        epoch_started = time.perf_counter()
        iterator = _PreparedBatchIterator(iter(loader), device)
        validation_metrics: dict[str, float] = {}
        for batch_index in range(len(loader)):
            pixels, cached_target, cached_semantic, _indices = iterator.consume()
            if batch_index + 1 < len(loader):
                iterator.preload()
            if cached_target is None or cached_semantic is None:
                if pixels is None or feature_encoder is None:
                    raise RuntimeError("online endpoint extraction is unavailable")
                endpoints = feature_encoder.encode_pair(pixels)
                target_tokens = endpoints.target
                semantic_tokens = endpoints.semantic
            else:
                target_tokens = cached_target
                semantic_tokens = cached_semantic
            if flow.endpoint == "target":
                endpoint = target_tokens
            else:
                if pixels is None:
                    raise RuntimeError("pixel flow requires RGB pixels")
                endpoint = normalizer.normalize_pixel(
                    patchify_rgb(prepare_pixels(pixels, device))
                )
            timesteps = sample_timesteps(
                endpoint.shape[0],
                streams.timesteps,
                device,
                loc=train_config.timestep_loc,
                scale=train_config.timestep_scale,
                shift=train_config.timestep_shift,
            )
            noise = torch.randn(
                endpoint.shape,
                generator=streams.noise,
                device=device,
                dtype=torch.float32,
            )
            path = flow_path(endpoint, noise, timesteps)
            prediction = execution_model(path.noisy, path.timestep, semantic_tokens)
            losses = flow_losses(
                prediction,
                path.clean,
            )
            losses.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                canonical_model.parameters(), train_config.grad_clip
            )
            progress = epoch_index + (batch_index + 1) / len(loader)
            lr = learning_rate_at_epoch(
                progress,
                train_config.optimizer.lr,
                train_config.optimizer.min_lr,
            )
            set_learning_rate(optimizer, lr)
            optimizer.step()
            optimizer.zero_grad()
            ema.update(canonical_model)
            state.global_step += 1
            reached_limit = (
                train_config.max_steps is not None
                and state.global_step >= train_config.max_steps
            )
            final_step = (
                epoch_index + 1 == train_config.epochs
                and batch_index + 1 == len(loader)
            )
            if logger.should_log_training(state.global_step, force=reached_limit or final_step):
                logger.log_training(
                    {
                        "train/loss": losses.total.item(),
                        "train/endpoint_loss": losses.endpoint.item(),
                        "train/timestep_mean": path.timestep.mean().item(),
                        "train/lr": lr,
                        "train/gradient_norm": float(grad_norm),
                        "train/epoch": progress,
                        "train/global_step": state.global_step,
                        "progress/images_seen": state.global_step
                        * train_config.global_batch_size,
                    },
                    step=state.global_step,
                    force=reached_limit or final_step,
                )
            if reached_limit:
                break
        iterator.close()
        state.epoch = epoch_index + 1
        final_epoch = state.epoch == train_config.epochs or (
            train_config.max_steps is not None
            and state.global_step >= train_config.max_steps
        )
        if validation_callback is not None and should_validate(
            state.epoch,
            train_config.epochs if not final_epoch else state.epoch,
            config.validation.interval,
            state.last_completed_validation_epoch,
            config.validation.validate_now,
        ):
            validation_metrics = validation_callback(state.epoch, final_epoch)
            state.last_completed_validation_epoch = state.epoch
            config.validation.validate_now = False
            logger.log(validation_metrics, step=state.global_step)
        if rank == 0 and (
            should_checkpoint(
                state.epoch, train_config.epochs, config.checkpoint.interval
            )
            or final_epoch
        ):
            lpips = validation_metrics.get(
                f"validation/lpips_{config.sampling.steps}step"
            )
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
                feature_fingerprint=vision_fingerprint(config.vision),
                decoder_sha256=decoder_sha256,
                latent_statistics_sha256=latent_statistics_sha256,
                endpoint_cache_fingerprint=endpoint_cache_fingerprint,
                endpoint_kind=flow.endpoint,
                lpips=(
                    lpips if lpips is not None and config.checkpoint.save_best_lpips else None
                ),
            )
            if config.wandb.checkpoint_artifacts:
                logger.log_checkpoint(str(path), f"{flow.endpoint}-epoch-{state.epoch:04d}")
        logger.log(
            {"train/epoch_seconds": time.perf_counter() - epoch_started},
            step=state.global_step,
        )
        if final_epoch:
            break
    return state
