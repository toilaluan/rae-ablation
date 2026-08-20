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
from .data import prepare_pixels
from .flops import model_flops_per_step, pixel_encoder_flops_per_step
from .model.tokens import classifier_free_labels
from .objective import (
    ddt_losses,
    mix_endpoints,
    noisy_interpolate,
    sample_mix_ratios,
    sample_timesteps,
)
from .optim import CompositeOptimizer, learning_rate_at_epoch, set_learning_rate
from .pixel_codec import PixelCodecArtifact
from .rng import RNGStreams
from .token_cache import TokenCache, normalize_tokens
from .validation import should_validate
from .wandb_utils import RunLogger


class TrainBatch(NamedTuple):
    pixels: Tensor
    rae_tokens: Tensor
    labels: Tensor
    indices: Tensor


def prepare_train_batch(
    batch: TrainBatch,
    device: torch.device | str,
    *,
    mean: Tensor | None = None,
    std: Tensor | None = None,
    non_blocking: bool = False,
) -> TrainBatch:
    if (mean is None) != (std is None):
        raise ValueError("mean and std must be supplied together")
    pixels = prepare_pixels(batch.pixels, device, non_blocking=non_blocking)
    tokens = batch.rae_tokens.to(device, non_blocking=non_blocking)
    if mean is not None and std is not None:
        tokens = normalize_tokens(tokens, mean, std)
    return TrainBatch(
        pixels,
        tokens,
        batch.labels.to(device, dtype=torch.long, non_blocking=non_blocking),
        batch.indices.to(device, dtype=torch.long, non_blocking=non_blocking),
    )


class _PreparedBatchIterator(Iterator[TrainBatch]):
    """Double-buffer batches and normalize raw latents on the target device."""

    def __init__(
        self,
        source: Iterator[TrainBatch],
        device: torch.device,
        mean: Tensor,
        std: Tensor,
    ) -> None:
        self.source = source
        self.device = device
        self.mean = mean.to(device)
        self.std = std.to(device)
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

        batch = self._next_batch
        self._next_batch = None

        if self._stream is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_stream(self._stream)
            for tensor in batch:
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
            self._next_batch = prepare_train_batch(
                batch,
                self.device,
                mean=self.mean,
                std=self.std,
            )
            return

        with torch.cuda.stream(self._stream):
            self._next_batch = prepare_train_batch(
                batch,
                self.device,
                mean=self.mean,
                std=self.std,
                non_blocking=True,
            )

    def close(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()
        self._next_batch = None


class _TrainBatchCollator:
    def __init__(self, cache: TokenCache) -> None:
        self.cache = cache

    def __call__(self, samples: list[Any]) -> TrainBatch:
        pixel_values, label_values, index_values = zip(*samples, strict=True)
        if any(value is None for value in pixel_values):
            raise ValueError("mixed-token training requires RGB pixels")
        pixels = default_collate(pixel_values)
        labels = default_collate(label_values)
        indices = default_collate(index_values)
        return TrainBatch(pixels, self.cache.raw_lookup(indices), labels, indices)


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
    pixel_codec: PixelCodecArtifact,
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

    base_step_flops = model_flops_per_step(config)
    pixel_encoder_step_flops = pixel_encoder_flops_per_step(config)
    step_flops = base_step_flops + pixel_encoder_step_flops
    token_mean = token_cache.mean.to(device)
    token_std = token_cache.std.to(device)
    pixel_mean = pixel_codec.statistics.mean.to(device)
    pixel_std = pixel_codec.statistics.std.to(device)
    codec_model = pixel_codec.model
    codec_model.eval()
    optimizer.zero_grad()
    for epoch_index in range(state.epoch, config.train.epochs):
        logger.event(f"epoch {epoch_index + 1}/{config.train.epochs} started")
        if sampler is not None:
            sampler.set_epoch(epoch_index)
        execution_model.train()
        epoch_started = time.perf_counter()
        iterator = _PreparedBatchIterator(iter(loader), device, token_mean, token_std)
        previous_wait = 0.0
        for batch_index in range(len(loader)):
            # Include first-batch pipeline warmup in the first step. Later steps
            # start after the preceding iteration has finished logging.
            step_started = epoch_started if batch_index == 0 else time.perf_counter()
            pixels, rae_tokens, labels, _indices = iterator.consume()
            rae_tokens = rae_tokens.float()
            with torch.inference_mode(), torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                raw_pixel_tokens = codec_model.encode(pixels)
            pixel_tokens = normalize_tokens(raw_pixel_tokens, pixel_mean, pixel_std)
            ratios = sample_mix_ratios(
                rae_tokens.shape[0],
                streams.mix_ratio,
                device,
                loc=config.mixing.loc,
                scale=config.mixing.scale,
                rae_endpoint_probability=config.mixing.rae_endpoint_probability,
                pixel_endpoint_probability=config.mixing.pixel_endpoint_probability,
                fixed_ratio=config.mixing.fixed_ratio,
            )
            clean = mix_endpoints(pixel_tokens, rae_tokens, ratios)
            timesteps = sample_timesteps(
                clean.shape[0],
                streams.timesteps,
                device,
                loc=config.train.timestep_loc,
                scale=config.train.timestep_scale,
            )
            model_labels = classifier_free_labels(
                labels,
                config.model.num_classes,
                config.model.cfg_dropout_prob,
                streams.label_dropout,
            )
            selected_noise = torch.randn(
                clean.shape, device=device, dtype=torch.float32, generator=streams.noise
            )
            selected_noisy = noisy_interpolate(clean, selected_noise, timesteps)
            output = execution_model(selected_noisy, timesteps, ratios, model_labels)
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
            reached_step_limit = (
                config.train.max_steps is not None
                and state.global_step >= config.train.max_steps
            )
            if batch_index + 1 < len(loader) and not reached_step_limit:
                # DataLoader dequeue, H2D, and GPU normalization now run while
                # the current step's queued CUDA work is still executing.
                iterator.preload()
            data_wait = iterator.data_wait_seconds - previous_wait
            previous_wait = iterator.data_wait_seconds
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
                "compute/model_flops_per_step": base_step_flops,
                "compute/pixel_encoder_flops_per_step": pixel_encoder_step_flops,
                "compute/total_flops_per_step": step_flops,
                "compute/cumulative_model_flops": state.global_step * base_step_flops,
                "compute/cumulative_total_flops": state.global_step * step_flops,
                "mixing/ratio_mean": ratios.mean().item(),
                "mixing/ratio_std": ratios.std(correction=0).item(),
                "mixing/rae_endpoint_fraction": (ratios == 0).float().mean().item(),
                "mixing/pixel_endpoint_fraction": (ratios == 1).float().mean().item(),
                "mixing/pixel_mean": pixel_tokens.mean().item(),
                "mixing/pixel_std": pixel_tokens.std(correction=0).item(),
                "mixing/rae_mean": rae_tokens.mean().item(),
                "mixing/rae_std": rae_tokens.std(correction=0).item(),
                "mixing/x0_mean": clean.mean().item(),
                "mixing/x0_std": clean.std(correction=0).item(),
                "mixing/expected_x0_variance": pixel_codec.statistics.expected_mixture_variance(
                    ratios
                ).mean().item(),
                "mixing/endpoint_cosine": torch.nn.functional.cosine_similarity(
                    pixel_tokens.flatten(2).transpose(1, 2),
                    rae_tokens.flatten(2).transpose(1, 2),
                    dim=-1,
                ).mean().item(),
                "throughput/images_per_second": config.train.global_batch_size / elapsed,
                "throughput/tokens_per_second": config.train.global_batch_size * 64 / elapsed,
                "throughput/step_seconds": elapsed,
                "throughput/data_wait_seconds": data_wait,
                "system/peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
            }
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
            fid = next(
                (
                    value
                    for name, value in validation_metrics.items()
                    if name.startswith("validation/FID-")
                ),
                None,
            )
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
                pixel_codec=pixel_codec,
                fid=fid if config.checkpoint.save_best_fid else None,
            )
        logger.log({"train/epoch_seconds": time.perf_counter() - epoch_started}, step=state.global_step)
        if final_epoch:
            break
    return state
