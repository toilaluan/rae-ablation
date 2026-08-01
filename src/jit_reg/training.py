from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler

from .checkpoint import EMA, CheckpointManager, should_checkpoint
from .config import ExperimentConfig
from .data import prepare_pixels
from .exploration import select_candidates
from .feature_cache import FeatureCache
from .flops import model_flops_per_step
from .latent_cache import LatentCache
from .model.tokens import classifier_free_labels
from .objective import noisy_interpolate, sample_timesteps, sit_losses
from .optim import CompositeOptimizer, learning_rate_at_epoch, set_learning_rate
from .repa import DINOv3FeatureExtractor
from .rng import RNGStreams
from .vae import Flux2Codec
from .validation import should_validate
from .wandb_utils import RunLogger


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    last_completed_validation_epoch: int | None = None


ValidationCallback = Callable[[int, bool], dict[str, float]]


class TrainBatch(NamedTuple):
    pixels: Tensor | None
    labels: Tensor
    indices: Tensor
    cached_latents: Tensor | None
    cached_repa: Tensor | None


def prepare_train_batch(
    batch: TrainBatch,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> TrainBatch:
    device = torch.device(device)
    return TrainBatch(
        (
            prepare_pixels(batch.pixels, device, non_blocking=non_blocking)
            if batch.pixels is not None
            else None
        ),
        batch.labels.to(device=device, non_blocking=non_blocking),
        batch.indices,
        (
            batch.cached_latents.to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            )
            if batch.cached_latents is not None
            else None
        ),
        (
            batch.cached_repa.to(
                device=device,
                dtype=torch.float32,
                non_blocking=non_blocking,
            )
            if batch.cached_repa is not None
            else None
        ),
    )


def _record_destination_stream(batch: TrainBatch, stream: torch.cuda.Stream) -> None:
    for value in (batch.pixels, batch.labels, batch.cached_latents, batch.cached_repa):
        if value is not None:
            value.record_stream(stream)


class _PreparedBatchIterator(Iterator[TrainBatch]):
    def __init__(self, iterator: Iterator[TrainBatch], device: torch.device) -> None:
        self.iterator = iterator
        self.device = device
        self.data_wait_seconds = 0.0

    def __iter__(self) -> _PreparedBatchIterator:
        return self

    def __next__(self) -> TrainBatch:
        started = time.perf_counter()
        batch = next(self.iterator)
        self.data_wait_seconds += time.perf_counter() - started
        return prepare_train_batch(batch, self.device)

    def close(self) -> None:
        return


class _CUDABatchPrefetcher(_PreparedBatchIterator):
    def __init__(self, iterator: Iterator[TrainBatch], device: torch.device) -> None:
        super().__init__(iterator, device)
        self.stream = torch.cuda.Stream(device=device)
        self._next_batch: TrainBatch | None = None
        self._sources: list[tuple[TrainBatch, torch.cuda.Event]] = []
        self._preload()

    def _release_completed_sources(self) -> None:
        self._sources = [(source, event) for source, event in self._sources if not event.query()]

    def _preload(self) -> None:
        self._release_completed_sources()
        started = time.perf_counter()
        try:
            source = next(self.iterator)
        except StopIteration:
            self.data_wait_seconds += time.perf_counter() - started
            self._next_batch = None
            return
        self.data_wait_seconds += time.perf_counter() - started
        finished = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            self._next_batch = prepare_train_batch(source, self.device, non_blocking=True)
            finished.record()
        self._sources.append((source, finished))

    def __next__(self) -> TrainBatch:
        if self._next_batch is None:
            raise StopIteration
        stream = torch.cuda.current_stream(self.device)
        stream.wait_stream(self.stream)
        batch = self._next_batch
        _record_destination_stream(batch, stream)
        self._next_batch = None
        self._preload()
        return batch

    def close(self) -> None:
        self.stream.synchronize()
        self._sources.clear()


class _TrainBatchCollator:
    def __init__(self, config: ExperimentConfig, dataset_size: int) -> None:
        self.config = config
        self.dataset_size = dataset_size
        self._feature_cache: FeatureCache | None = None
        self._latent_cache: LatentCache | None = None

    def __call__(self, samples: list[Any]) -> TrainBatch:
        pixel_values, label_values, index_values = zip(*samples, strict=True)
        if all(value is None for value in pixel_values):
            pixels = None
        elif any(value is None for value in pixel_values):
            raise ValueError("loader batch mixes examples with and without pixels")
        else:
            pixels = default_collate(pixel_values)
        labels = default_collate(label_values)
        indices = default_collate(index_values)
        cached_repa = None
        if self.config.repa.enabled and self.config.repa.cache.enabled:
            if self._feature_cache is None:
                self._feature_cache = FeatureCache(self.config, self.dataset_size)
            cached_repa = self._feature_cache.lookup(indices)
        cached_latents = None
        if self.config.vae.cache.enabled:
            if self._latent_cache is None:
                self._latent_cache = LatentCache(self.config, self.dataset_size)
            cached_latents = self._latent_cache.lookup(indices)
        return TrainBatch(pixels, labels, indices, cached_latents, cached_repa)


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
            collate_fn=_TrainBatchCollator(config, len(dataset)),
            drop_last=True,
        ),
        sampler,
    )


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
    feature_encoder: DINOv3FeatureExtractor | None = None,
    feature_cache: FeatureCache | None = None,
    vae_codec: Flux2Codec | None = None,
    latent_cache: LatentCache | None = None,
    state: TrainState | None = None,
    validation_callback: ValidationCallback | None = None,
) -> TrainState:
    state = state or TrainState()
    rank, world_size = distributed_info()
    device = next(canonical_model.parameters()).device
    if loader.batch_size != local_batch_size(config.train.global_batch_size, world_size):
        raise ValueError("train loader batch size does not match the configured global batch")
    target_sources = int(feature_encoder is not None) + int(feature_cache is not None)
    if config.repa.enabled != (target_sources == 1):
        raise ValueError("active REPA requires exactly one online or cached target source")
    input_sources = int(vae_codec is not None) + int(latent_cache is not None)
    if config.model.input_space == "flux2_latents" and input_sources != 1:
        raise ValueError("latent training requires exactly one online VAE or latent cache")
    if config.model.input_space == "pixels" and input_sources:
        raise ValueError("pixel training cannot use a VAE or latent cache")

    step_flops = model_flops_per_step(config)
    optimizer.zero_grad()
    for epoch_index in range(state.epoch, config.train.epochs):
        logger.event(f"epoch {epoch_index + 1}/{config.train.epochs} started")
        if sampler is not None:
            sampler.set_epoch(epoch_index)
        execution_model.train()
        epoch_started = time.perf_counter()
        iterator: _PreparedBatchIterator
        raw_iterator = iter(loader)
        iterator = (
            _CUDABatchPrefetcher(raw_iterator, device)
            if device.type == "cuda"
            else _PreparedBatchIterator(raw_iterator, device)
        )
        previous_wait = iterator.data_wait_seconds
        for batch_index in range(len(loader)):
            step_started = time.perf_counter()
            batch = next(iterator)
            data_wait = iterator.data_wait_seconds - previous_wait
            previous_wait = iterator.data_wait_seconds
            pixels, labels, _indices, cached_latents, cached_repa = batch
            if config.model.input_space == "pixels":
                if pixels is None:
                    raise RuntimeError("pixel training requires RGB input")
                clean = pixels.mul(2).sub(1)
            elif latent_cache is not None:
                if cached_latents is None:
                    raise RuntimeError("cached latent training requires cached latents")
                clean = cached_latents
            else:
                if vae_codec is None or pixels is None:
                    raise RuntimeError("online latent training requires RGB input and VAE")
                clean = vae_codec.encode(pixels)

            repa_target = cached_repa
            if feature_encoder is not None:
                if pixels is None:
                    raise RuntimeError("online REPA requires RGB pixels")
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    repa_target = feature_encoder.targets(pixels, config.model.grid_size)

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
                config.model.class_dropout_prob,
                streams.label_dropout,
            )
            k = config.exploration.k
            if k == 1:
                selected_noise = torch.randn(
                    clean.shape,
                    device=device,
                    dtype=torch.float32,
                    generator=streams.noise,
                )
                selected_noisy = noisy_interpolate(clean, selected_noise, timesteps)
                selected_score = clean.new_zeros(clean.shape[0])
                candidate_zero_score = selected_score
                selected_indices = torch.zeros(
                    clean.shape[0], device=device, dtype=torch.long
                )
            else:
                noise_candidates = torch.randn(
                    (clean.shape[0], k, *clean.shape[1:]),
                    device=device,
                    dtype=torch.float32,
                    generator=streams.noise,
                )
                t = timesteps.reshape(clean.shape[0], 1, 1, 1, 1)
                noisy_candidates = (
                    t * clean[:, None].float() + (1 - t) * noise_candidates
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    selected = select_candidates(
                        execution_model,
                        noisy_candidates,
                        noise_candidates,
                        clean,
                        timesteps,
                        model_labels,
                        selector=config.exploration.selector,
                        repa_target=repa_target,
                        candidate_chunk_size=config.exploration.candidate_chunk_size,
                        velocity_denominator_floor=config.train.velocity_denominator_floor,
                    )
                selected_noisy, selected_noise = selected.noisy, selected.noise
                selected_indices = selected.indices
                selected_score = selected.selected_score
                candidate_zero_score = selected.candidate_zero_score

            gpu_started = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            gpu_finished = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if gpu_started is not None:
                gpu_started.record()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = execution_model(
                    selected_noisy,
                    timesteps,
                    model_labels,
                    return_repa=config.repa.enabled,
                )
            losses = sit_losses(
                clean,
                selected_noise,
                selected_noisy,
                timesteps,
                output.x_prediction,
                repa_prediction=output.repa_prediction,
                repa_target=repa_target,
                repa_weight=config.repa.loss_weight,
                velocity_denominator_floor=config.train.velocity_denominator_floor,
            )
            losses.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                canonical_model.parameters(), config.train.grad_clip
            )
            progress = epoch_index + (batch_index + 1) / len(loader)
            lr = learning_rate_at_epoch(
                progress,
                config.train.optimizer.lr,
                config.train.optimizer.min_lr,
            )
            set_learning_rate(optimizer, lr)
            optimizer.step()
            optimizer.zero_grad()
            ema.update(canonical_model)
            if gpu_finished is not None:
                gpu_finished.record()
                gpu_finished.synchronize()
            state.global_step += 1
            elapsed = max(time.perf_counter() - step_started, 1e-9)
            gpu_seconds = (
                gpu_started.elapsed_time(gpu_finished) / 1_000
                if gpu_started is not None and gpu_finished is not None
                else elapsed
            )
            images_seen = state.global_step * config.train.global_batch_size
            cumulative_flops = state.global_step * step_flops
            logger.log_training(
                {
                    "train/loss": losses.total.item(),
                    "train/image_velocity_loss": losses.velocity.item(),
                    "train/repa_loss": losses.repa.item(),
                    "train/timestep_mean": timesteps.mean().item(),
                    "train/lr": lr,
                    "train/gradient_norm": float(grad_norm),
                    "train/ema_decay": ema.decay,
                    "train/epoch": progress,
                    "train/global_step": state.global_step,
                    "progress/images_seen": images_seen,
                    "compute/model_flops_per_step": step_flops,
                    "compute/cumulative_model_flops": cumulative_flops,
                    "exploration/k": k,
                    "exploration/selected_score": selected_score.mean().item(),
                    "exploration/candidate_zero_score": candidate_zero_score.mean().item(),
                    "exploration/score_improvement": (
                        candidate_zero_score - selected_score
                    ).mean().item(),
                    "exploration/selected_index_mean": selected_indices.float().mean().item(),
                    "throughput/images_per_second": config.train.global_batch_size / elapsed,
                    "throughput/selector_candidates_per_second": (
                        config.train.global_batch_size * (k if k > 1 else 0) / elapsed
                    ),
                    "throughput/jit_tokens_per_second": (
                        config.train.global_batch_size * config.model.grid_size**2 / elapsed
                    ),
                    "throughput/step_seconds": elapsed,
                    "throughput/data_wait_seconds": data_wait,
                    "throughput/host_batch_seconds": elapsed,
                    "throughput/optimizer_host_seconds": 0.0,
                    "throughput/logging_seconds": 0.0,
                    "throughput/gpu_input_seconds": 0.0,
                    "throughput/gpu_forward_backward_seconds": gpu_seconds,
                    "throughput/gpu_optimizer_seconds": 0.0,
                    "system/peak_memory_gib": (
                        torch.cuda.max_memory_allocated(device) / 2**30
                        if device.type == "cuda"
                        else 0.0
                    ),
                },
                step=state.global_step,
            )
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
            validation_metrics.update(
                {
                    "progress/epoch": float(state.epoch),
                    "progress/images_seen": float(
                        state.global_step * config.train.global_batch_size
                    ),
                    "compute/cumulative_model_flops": float(state.global_step * step_flops),
                }
            )
            state.last_completed_validation_epoch = state.epoch
            config.validation.validate_now = False
            logger.log(validation_metrics, step=state.global_step)

        if rank == 0 and (
            should_checkpoint(state.epoch, config.train.epochs, config.checkpoint.interval)
            or final_epoch
        ):
            count = (
                config.validation.final_samples
                if final_epoch
                else config.validation.periodic_samples
            )
            suffix = f"{count // 1000}K" if count % 1000 == 0 else str(count)
            fid = validation_metrics.get(f"validation/FID-{suffix}")
            if not config.checkpoint.save_best_fid:
                fid = None
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
                feature_cache_fingerprints=(
                    feature_cache.fingerprints if feature_cache is not None else None
                ),
                latent_cache_fingerprint=(
                    latent_cache.fingerprint if latent_cache is not None else None
                ),
                fid=fid,
            )
            if config.wandb.checkpoint_artifacts:
                logger.log_checkpoint(str(path), f"checkpoint-epoch-{state.epoch:04d}")
            logger.event(f"checkpoint saved: {path}")
        logger.log(
            {"train/epoch_seconds": time.perf_counter() - epoch_started},
            step=state.global_step,
        )
        if final_epoch:
            break
    return state
