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
from .feature_cache import FeatureCache
from .latent_cache import LatentCache
from .model.ddt import classifier_free_labels
from .objective import (
    jit_losses,
    noisy_interpolate,
    sample_timesteps,
)
from .optim import CompositeOptimizer, learning_rate_at_epoch, set_learning_rate
from .reg import (
    DINOv3FeatureExtractor,
    FeatureTargets,
    repa_target_for_scope,
    scale_reg_targets,
)
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
    cached_targets: FeatureTargets | None


def _prepare_pixels(
    pixels: Tensor,
    device: torch.device,
    *,
    non_blocking: bool,
) -> Tensor:
    source_is_uint8 = pixels.dtype == torch.uint8
    pixels = pixels.to(
        device=device,
        dtype=torch.float32,
        non_blocking=non_blocking,
    )
    if source_is_uint8:
        pixels.div_(255)
    return pixels


def prepare_train_batch(
    batch: TrainBatch,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> TrainBatch:
    """Move and convert a loader batch without changing its pixel values."""
    device = torch.device(device)
    cached_targets = batch.cached_targets
    cached_latents = (
        batch.cached_latents.to(
            device=device,
            dtype=torch.float32,
            non_blocking=non_blocking,
        )
        if batch.cached_latents is not None
        else None
    )
    prepared_targets = (
        FeatureTargets(
            (
                cached_targets.reg.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=non_blocking,
                )
                if cached_targets.reg is not None
                else None
            ),
            (
                cached_targets.repa.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=non_blocking,
                )
                if cached_targets.repa is not None
                else None
            ),
        )
        if cached_targets is not None
        else None
    )
    return TrainBatch(
        (
            _prepare_pixels(batch.pixels, device, non_blocking=non_blocking)
            if batch.pixels is not None
            else None
        ),
        batch.labels.to(device=device, non_blocking=non_blocking),
        batch.indices,
        cached_latents,
        prepared_targets,
    )


def _record_destination_stream(
    batch: TrainBatch,
    stream: torch.cuda.Stream,
) -> None:
    if batch.pixels is not None:
        batch.pixels.record_stream(stream)
    batch.labels.record_stream(stream)
    if batch.cached_latents is not None:
        batch.cached_latents.record_stream(stream)
    if batch.cached_targets is not None:
        if batch.cached_targets.reg is not None:
            batch.cached_targets.reg.record_stream(stream)
        if batch.cached_targets.repa is not None:
            batch.cached_targets.repa.record_stream(stream)


class _PreparedBatchIterator(Iterator[TrainBatch]):
    """Synchronously prepare CPU batches while tracking loader wait time."""

    def __init__(
        self,
        iterator: Iterator[TrainBatch],
        device: torch.device,
    ) -> None:
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

    def take_input_timing(
        self,
    ) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
        return None

    def close(self) -> None:
        return


class _CUDABatchPrefetcher(_PreparedBatchIterator):
    """Overlap pinned-memory batch staging with model work on a CUDA stream."""

    def __init__(
        self,
        iterator: Iterator[TrainBatch],
        device: torch.device,
    ) -> None:
        super().__init__(iterator, device)
        self.stream = torch.cuda.Stream(device=device)
        self._next_batch: TrainBatch | None = None
        self._next_timing: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
        self._current_timing: (
            tuple[torch.cuda.Event, torch.cuda.Event] | None
        ) = None
        self._sources: list[tuple[TrainBatch, torch.cuda.Event]] = []
        self._preload()

    def _release_completed_sources(self) -> None:
        self._sources = [
            (source, finished)
            for source, finished in self._sources
            if not finished.query()
        ]

    def _preload(self) -> None:
        self._release_completed_sources()
        started_waiting = time.perf_counter()
        try:
            source = next(self.iterator)
        except StopIteration:
            self.data_wait_seconds += time.perf_counter() - started_waiting
            self._next_batch = None
            self._next_timing = None
            return
        self.data_wait_seconds += time.perf_counter() - started_waiting
        input_started = torch.cuda.Event(enable_timing=True)
        input_finished = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            input_started.record()
            self._next_batch = prepare_train_batch(
                source,
                self.device,
                non_blocking=True,
            )
            input_finished.record()
        self._sources.append((source, input_finished))
        self._next_timing = (input_started, input_finished)

    def __next__(self) -> TrainBatch:
        if self._next_batch is None:
            raise StopIteration
        consuming_stream = torch.cuda.current_stream(self.device)
        consuming_stream.wait_stream(self.stream)
        batch = self._next_batch
        self._current_timing = self._next_timing
        _record_destination_stream(batch, consuming_stream)
        self._next_batch = None
        self._next_timing = None
        self._preload()
        return batch

    def take_input_timing(
        self,
    ) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
        timing = self._current_timing
        self._current_timing = None
        return timing

    def close(self) -> None:
        self.stream.synchronize()
        self._sources.clear()
        self._next_batch = None
        self._next_timing = None


class _TrainBatchCollator:
    """Collate images and lazily gather cached targets inside each loader worker."""

    def __init__(
        self,
        config: ExperimentConfig,
        dataset_size: int,
    ) -> None:
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
        cached_targets = None
        cache_active = (
            self.config.feature_cache.enabled
            and self.config.model.features.enabled
        )
        if cache_active:
            if self._feature_cache is None:
                self._feature_cache = FeatureCache(
                    self.config,
                    self.dataset_size,
                )
            cached_targets = self._feature_cache.lookup(indices)
        cached_latents = None
        if self.config.vae.cache.enabled:
            if self._latent_cache is None:
                self._latent_cache = LatentCache(
                    self.config,
                    self.dataset_size,
                )
            cached_latents = self._latent_cache.lookup(indices)
        return TrainBatch(
            pixels,
            labels,
            indices,
            cached_latents,
            cached_targets,
        )


def distributed_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def local_batch_size(global_batch_size: int, world_size: int) -> int:
    if world_size < 1:
        raise ValueError("distributed world size must be positive")
    if global_batch_size < 1 or global_batch_size % world_size:
        raise ValueError(
            "train.global_batch_size must be positive and exactly divisible "
            "by the distributed world size"
        )
    return global_batch_size // world_size


def make_train_loader(
    dataset: torch.utils.data.Dataset[Any],
    config: ExperimentConfig,
    streams: RNGStreams,
) -> tuple[DataLoader[Any], DistributedSampler[Any] | None]:
    rank, world_size = distributed_info()
    rank_batch_size = local_batch_size(
        config.train.global_batch_size, world_size
    )
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
            batch_size=rank_batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            generator=streams.shuffle,
            num_workers=config.data.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.data.workers > 0,
            prefetch_factor=config.data.prefetch_factor if config.data.workers > 0 else None,
            collate_fn=_TrainBatchCollator(
                config,
                len(dataset),
            ),
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
    expected_batch_size = local_batch_size(
        config.train.global_batch_size, world_size
    )
    if loader.batch_size != expected_batch_size:
        raise ValueError(
            "train loader batch size does not match global_batch_size / world_size"
        )
    target_sources = int(feature_encoder is not None) + int(feature_cache is not None)
    if config.model.features.enabled != (target_sources == 1):
        raise ValueError(
            "active REG/REPA features require exactly one online or cached target source"
        )
    input_sources = int(vae_codec is not None) + int(latent_cache is not None)
    if config.model.input_space == "flux2_latents":
        if input_sources != 1:
            raise ValueError(
                "latent training requires exactly one online VAE or latent-cache source"
            )
    elif input_sources:
        raise ValueError("pixel training cannot use a VAE or latent cache")
    optimizer.zero_grad()

    for epoch_index in range(state.epoch, config.train.epochs):
        logger.event(f"epoch {epoch_index + 1}/{config.train.epochs} started")
        if sampler is not None:
            sampler.set_epoch(epoch_index)
        execution_model.train()
        if feature_encoder is not None:
            feature_encoder.eval()
        if vae_codec is not None:
            vae_codec.eval()
        epoch_started = time.perf_counter()
        step_started = epoch_started
        previous_logging_seconds = 0.0
        accumulated_host_batch = 0.0
        cuda_batch_timings: list[
            tuple[
                torch.cuda.Event,
                torch.cuda.Event,
                torch.cuda.Event,
                torch.cuda.Event,
            ]
        ] = []
        iterator_started = time.perf_counter()
        loader_iterator = iter(loader)
        batch_iterator: _PreparedBatchIterator
        if device.type == "cuda":
            batch_iterator = _CUDABatchPrefetcher(loader_iterator, device)
        else:
            batch_iterator = _PreparedBatchIterator(loader_iterator, device)
        accumulated_data_wait = time.perf_counter() - iterator_started
        for batch_index in range(len(loader)):
            previous_data_wait = batch_iterator.data_wait_seconds
            batch = next(batch_iterator)
            batch_ready = time.perf_counter()
            accumulated_data_wait += (
                batch_iterator.data_wait_seconds - previous_data_wait
            )
            host_batch_started = batch_ready
            cuda_input_timing = batch_iterator.take_input_timing()
            cuda_model_started = None
            cuda_backward_finished = None
            if device.type == "cuda":
                cuda_model_started = torch.cuda.Event(enable_timing=True)
                cuda_backward_finished = torch.cuda.Event(enable_timing=True)
            pixels = batch.pixels
            labels = batch.labels
            cached_latents = batch.cached_latents
            cached_targets = batch.cached_targets
            if config.model.input_space == "pixels":
                if pixels is None:
                    raise RuntimeError("pixel training requires RGB input")
                if cached_latents is not None:
                    raise RuntimeError("pixel training received cached latents")
                clean = pixels.mul(2).sub(1)
            elif latent_cache is not None:
                if cached_latents is None:
                    raise RuntimeError(
                        "cached latent training requires latents from the loader"
                    )
                clean = cached_latents
            else:
                if vae_codec is None or pixels is None:
                    raise RuntimeError(
                        "online latent training requires a VAE and RGB input"
                    )
                if cached_latents is not None:
                    raise RuntimeError(
                        "online latent training received unexpected cached latents"
                    )
                clean = vae_codec.encode(pixels)
            timesteps = sample_timesteps(
                clean.shape[0],
                streams.timesteps,
                device,
                loc=config.train.timestep_loc,
                scale=config.train.timestep_scale,
            )
            image_noise = torch.randn(
                clean.shape,
                device=device,
                dtype=torch.float32,
                generator=streams.noise,
            )
            noisy_image = noisy_interpolate(clean, image_noise, timesteps)
            model_labels = classifier_free_labels(
                labels,
                config.model.num_classes,
                config.model.class_dropout_prob,
                streams.label_dropout,
            )

            targets = None
            reg_noise = None
            noisy_reg = None
            reg_timesteps = None
            if feature_cache is not None:
                if cached_targets is None:
                    raise RuntimeError(
                        "cached feature training requires targets from the train loader"
                    )
                targets = cached_targets
            elif feature_encoder is not None:
                if cached_targets is not None:
                    raise RuntimeError(
                        "online feature training received unexpected cached targets"
                    )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    if pixels is None:
                        raise RuntimeError(
                            "online feature training requires RGB input"
                        )
                    targets = feature_encoder.targets(pixels, config.model.grid_size)
            if targets is not None and targets.reg is not None:
                targets = scale_reg_targets(
                    targets, config.model.features.reg.target_std
                )
            if targets is not None and targets.reg is not None:
                reg_timesteps = sample_timesteps(
                    clean.shape[0],
                    streams.reg_timesteps,
                    device,
                    loc=config.train.reg_timestep_loc,
                    scale=config.train.reg_timestep_scale,
                )
                reg_noise = torch.randn(
                    targets.reg.shape,
                    device=device,
                    dtype=torch.float32,
                    generator=streams.reg_noise,
                )
                noisy_reg = noisy_interpolate(targets.reg, reg_noise, reg_timesteps)

            if cuda_model_started is not None:
                cuda_model_started.record()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = execution_model(
                    noisy_image,
                    timesteps,
                    model_labels,
                    noisy_reg=noisy_reg,
                    reg_timestep=reg_timesteps,
                    return_repa=config.model.features.repa.enabled,
                )
            repa_target = (
                repa_target_for_scope(
                    targets,
                    config.model.features.repa.token_scope,
                )
                if targets is not None and config.model.features.repa.enabled
                else None
            )
            losses = jit_losses(
                clean,
                image_noise,
                timesteps,
                output.image,
                clean_reg=targets.reg if targets is not None else None,
                reg_noise=reg_noise,
                reg_prediction=output.reg,
                reg_early_prediction=output.reg_early,
                reg_timestep=reg_timesteps,
                repa_prediction=output.repa,
                repa_target=repa_target,
                reg_weight=config.model.features.reg.loss_weight,
                repa_weight=config.model.features.repa.loss_weight,
            )
            losses.total.backward()
            if cuda_backward_finished is not None:
                cuda_backward_finished.record()
                assert (
                    cuda_input_timing is not None
                    and cuda_model_started is not None
                )
                cuda_batch_timings.append(
                    (
                        cuda_input_timing[0],
                        cuda_input_timing[1],
                        cuda_model_started,
                        cuda_backward_finished,
                    )
                )
            accumulated_host_batch += time.perf_counter() - host_batch_started

            progress = epoch_index + (batch_index + 1) / len(loader)
            lr = learning_rate_at_epoch(
                progress,
                config.train.optimizer.lr,
                config.train.optimizer.min_lr,
            )
            set_learning_rate(optimizer, lr)
            optimizer_host_started = time.perf_counter()
            cuda_optimizer_started = None
            cuda_optimizer_finished = None
            if device.type == "cuda":
                cuda_optimizer_started = torch.cuda.Event(enable_timing=True)
                cuda_optimizer_finished = torch.cuda.Event(enable_timing=True)
                cuda_optimizer_started.record()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                canonical_model.parameters(), config.train.grad_clip
            )
            optimizer.step()
            optimizer.zero_grad()
            ema.update(canonical_model)
            if cuda_optimizer_finished is not None:
                cuda_optimizer_finished.record()
            optimizer_host_seconds = time.perf_counter() - optimizer_host_started
            state.global_step += 1
            gpu_input_seconds = 0.0
            gpu_forward_backward_seconds = 0.0
            gpu_optimizer_seconds = 0.0
            if cuda_optimizer_finished is not None:
                cuda_optimizer_finished.synchronize()
                gpu_input_seconds = sum(
                    started.elapsed_time(finished)
                    for started, finished, _model_started, _backward_finished
                    in cuda_batch_timings
                ) / 1_000
                gpu_forward_backward_seconds = sum(
                    model_started.elapsed_time(backward_finished)
                    for _input_started, _input_finished, model_started, backward_finished
                    in cuda_batch_timings
                ) / 1_000
                assert cuda_optimizer_started is not None
                gpu_optimizer_seconds = (
                    cuda_optimizer_started.elapsed_time(cuda_optimizer_finished) / 1_000
                )
            now = time.perf_counter()
            elapsed = max(now - step_started, 1e-9)
            images = config.train.global_batch_size
            metrics = {
                "train/loss": losses.total.item(),
                "train/image_velocity_loss": losses.image_velocity.item(),
                "train/reg_velocity_loss": losses.reg_velocity.item(),
                "train/reg_early_velocity_loss": losses.reg_early_velocity.item(),
                "train/reg_x0_mse": losses.reg_x0_mse.item(),
                "train/reg_x0_cosine": losses.reg_x0_cosine.item(),
                "train/reg_early_x0_mse": losses.reg_early_x0_mse.item(),
                "train/reg_early_x0_cosine": losses.reg_early_x0_cosine.item(),
                "train/repa_loss": losses.repa.item(),
                "train/image_timestep_mean": timesteps.mean().item(),
                "train/reg_timestep_mean": (
                    reg_timesteps.mean().item()
                    if reg_timesteps is not None
                    else 0.0
                ),
                "train/lr": lr,
                "train/lr_adamw": lr,
                "train/lr_muon": (
                    lr
                    if config.train.optimizer.kind in {"muon", "fused_muon"}
                    else 0.0
                ),
                "train/gradient_norm": float(grad_norm),
                "train/ema_decay": ema.decay,
                "throughput/images_per_second": images / elapsed,
                "throughput/jit_tokens_per_second": images
                * config.model.grid_size**2
                / elapsed,
                "throughput/step_seconds": elapsed,
                "throughput/data_wait_seconds": accumulated_data_wait,
                "throughput/host_batch_seconds": accumulated_host_batch,
                "throughput/optimizer_host_seconds": optimizer_host_seconds,
                "throughput/logging_seconds": previous_logging_seconds,
                "throughput/gpu_input_seconds": gpu_input_seconds,
                "throughput/gpu_forward_backward_seconds": gpu_forward_backward_seconds,
                "throughput/gpu_optimizer_seconds": gpu_optimizer_seconds,
                "throughput/gpu_step_seconds": (
                    gpu_input_seconds
                    + gpu_forward_backward_seconds
                    + gpu_optimizer_seconds
                ),
                "system/peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda"
                else 0,
                "train/epoch": progress,
                "train/global_step": state.global_step,
            }
            logging_started = time.perf_counter()
            logger.log_training(
                metrics,
                step=state.global_step,
            )
            previous_logging_seconds = time.perf_counter() - logging_started
            step_started = now
            accumulated_data_wait = 0.0
            accumulated_host_batch = 0.0
            cuda_batch_timings.clear()
            if config.train.max_steps is not None and state.global_step >= config.train.max_steps:
                break
        batch_iterator.close()

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
            logger.event(f"validation started after epoch {state.epoch}")
            validation_started = time.perf_counter()
            validation_metrics = validation_callback(state.epoch, final_epoch)
            validation_seconds = time.perf_counter() - validation_started
            validation_metrics["throughput/validation_seconds"] = validation_seconds
            state.last_completed_validation_epoch = state.epoch
            config.validation.validate_now = False
            logger.log(validation_metrics, step=state.global_step)
            logger.event(f"validation finished in {validation_seconds:.1f}s")
            for name, value in validation_metrics.items():
                if "FID" in name:
                    logger.event(f"{name}: {value:.6f}")

        if rank == 0 and (
            should_checkpoint(state.epoch, config.train.epochs, config.checkpoint.interval)
            or final_epoch
        ):
            checkpoint_started = time.perf_counter()
            checkpoint_fid_count = (
                config.validation.periodic_samples
                if config.model.features.reg.enabled or not final_epoch
                else config.validation.final_samples
            )
            checkpoint_fid_suffix = (
                f"{checkpoint_fid_count // 1000}K"
                if checkpoint_fid_count % 1000 == 0
                else str(checkpoint_fid_count)
            )
            fid = validation_metrics.get(
                f"validation/FID-{checkpoint_fid_suffix}"
            )
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
                best_reg_timestep_shift=validation_metrics.get(
                    "validation/best_reg_timestep_shift"
                ),
            )
            if config.wandb.checkpoint_artifacts:
                logger.log_checkpoint(str(path), f"checkpoint-epoch-{state.epoch:04d}")
            checkpoint_seconds = time.perf_counter() - checkpoint_started
            logger.log(
                {"throughput/checkpoint_seconds": checkpoint_seconds},
                step=state.global_step,
            )
            logger.event(f"checkpoint saved in {checkpoint_seconds:.1f}s: {path}")

        logger.log(
            {"train/epoch_seconds": time.perf_counter() - epoch_started},
            step=state.global_step,
        )
        logger.event(f"epoch {state.epoch} finished in {time.perf_counter() - epoch_started:.1f}s")
        if final_epoch:
            break
    return state
