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
from .image_latent_cache import ImageLatentCache
from .mage import MageVAECodec
from .model.tokens import classifier_free_labels
from .objective import (
    CascadeLoss,
    cascade_losses,
    gaussian_flow_path,
    latent_loss,
    sample_timesteps,
)
from .optim import CompositeOptimizer, learning_rate_at_epoch, set_learning_rate
from .rng import RNGStreams
from .semantic_token_cache import SemanticTokenCache
from .validation import should_validate
from .vision import DINOSemanticEncoder
from .wandb_utils import RunLogger


class TrainBatch(NamedTuple):
    pixels: Tensor | None
    image_latents: Tensor | None
    semantic_tokens: Tensor | None
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
        batch.image_latents.to(device, non_blocking=non_blocking)
        if batch.image_latents is not None
        else None,
        batch.semantic_tokens.to(device, non_blocking=non_blocking)
        if batch.semantic_tokens is not None
        else None,
        batch.labels.to(device, dtype=torch.long, non_blocking=non_blocking),
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


class RepresentationCacheCollator:
    def __init__(
        self,
        image_latent_cache: ImageLatentCache | None,
        semantic_token_cache: SemanticTokenCache | None,
    ) -> None:
        self.image_latent_cache = image_latent_cache
        self.semantic_token_cache = semantic_token_cache

    def __call__(self, samples: list[Any]) -> TrainBatch:
        pixel_values, label_values, index_values = zip(*samples, strict=True)
        indices = default_collate(index_values)
        pixels = (
            None
            if all(value is None for value in pixel_values)
            else default_collate(pixel_values)
        )
        return TrainBatch(
            pixels,  # type: ignore[arg-type]
            self.image_latent_cache.lookup(indices)
            if self.image_latent_cache is not None
            else None,
            self.semantic_token_cache.lookup(indices)
            if self.semantic_token_cache is not None
            else None,
            default_collate(label_values),
            indices,
        )


def distributed_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def local_batch_size(global_batch_size: int, world_size: int) -> int:
    if world_size < 1 or global_batch_size < 1 or global_batch_size % world_size:
        raise ValueError("global batch size must be divisible by distributed world size")
    return global_batch_size // world_size


def training_requires_pixels(
    architecture: str,
    image_latent_cache: ImageLatentCache | None,
    semantic_token_cache: SemanticTokenCache | None,
) -> bool:
    return image_latent_cache is None or (
        architecture != "latent" and semantic_token_cache is None
    )


def make_train_loader(
    dataset: torch.utils.data.Dataset[Any],
    config: ExperimentConfig,
    streams: RNGStreams,
    image_latent_cache: ImageLatentCache | None = None,
    semantic_token_cache: SemanticTokenCache | None = None,
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
            prefetch_factor=(
                config.data.prefetch_factor if config.data.workers > 0 else None
            ),
            collate_fn=RepresentationCacheCollator(
                image_latent_cache, semantic_token_cache
            ),
            drop_last=True,
        ),
        sampler,
    )


def extract_representations(
    batch: TrainBatch,
    architecture: str,
    mage_codec: MageVAECodec | None,
    semantic_encoder: DINOSemanticEncoder | None,
) -> tuple[Tensor, Tensor | None]:
    pixels, cached_image_latents, cached_semantic_tokens, _labels, _indices = batch
    if cached_image_latents is None:
        if pixels is None or mage_codec is None:
            raise RuntimeError("online Mage-VAE image latent extraction is unavailable")
        image_latents = mage_codec.encode_pixels(pixels)
    else:
        image_latents = cached_image_latents
    if architecture == "latent":
        return image_latents, None
    if cached_semantic_tokens is None:
        if pixels is None or semantic_encoder is None:
            raise RuntimeError("online DINO semantic extraction is unavailable")
        semantic_tokens = semantic_encoder.encode(pixels)
    else:
        semantic_tokens = cached_semantic_tokens
    return image_latents, semantic_tokens


def compute_training_losses(
    model: nn.Module,
    image_latents: Tensor,
    semantic_tokens: Tensor | None,
    labels: Tensor,
    streams: RNGStreams,
    config: ExperimentConfig,
) -> CascadeLoss:
    batch_size = image_latents.shape[0]
    device = image_latents.device
    image_timestep = sample_timesteps(
        batch_size,
        streams.image_timesteps,
        device,
        loc=config.train.timestep_loc,
        scale=config.train.timestep_scale,
        shift=config.train.timestep_shift,
    )
    image_noise = torch.randn(
        image_latents.shape,
        device=device,
        dtype=torch.float32,
        generator=streams.image_noise,
    )
    image_path = gaussian_flow_path(image_latents, image_noise, image_timestep)
    image_labels = classifier_free_labels(
        labels,
        config.model.num_classes,
        config.model.class_dropout_probability,
        streams.label_dropout,
    )
    if config.model.architecture == "latent":
        prediction = model(image_path.noisy, image_timestep, image_labels)
        return latent_loss(
            image_path,
            prediction,
            config.train.velocity_denominator_floor,
        )
    if semantic_tokens is None:
        raise ValueError("cascade training requires semantic tokens")
    semantic_timestep = sample_timesteps(
        batch_size,
        streams.semantic_timesteps,
        device,
        loc=config.train.timestep_loc,
        scale=config.train.timestep_scale,
        shift=config.train.timestep_shift,
    )
    semantic_noise = torch.randn(
        semantic_tokens.shape,
        device=device,
        dtype=torch.float32,
        generator=streams.semantic_noise,
    )
    semantic_path = gaussian_flow_path(
        semantic_tokens, semantic_noise, semantic_timestep
    )
    predicted_mask = torch.rand(
        batch_size,
        device=device,
        generator=streams.semantic_reuse_mask,
    ) < config.train.predicted_semantic_condition_probability
    output = model(
        semantic_path.noisy,
        semantic_timestep,
        labels,
        image_path.noisy,
        image_timestep,
        image_labels,
        semantic_tokens,
        predicted_mask,
    )
    return cascade_losses(
        semantic_path=semantic_path,
        image_latent_path=image_path,
        semantic_full_prediction=output.semantic.full,
        semantic_base_prediction=output.semantic.base,
        image_latent_prediction=output.image_latent,
        predicted_semantic_mask=predicted_mask,
        semantic_base_loss_weight=config.train.semantic_base_loss_weight,
        velocity_denominator_floor=config.train.velocity_denominator_floor,
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
    mage_codec: MageVAECodec | None,
    semantic_encoder: DINOSemanticEncoder | None,
    loader: DataLoader[Any],
    sampler: DistributedSampler[Any] | None,
    optimizer: CompositeOptimizer,
    ema: EMA,
    streams: RNGStreams,
    config: ExperimentConfig,
    checkpoint_manager: CheckpointManager,
    data_fingerprint: str,
    representation_provenance: dict[str, Any],
    cache_provenance: dict[str, str],
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
    optimizer.zero_grad()
    for epoch_index in range(state.epoch, config.train.epochs):
        logger.event(f"epoch {epoch_index + 1}/{config.train.epochs} started")
        if sampler is not None:
            sampler.set_epoch(epoch_index)
        execution_model.train()
        if mage_codec is not None:
            mage_codec.eval()
        if semantic_encoder is not None:
            semantic_encoder.eval()
        epoch_started = time.perf_counter()
        iterator = _PreparedBatchIterator(iter(loader), device)
        validation_metrics: dict[str, float] = {}
        for batch_index in range(len(loader)):
            batch = iterator.consume()
            if batch_index + 1 < len(loader):
                iterator.preload()
            image_latents, semantic_tokens = extract_representations(
                batch,
                config.model.architecture,
                mage_codec,
                semantic_encoder,
            )
            losses = compute_training_losses(
                execution_model,
                image_latents,
                semantic_tokens,
                batch.labels,
                streams,
                config,
            )
            losses.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                canonical_model.parameters(), config.train.grad_clip
            )
            progress = epoch_index + (batch_index + 1) / len(loader)
            learning_rate = learning_rate_at_epoch(
                progress,
                config.train.optimizer.lr,
                config.train.optimizer.min_lr,
            )
            set_learning_rate(optimizer, learning_rate)
            optimizer.step()
            optimizer.zero_grad()
            ema.update(canonical_model)
            state.global_step += 1
            reached_limit = (
                config.train.max_steps is not None
                and state.global_step >= config.train.max_steps
            )
            final_step = reached_limit or (
                epoch_index + 1 == config.train.epochs
                and batch_index + 1 == len(loader)
            )
            if logger.should_log_training(state.global_step, force=final_step):
                logger.log_training(
                    {
                        "train/loss": losses.total.item(),
                        "train/image_latent_velocity_loss": losses.image_latent_velocity.item(),
                        "train/semantic_full_velocity_loss": losses.semantic_full_velocity.item(),
                        "train/semantic_base_velocity_loss": losses.semantic_base_velocity.item(),
                        "train/predicted_semantic_condition_fraction": losses.predicted_semantic_condition_fraction.item(),
                        "train/lr": learning_rate,
                        "train/gradient_norm": float(gradient_norm),
                        "train/epoch": progress,
                        "train/global_step": state.global_step,
                        "progress/images_seen": state.global_step
                        * config.train.global_batch_size,
                    },
                    step=state.global_step,
                    force=final_step,
                )
            if reached_limit:
                break
        iterator.close()
        state.epoch = epoch_index + 1
        final_epoch = state.epoch == config.train.epochs or (
            config.train.max_steps is not None
            and state.global_step >= config.train.max_steps
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
            should_checkpoint(
                state.epoch, config.train.epochs, config.checkpoint.interval
            )
            or final_epoch
        ):
            fid_values = [
                value
                for key, value in validation_metrics.items()
                if key.startswith("validation/FID-")
            ]
            checkpoint_manager.save(
                completed_epoch=state.epoch,
                global_step=state.global_step,
                model=canonical_model,
                ema=ema,
                optimizer_state=optimizer.state_dict(),
                rng_streams=streams.state_dict(),
                architecture=config.model.architecture,
                data_fingerprint=data_fingerprint,
                validation_plan_hash=validation_plan_hash,
                validation_plan_path=validation_plan_path,
                last_completed_validation_epoch=state.last_completed_validation_epoch,
                wandb_run_id=logger.run_id,
                resolved_config=config.as_dict(),
                representation_provenance=representation_provenance,
                cache_provenance=cache_provenance,
                fid=(
                    fid_values[0]
                    if fid_values and config.checkpoint.save_best_fid
                    else None
                ),
            )
        logger.log(
            {"train/epoch_seconds": time.perf_counter() - epoch_started},
            step=state.global_step,
        )
        if final_epoch:
            break
    return state
