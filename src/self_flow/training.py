from __future__ import annotations

import time
from collections.abc import Callable
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
from .latent_cache import LatentCache
from .model import classifier_free_labels
from .objective import SelfFlowLoss, self_flow_loss
from .optim import CompositeOptimizer, learning_rate_at_epoch, set_learning_rate
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


def _add_optional_group_metrics(
    metrics: dict[str, float], group_names: list[str], losses: SelfFlowLoss
) -> None:
    if losses.sigreg_groups:
        for name, value in zip(group_names, losses.sigreg_groups, strict=True):
            metrics[f"train/sigreg_{name}"] = value.item()
    if losses.feature_mean_norms:
        for name, mean_norm, std_mean, covariance_rms in zip(
            group_names,
            losses.feature_mean_norms,
            losses.feature_std_means,
            losses.feature_covariance_offdiag_rms,
            strict=True,
        ):
            metrics[f"features/{name}_mean_norm"] = mean_norm.item()
            metrics[f"features/{name}_std_mean"] = std_mean.item()
            metrics[f"features/{name}_covariance_offdiag_rms"] = covariance_rms.item()


class TrainBatch(NamedTuple):
    pixels: Tensor | None
    labels: Tensor
    indices: Tensor
    cached_latents: Tensor | None


def prepare_train_batch(
    batch: TrainBatch,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> TrainBatch:
    device = torch.device(device)
    return TrainBatch(
        prepare_pixels(batch.pixels, device, non_blocking=non_blocking)
        if batch.pixels is not None
        else None,
        batch.labels.to(device=device, non_blocking=non_blocking),
        batch.indices,
        batch.cached_latents.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        if batch.cached_latents is not None
        else None,
    )


class _TrainBatchCollator:
    def __init__(self, config: ExperimentConfig, dataset_size: int) -> None:
        self.config = config
        self.dataset_size = dataset_size
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
        cached_latents = None
        if self.config.vae.cache.enabled:
            if self._latent_cache is None:
                self._latent_cache = LatentCache(self.config, self.dataset_size)
            cached_latents = self._latent_cache.lookup(indices)
        return TrainBatch(pixels, labels, indices, cached_latents)


def distributed_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def local_batch_size(global_batch_size: int, world_size: int) -> int:
    if world_size < 1 or global_batch_size < 1 or global_batch_size % world_size:
        raise ValueError("global batch size must be positive and divisible by world size")
    return global_batch_size // world_size


def make_train_loader(
    dataset: torch.utils.data.Dataset[Any],
    config: ExperimentConfig,
    streams: RNGStreams,
) -> tuple[DataLoader[Any], DistributedSampler[Any] | None]:
    rank, world_size = distributed_info()
    rank_batch = local_batch_size(config.train.global_batch_size, world_size)
    if config.objective.kind == "sigreg" and rank_batch < 2:
        raise ValueError("SIGReg requires a local-rank batch size of at least two")
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
    loader = DataLoader(
        dataset,
        batch_size=rank_batch,
        sampler=sampler,
        shuffle=sampler is None,
        generator=streams.shuffle,
        num_workers=config.data.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.data.workers > 0,
        prefetch_factor=config.data.prefetch_factor if config.data.workers > 0 else None,
        collate_fn=_TrainBatchCollator(config, len(dataset)),
        drop_last=True,
    )
    return loader, sampler


def _clean_latents(
    batch: TrainBatch,
    *,
    vae_codec: Flux2Codec | None,
    latent_cache: LatentCache | None,
) -> Tensor:
    if latent_cache is not None:
        if batch.cached_latents is None:
            raise RuntimeError("cached-latent training received no cached latents")
        return batch.cached_latents
    if vae_codec is None or batch.pixels is None:
        raise RuntimeError("online FLUX.2 training requires pixels and a VAE")
    if batch.cached_latents is not None:
        raise RuntimeError("online FLUX.2 training received cached latents")
    return vae_codec.encode(batch.pixels)


def train(
    *,
    canonical_model: nn.Module,
    execution_model: nn.Module,
    loader: DataLoader[Any],
    sampler: DistributedSampler[Any] | None,
    optimizer: CompositeOptimizer,
    ema: EMA | None,
    streams: RNGStreams,
    config: ExperimentConfig,
    checkpoint_manager: CheckpointManager,
    data_fingerprint: str,
    validation_plan_hash: str,
    validation_plan_path: str | Path,
    logger: RunLogger,
    vae_codec: Flux2Codec | None = None,
    latent_cache: LatentCache | None = None,
    state: TrainState | None = None,
    validation_callback: ValidationCallback | None = None,
) -> TrainState:
    state = state or TrainState()
    rank, world_size = distributed_info()
    device = next(canonical_model.parameters()).device
    if loader.batch_size != local_batch_size(config.train.global_batch_size, world_size):
        raise ValueError("loader batch size does not match global batch size / world size")
    if int(vae_codec is not None) + int(latent_cache is not None) != 1:
        raise ValueError("training requires exactly one online VAE or latent-cache source")
    if (config.objective.kind == "ema") != (ema is not None):
        raise ValueError("EMA exists if and only if objective.kind=ema")
    optimizer.zero_grad()

    for epoch_index in range(state.epoch, config.train.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch_index)
        execution_model.train()
        if vae_codec is not None:
            vae_codec.eval()
        epoch_started = time.perf_counter()
        logger.event(f"epoch {epoch_index + 1}/{config.train.epochs} started")
        for batch_index, source_batch in enumerate(loader):
            step_started = time.perf_counter()
            batch = prepare_train_batch(source_batch, device, non_blocking=device.type == "cuda")
            clean = _clean_latents(batch, vae_codec=vae_codec, latent_cache=latent_cache)
            labels = classifier_free_labels(
                batch.labels,
                config.model.num_classes,
                config.model.class_dropout_prob,
                streams.label_dropout,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                losses = self_flow_loss(
                    execution_model,
                    clean,
                    labels,
                    streams,
                    config,
                    ema_model=ema.model if ema is not None else None,
                    collect_diagnostics=(state.global_step + 1) % logger.log_interval == 0,
                )
            losses.total.backward()
            progress = epoch_index + (batch_index + 1) / len(loader)
            lr = learning_rate_at_epoch(
                progress, config.train.optimizer.lr, config.train.optimizer.min_lr
            )
            set_learning_rate(optimizer, lr)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                canonical_model.parameters(), config.train.grad_clip
            )
            optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(canonical_model)
            state.global_step += 1

            elapsed = max(time.perf_counter() - step_started, 1e-9)
            metrics: dict[str, float] = {
                "train/loss": losses.total.item(),
                "train/mixed_velocity_loss": losses.mixed_velocity.item(),
                "train/cleaner_velocity_loss": losses.cleaner_velocity.item(),
                "train/alignment_loss": losses.alignment.item(),
                "train/sigreg_loss": losses.sigreg.item(),
                "train/mixed_timestep_mean": losses.mixed_timestep_mean.item(),
                "train/cleaner_timestep_mean": losses.cleaner_timestep_mean.item(),
                "train/lr": lr,
                "train/lr_adamw": lr,
                "train/lr_muon": lr if config.train.optimizer.kind == "muon" else 0.0,
                "train/gradient_norm": float(grad_norm),
                "train/epoch": progress,
                "train/global_step": float(state.global_step),
                "throughput/images_per_second": config.train.global_batch_size / elapsed,
                "throughput/tokens_per_second": config.train.global_batch_size
                * config.model.grid_size**2
                / elapsed,
                "throughput/step_seconds": elapsed,
                "system/peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda"
                else 0.0,
            }
            for layer, value in zip(
                config.objective.align_layers, losses.alignment_layers, strict=True
            ):
                metrics[f"train/alignment_layer_{layer}"] = value.item()
            group_names = [f"mixed_layer_{layer}" for layer in config.objective.align_layers] + [
                f"cleaner_layer_{layer}" for layer in config.objective.align_layers
            ]
            _add_optional_group_metrics(metrics, group_names, losses)
            logger.log_training(metrics, step=state.global_step)
            if config.train.max_steps is not None and state.global_step >= config.train.max_steps:
                break

        state.epoch = epoch_index + 1
        final_epoch = state.epoch == config.train.epochs or (
            config.train.max_steps is not None and state.global_step >= config.train.max_steps
        )
        validation_metrics: dict[str, float] = {}
        if validation_callback is not None and should_validate(
            state.epoch,
            state.epoch if final_epoch else config.train.epochs,
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
            count = (
                config.validation.final_samples
                if final_epoch
                else config.validation.periodic_samples
            )
            suffix = f"{count // 1000}K" if count % 1000 == 0 else str(count)
            fid = (
                validation_metrics.get(f"validation/FID-{suffix}")
                if config.checkpoint.save_best_fid
                else None
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
                latent_cache_fingerprint=latent_cache.fingerprint
                if latent_cache is not None
                else None,
                fid=fid,
            )
            if config.wandb.checkpoint_artifacts:
                logger.log_checkpoint(str(path), f"checkpoint-epoch-{state.epoch:04d}")
            logger.event(f"checkpoint saved: {path}")
        logger.log(
            {"train/epoch_seconds": time.perf_counter() - epoch_started}, step=state.global_step
        )
        if final_epoch:
            break
    return state
