from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .checkpoint import EMA, CheckpointManager, should_checkpoint
from .config import ExperimentConfig
from .data import random_horizontal_flip
from .model.sit import classifier_free_labels
from .objective import (
    jit_losses,
    noisy_interpolate,
    sample_timesteps,
)
from .optim import CompositeOptimizer, learning_rate_at_epoch, set_learning_rate
from .reg import DINOv3FeatureExtractor, REGTargets
from .reg_cache import REGTokenCache
from .rng import RNGStreams
from .validation import should_validate
from .wandb_utils import RunLogger


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    last_completed_validation_epoch: int | None = None


ValidationCallback = Callable[[int, bool], dict[str, float]]


def distributed_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def make_train_loader(
    dataset: torch.utils.data.Dataset[Any],
    config: ExperimentConfig,
    streams: RNGStreams,
) -> tuple[DataLoader[Any], DistributedSampler[Any] | None]:
    rank, world_size = distributed_info()
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
            batch_size=config.train.micro_batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            generator=streams.shuffle,
            num_workers=config.data.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.data.workers > 0,
            prefetch_factor=config.data.prefetch_factor if config.data.workers > 0 else None,
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
    reg_encoder: DINOv3FeatureExtractor | None = None,
    reg_cache: REGTokenCache | None = None,
    state: TrainState | None = None,
    validation_callback: ValidationCallback | None = None,
) -> TrainState:
    state = state or TrainState()
    rank, world_size = distributed_info()
    device = next(canonical_model.parameters()).device
    accumulation = config.train.global_batch_size // (config.train.micro_batch_size * world_size)
    if accumulation < 1 or config.train.global_batch_size % (
        config.train.micro_batch_size * world_size
    ):
        raise ValueError("global batch size must divide micro batch size * world size exactly")
    target_sources = int(reg_encoder is not None) + int(reg_cache is not None)
    if config.model.reg.enabled != (target_sources == 1):
        raise ValueError("REG requires exactly one online or cached target source")
    if reg_cache is not None and config.model.reg.repa_enabled:
        raise ValueError("cached REG targets cannot provide REPA")
    optimizer.zero_grad()

    for epoch_index in range(state.epoch, config.train.epochs):
        logger.event(f"epoch {epoch_index + 1}/{config.train.epochs} started")
        if sampler is not None:
            sampler.set_epoch(epoch_index)
        execution_model.train()
        if reg_encoder is not None:
            reg_encoder.eval()
        epoch_started = time.perf_counter()
        step_started = epoch_started
        last_compute_end = epoch_started
        accumulated_data_wait = 0.0
        for batch_index, (pixels, labels, indices) in enumerate(loader):
            batch_ready = time.perf_counter()
            accumulated_data_wait += batch_ready - last_compute_end
            pixels = pixels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pixels = random_horizontal_flip(
                pixels,
                streams.augmentations,
                config.train.horizontal_flip_probability,
            )
            clean = pixels.mul(2).sub(1)
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
            if reg_cache is not None:
                targets = REGTargets(
                    reg_cache.lookup(indices).to(device=device, dtype=torch.float32),
                    None,
                )
            elif reg_encoder is not None:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    targets = reg_encoder.targets(pixels, config.model.grid_size)
            if targets is not None:
                reg_noise = torch.randn(
                    targets.reg_tokens.shape,
                    device=device,
                    dtype=torch.float32,
                    generator=streams.reg_noise,
                )
                noisy_reg = noisy_interpolate(targets.reg_tokens, reg_noise, timesteps)

            update = (batch_index + 1) % accumulation == 0
            sync_context = (
                execution_model.no_sync()
                if isinstance(execution_model, DistributedDataParallel) and not update
                else contextlib.nullcontext()
            )
            with sync_context:
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
                        return_repa=targets is not None and targets.repa is not None,
                    )
                losses = jit_losses(
                    clean,
                    image_noise,
                    timesteps,
                    output.image,
                    clean_reg=targets.reg_tokens if targets is not None else None,
                    reg_noise=reg_noise,
                    reg_prediction=output.reg,
                    repa_prediction=output.repa,
                    repa_target=targets.repa if targets is not None else None,
                    reg_weight=config.model.reg.cls_loss_weight,
                    repa_weight=config.model.reg.repa_loss_weight,
                )
                (losses.total / accumulation).backward()
            last_compute_end = time.perf_counter()
            if not update:
                continue

            progress = epoch_index + (batch_index + 1) / len(loader)
            lr = learning_rate_at_epoch(
                progress,
                config.train.optimizer.lr,
                config.train.optimizer.min_lr,
            )
            set_learning_rate(optimizer, lr)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                canonical_model.parameters(), config.train.grad_clip
            )
            optimizer.step()
            optimizer.zero_grad()
            ema.update(canonical_model)
            state.global_step += 1
            now = time.perf_counter()
            elapsed = max(now - step_started, 1e-9)
            images = config.train.global_batch_size
            logger.log_training(
                {
                    "train/loss": losses.total.item(),
                    "train/image_velocity_loss": losses.image_velocity.item(),
                    "train/reg_velocity_loss": losses.reg_velocity.item(),
                    "train/reg_x0_mse": losses.reg_x0_mse.item(),
                    "train/reg_x0_cosine": losses.reg_x0_cosine.item(),
                    "train/repa_loss": losses.repa.item(),
                    "train/image_timestep_mean": timesteps.mean().item(),
                    "train/lr": lr,
                    "train/lr_adamw": lr,
                    "train/lr_muon": lr if config.train.optimizer.kind == "muon" else 0.0,
                    "train/gradient_norm": float(grad_norm),
                    "train/ema_decay": ema.decay,
                    "throughput/images_per_second": images / elapsed,
                    "throughput/jit_tokens_per_second": images
                    * config.model.grid_size**2
                    / elapsed,
                    "throughput/step_seconds": elapsed,
                    "throughput/data_wait_seconds": accumulated_data_wait,
                    "system/peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else 0,
                    "train/epoch": progress,
                    "train/global_step": state.global_step,
                },
                step=state.global_step,
            )
            step_started = now
            last_compute_end = now
            accumulated_data_wait = 0.0
            if config.train.max_steps is not None and state.global_step >= config.train.max_steps:
                break

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
            validation_metrics = validation_callback(state.epoch, final_epoch)
            state.last_completed_validation_epoch = state.epoch
            config.validation.validate_now = False
            logger.log(validation_metrics, step=state.global_step)
            for name, value in validation_metrics.items():
                if "FID" in name:
                    logger.event(f"{name}: {value:.6f}")

        if rank == 0 and (
            should_checkpoint(state.epoch, config.train.epochs, config.checkpoint.interval)
            or final_epoch
        ):
            fid = validation_metrics.get("validation/FID-50K")
            if fid is None:
                fid = validation_metrics.get("validation/FID-5K")
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
                reg_cache_fingerprint=(
                    reg_cache.fingerprint if reg_cache is not None else None
                ),
                fid=fid,
            )
            logger.event(f"checkpoint saved: {path}")
            if config.wandb.checkpoint_artifacts:
                logger.log_checkpoint(str(path), f"checkpoint-epoch-{state.epoch:04d}")

        logger.log(
            {"train/epoch_seconds": time.perf_counter() - epoch_started},
            step=state.global_step,
        )
        logger.event(f"epoch {state.epoch} finished in {time.perf_counter() - epoch_started:.1f}s")
        if final_epoch:
            break
    return state
