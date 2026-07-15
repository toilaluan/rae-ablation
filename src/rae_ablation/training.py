from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .checkpoint import EMA, CheckpointManager, should_checkpoint
from .config import ExperimentConfig
from .flow import flow_losses, noisy_interpolate, sample_timesteps
from .latents import LatentStatistics
from .optim import CompositeOptimizer, learning_rate_at_epoch, set_learning_rate
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
            num_workers=config.cache.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.cache.workers > 0,
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
    statistics: LatentStatistics,
    optimizer: CompositeOptimizer,
    ema: EMA,
    streams: RNGStreams,
    config: ExperimentConfig,
    checkpoint_manager: CheckpointManager,
    cache_fingerprint: str,
    validation_plan_hash: str,
    logger: RunLogger,
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
    optimizer.zero_grad()

    for epoch_index in range(state.epoch, config.train.epochs):
        logger.event(f"epoch {epoch_index + 1}/{config.train.epochs} started")
        if sampler is not None:
            sampler.set_epoch(epoch_index)
        execution_model.train()
        epoch_started = time.perf_counter()
        step_started = epoch_started
        last_compute_end = epoch_started
        accumulated_data_wait = 0.0
        for batch_index, (latent, labels, _indices) in enumerate(loader):
            batch_ready = time.perf_counter()
            accumulated_data_wait += batch_ready - last_compute_end
            latent = latent.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            clean = statistics.normalize(latent.float())
            timestep = sample_timesteps(
                clean.shape[0], tuple(clean.shape[1:]), streams.timesteps, device
            )
            noise = torch.randn(
                clean.shape, device=device, dtype=torch.float32, generator=streams.noise
            )
            x_t = noisy_interpolate(clean, noise, timestep)
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
                    output = execution_model(x_t, timestep, labels)
                losses = flow_losses(clean, noise, timestep, output.full, output.early)
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
                    "train/loss_velocity_combined": losses.total.item(),
                    "train/loss_velocity_full": losses.full.item(),
                    "train/loss_velocity_early": losses.early.item(),
                    "train/lr": lr,
                    "train/lr_adamw": lr,
                    "train/lr_muon": lr if config.train.optimizer.kind == "muon" else 0.0,
                    "train/gradient_norm": float(grad_norm),
                    "train/ema_decay": ema.decay,
                    "throughput/images_per_second": images / elapsed,
                    "throughput/latent_tokens_per_second": images
                    * clean.shape[-2]
                    * clean.shape[-1]
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
                scheduler_state={"epoch": state.epoch},
                rng_streams=streams.state_dict(),
                cache_fingerprint=cache_fingerprint,
                validation_plan_hash=validation_plan_hash,
                last_completed_validation_epoch=state.last_completed_validation_epoch,
                wandb_run_id=logger.run_id,
                resolved_config=config.as_dict(),
                fid=fid,
            )
            checkpoint_manager.export_ema(ema, path / "ema-export.safetensors")
            logger.event(f"checkpoint saved: {path}")
            if config.wandb.checkpoint_artifacts:
                logger.log_checkpoint(str(path), f"checkpoint-epoch-{state.epoch:04d}")

        logger.log(
            {"train/epoch_seconds": time.perf_counter() - epoch_started}, step=state.global_step
        )
        logger.event(f"epoch {state.epoch} finished in {time.perf_counter() - epoch_started:.1f}s")
        if final_epoch:
            break
    return state
