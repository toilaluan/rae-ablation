from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate

from .checkpoint import EMA
from .config import ExperimentConfig
from .data import prepare_pixels
from .pixel_decoder import (
    ReconstructionObjective,
    TransformerPixelDecoder,
    discriminator_hinge_loss,
)
from .vision import DINOEndpointEncoder, vision_fingerprint
from .wandb_utils import RunLogger


@dataclass
class DecoderState:
    epoch: int = 0
    global_step: int = 0
    wandb_run_id: str | None = None


def _distributed() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _collate(samples: list[Any]) -> tuple[Tensor, Tensor]:
    pixels, _labels, indices = zip(*samples, strict=True)
    if any(value is None for value in pixels):
        raise ValueError("pixel decoder training requires RGB pixels")
    return default_collate(pixels), default_collate(indices)  # type: ignore[arg-type]


def make_decoder_loader(
    dataset: torch.utils.data.Dataset[Any],
    config: ExperimentConfig,
    *,
    training: bool,
) -> tuple[DataLoader[Any], DistributedSampler[Any] | None]:
    rank, world_size = _distributed()
    global_batch = config.decoder.train.global_batch_size
    if global_batch % world_size:
        raise ValueError("decoder global batch size must be divisible by world size")
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=training,
            seed=config.decoder.train.seed,
            drop_last=training,
        )
        if world_size > 1
        else None
    )
    workers = config.data.workers
    return (
        DataLoader(
            dataset,
            batch_size=global_batch // world_size,
            sampler=sampler,
            shuffle=training and sampler is None,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            prefetch_factor=config.data.prefetch_factor if workers > 0 else None,
            collate_fn=_collate,
            drop_last=training,
        ),
        sampler,
    )


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _set_requires_grad(model: nn.Module, enabled: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)


def _reduce(values: Tensor) -> Tensor:
    result = values.detach().float()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(result)
        result /= torch.distributed.get_world_size()
    return result


def _lr(progress: float, config: ExperimentConfig) -> float:
    train = config.decoder.train
    cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi))).item()
    return train.min_lr + (train.lr - train.min_lr) * cosine


def _set_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _save_grid(target: Tensor, output: Tensor, path: Path) -> None:
    count = min(target.shape[0], 8)
    if count < 1:
        return
    target_u8 = target[:count].detach().float().clamp(0, 1).mul(255).byte().cpu()
    output_u8 = output[:count].detach().float().clamp(0, 1).mul(255).byte().cpu()
    size = target.shape[-1]
    canvas = Image.new("RGB", (count * size, 2 * size + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for index in range(count):
        real = Image.fromarray(target_u8[index].permute(1, 2, 0).numpy())
        fake = Image.fromarray(output_u8[index].permute(1, 2, 0).numpy())
        canvas.paste(real, (index * size, 0))
        canvas.paste(fake, (index * size, size + 24))
    draw.text((4, size + 4), "top: source | bottom: reconstruction", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".partial-{os.getpid()}.png")
    canvas.save(temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def validate_decoder(
    decoder: nn.Module,
    ema: EMA,
    feature_encoder: DINOEndpointEncoder,
    loader: DataLoader[Any],
    config: ExperimentConfig,
    output: Path,
) -> dict[str, float]:
    rank, _world_size = _distributed()
    device = next(_unwrap(decoder).parameters()).device
    evaluation = TransformerPixelDecoder(config.decoder).to(device)
    ema.copy_to(evaluation)
    evaluation.eval()
    total = torch.zeros(4, device=device)
    preview: tuple[Tensor, Tensor] | None = None
    for batch_index, (pixels, _indices) in enumerate(loader):
        if batch_index >= config.decoder.train.validation_batches:
            break
        raw_pixels = pixels.to(device, non_blocking=True)
        target = prepare_pixels(raw_pixels, device)
        tokens = feature_encoder.encode_target(raw_pixels)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            reconstructed = evaluation(tokens)
        difference = reconstructed.float() - target
        batch = target.shape[0]
        total[0] += difference.abs().flatten(1).mean(1).sum()
        total[1] += difference.square().flatten(1).mean(1).sum()
        total[2] += batch
        total[3] += 1
        if preview is None:
            preview = target, reconstructed
    total = _reduce(total)
    count = max(total[2].item(), 1)
    mse = total[1].item() / count
    metrics = {
        "decoder/validation_l1": total[0].item() / count,
        "decoder/validation_mse": mse,
        "decoder/validation_psnr": -10 * torch.log10(torch.tensor(max(mse, 1e-12))).item(),
    }
    if rank == 0 and preview is not None:
        _save_grid(*preview, output)
    del evaluation
    return metrics


def save_decoder_checkpoint(
    directory: Path,
    state: DecoderState,
    decoder: nn.Module,
    ema: EMA,
    discriminator: nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> Path:
    name = f"epoch-{state.epoch:04d}"
    target = directory / name
    if target.exists():
        raise FileExistsError(target)
    temporary = directory / f".{name}.partial-{os.getpid()}"
    temporary.mkdir(parents=True)
    save_file(
        {key: value.detach().contiguous() for key, value in _unwrap(decoder).state_dict().items()},
        str(temporary / "decoder.safetensors"),
    )
    save_file(
        {key: value.detach().contiguous() for key, value in ema.state_dict().items()},
        str(temporary / "ema.safetensors"),
    )
    save_file(
        {
            key: value.detach().contiguous()
            for key, value in _unwrap(discriminator).state_dict().items()
        },
        str(temporary / "discriminator.safetensors"),
    )
    torch.save(
        {
            "schema_version": 14,
            "epoch": state.epoch,
            "global_step": state.global_step,
            "wandb_run_id": state.wandb_run_id,
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "vision_fingerprint": vision_fingerprint(config.vision),
            "cpu_rng_state": torch.random.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
        temporary / "training.pt",
    )
    (temporary / "config.json").write_text(
        json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, target)
    link = directory / "last"
    temporary_link = directory / f".last-{os.getpid()}"
    temporary_link.symlink_to(name, target_is_directory=True)
    os.replace(temporary_link, link)
    return target


def load_decoder_checkpoint(
    path: Path,
    decoder: nn.Module,
    ema: EMA,
    discriminator: nn.Module,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
) -> DecoderState:
    resolved = (path / "last").resolve() if (path / "last").is_dir() else path.resolve()
    state = torch.load(resolved / "training.pt", map_location="cpu", weights_only=False)
    if state.get("schema_version") != 14:
        raise ValueError("decoder checkpoint uses an incompatible schema")
    expected = vision_fingerprint(config.vision)
    if state.get("vision_fingerprint") != expected:
        raise ValueError("decoder checkpoint DINO provenance mismatch")
    _unwrap(decoder).load_state_dict(load_file(str(resolved / "decoder.safetensors")))
    ema.load_state_dict(load_file(str(resolved / "ema.safetensors")))
    _unwrap(discriminator).load_state_dict(
        load_file(str(resolved / "discriminator.safetensors"))
    )
    generator_optimizer.load_state_dict(state["generator_optimizer"])
    discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
    torch.random.set_rng_state(state["cpu_rng_state"])
    if state.get("cuda_rng_states") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])
    return DecoderState(
        int(state["epoch"]), int(state["global_step"]), state.get("wandb_run_id")
    )


def train_decoder(
    *,
    canonical_decoder: nn.Module,
    execution_decoder: nn.Module,
    feature_encoder: DINOEndpointEncoder,
    canonical_discriminator: nn.Module,
    execution_discriminator: nn.Module,
    objective: ReconstructionObjective,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    sampler: DistributedSampler[Any] | None,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    ema: EMA,
    config: ExperimentConfig,
    output: Path,
    logger: RunLogger,
    state: DecoderState | None = None,
) -> DecoderState:
    state = state or DecoderState()
    state.wandb_run_id = logger.run_id
    rank, _world_size = _distributed()
    device = next(canonical_decoder.parameters()).device
    train = config.decoder.train
    total_steps = train.max_steps or (train.epochs * len(train_loader))
    gan_start = int(config.decoder.loss.gan_start_fraction * total_steps)
    output.mkdir(parents=True, exist_ok=True)
    for epoch_index in range(state.epoch, train.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch_index)
        canonical_decoder.train()
        canonical_discriminator.train()
        feature_encoder.eval()
        for pixels, _indices in train_loader:
            started = time.perf_counter()
            progress = min((state.global_step + 1) / max(total_steps, 1), 1.0)
            learning_rate = _lr(progress, config)
            _set_lr(generator_optimizer, learning_rate)
            _set_lr(discriminator_optimizer, learning_rate)
            raw_pixels = pixels.to(device, non_blocking=True)
            target = prepare_pixels(raw_pixels, device)
            tokens = feature_encoder.encode_target(raw_pixels)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                reconstructed = execution_decoder(tokens)
            gan_active = config.decoder.loss.gan_enabled and state.global_step >= gan_start
            discriminator_loss = target.new_zeros(())
            if gan_active:
                discriminator_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    real_logits = execution_discriminator(target)
                    fake_logits = execution_discriminator(reconstructed.detach())
                discriminator_loss = discriminator_hinge_loss(real_logits, fake_logits)
                discriminator_loss.backward()
                discriminator_optimizer.step()

            generator_optimizer.zero_grad(set_to_none=True)
            _set_requires_grad(canonical_discriminator, False)
            with torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                fake_logits_for_generator = (
                    execution_discriminator(reconstructed) if gan_active else None
                )
            losses = objective(
                reconstructed, target, fake_logits=fake_logits_for_generator
            )
            losses.total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                canonical_decoder.parameters(), train.grad_clip
            )
            generator_optimizer.step()
            _set_requires_grad(canonical_discriminator, True)
            ema.update(canonical_decoder)
            state.global_step += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = max(time.perf_counter() - started, 1e-9)
            metrics = {
                "decoder/epoch": epoch_index + state.global_step / max(total_steps, 1),
                "decoder/global_step": state.global_step,
                "decoder/loss": losses.total.item(),
                "decoder/l1": losses.l1.item(),
                "decoder/lpips": losses.lpips.item(),
                "decoder/generator_adversarial": losses.adversarial.item(),
                "decoder/discriminator": discriminator_loss.item(),
                "decoder/gan_active": float(gan_active),
                "decoder/lr": learning_rate,
                "decoder/gradient_norm": float(grad_norm),
                "throughput/images_per_second": target.shape[0] / elapsed,
                "throughput/step_seconds": elapsed,
                "throughput/data_wait_seconds": 0.0,
                "system/peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30
                if device.type == "cuda"
                else 0.0,
            }
            logger.log_decoder_training(metrics, state.global_step)
            if rank == 0 and state.global_step % train.sample_interval == 0:
                preview = output / "previews" / f"step-{state.global_step:08d}.png"
                _save_grid(target, reconstructed, preview)
                logger.log_grid(
                    str(preview),
                    f"decoder step {state.global_step}",
                    state.global_step,
                    key="decoder/train_preview",
                )
            if train.max_steps is not None and state.global_step >= train.max_steps:
                break
        state.epoch = epoch_index + 1
        final = state.epoch == train.epochs or (
            train.max_steps is not None and state.global_step >= train.max_steps
        )
        if state.epoch % train.validation_interval == 0 or final:
            preview = output / "previews" / f"validation-{state.epoch:04d}.png"
            validation_metrics = validate_decoder(
                canonical_decoder,
                ema,
                feature_encoder,
                validation_loader,
                config,
                preview,
            )
            logger.log(validation_metrics, state.global_step)
            if rank == 0 and preview.is_file():
                logger.log_grid(
                    str(preview),
                    f"decoder EMA epoch {state.epoch}",
                    state.global_step,
                    key="decoder/validation_preview",
                )
        if rank == 0:
            checkpoint = save_decoder_checkpoint(
                output / "checkpoints",
                state,
                canonical_decoder,
                ema,
                canonical_discriminator,
                generator_optimizer,
                discriminator_optimizer,
                config,
            )
            if config.wandb.checkpoint_artifacts:
                logger.log_checkpoint(str(checkpoint), f"decoder-epoch-{state.epoch:04d}")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if final:
            break
    return state
