from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.utils.data._utils.collate import default_collate

from .config import ExperimentConfig
from .data import prepare_pixels
from .pixel_codec import (
    PixelCodec,
    PixelTokenStatistics,
    adaptive_alignment_weight,
    vision_foundation_losses,
)
from .token_cache import TokenCache, normalize_tokens


class CodecBatch(NamedTuple):
    pixels: Tensor
    rae_tokens: Tensor


class CodecBatchCollator:
    def __init__(self, cache: TokenCache) -> None:
        self.cache = cache

    def __call__(self, samples: list[Any]) -> CodecBatch:
        pixels, _labels, indices = zip(*samples, strict=True)
        if any(value is None for value in pixels):
            raise ValueError("pixel codec training requires RGB images")
        index_tensor = default_collate(indices)
        return CodecBatch(default_collate(pixels), self.cache.raw_lookup(index_tensor))


def distributed_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def make_codec_loader(
    dataset: torch.utils.data.Dataset[Any],
    cache: TokenCache,
    config: ExperimentConfig,
    *,
    shuffle: bool = True,
    drop_last: bool = True,
) -> tuple[DataLoader[Any], DistributedSampler[Any] | None]:
    rank, world_size = distributed_info()
    if config.codec_train.global_batch_size % world_size:
        raise ValueError("codec global batch size must be divisible by world size")
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=config.train.seed,
            drop_last=drop_last,
        )
        if world_size > 1
        else None
    )
    return (
        DataLoader(
            dataset,
            batch_size=config.codec_train.global_batch_size // world_size,
            sampler=sampler,
            shuffle=shuffle and sampler is None,
            num_workers=config.data.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.data.workers > 0,
            prefetch_factor=config.data.prefetch_factor if config.data.workers > 0 else None,
            collate_fn=CodecBatchCollator(cache),
            drop_last=drop_last,
        ),
        sampler,
    )


def make_statistics_loader(
    dataset: torch.utils.data.Dataset[Any],
    cache: TokenCache,
    config: ExperimentConfig,
) -> DataLoader[Any]:
    rank, world_size = distributed_info()
    local = Subset(dataset, list(range(rank, len(dataset), world_size)))
    return DataLoader(
        local,
        batch_size=max(1, config.codec_train.global_batch_size // world_size),
        shuffle=False,
        num_workers=config.data.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.data.workers > 0,
        prefetch_factor=config.data.prefetch_factor if config.data.workers > 0 else None,
        collate_fn=CodecBatchCollator(cache),
        drop_last=False,
    )
class LPIPSReconstructionLoss(nn.Module):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        if weight <= 0:
            raise ValueError("LPIPS weight must be positive")
        import lpips

        self.weight = weight
        self.lpips = lpips.LPIPS(net="alex").eval()
        for parameter in self.lpips.parameters():
            parameter.requires_grad_(False)

    def forward(self, reconstruction: Tensor, pixels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        l1 = F.l1_loss(reconstruction.float(), pixels.float())
        perceptual = self.lpips(
            reconstruction.float().mul(2).sub(1), pixels.float().mul(2).sub(1)
        ).mean()
        # perceptual = torch.zeros((), device=reconstruction.device)
        return l1 + self.weight * perceptual, l1, perceptual


@dataclass(frozen=True)
class CodecStepMetrics:
    total: float
    reconstruction: float
    l1: float
    lpips: float
    vf: float
    marginal_cosine: float
    marginal_distance_matrix: float
    alignment_weight: float
    gradient_norm: float


def codec_training_step(
    model: PixelCodec,
    execution_model: nn.Module,
    batch: CodecBatch,
    reconstruction_loss: LPIPSReconstructionLoss,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    cache: TokenCache,
    device: torch.device,
) -> CodecStepMetrics:
    pixels = prepare_pixels(batch.pixels, device, non_blocking=True)
    rae = normalize_tokens(
        batch.rae_tokens.to(device, non_blocking=True),
        cache.mean.to(device),
        cache.std.to(device),
    )
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        reconstruction, pixel_tokens = execution_model(pixels)
        reconstruction_total, l1, perceptual = reconstruction_loss(reconstruction, pixels)
        vf = vision_foundation_losses(
            pixel_tokens,
            rae,
            cosine_margin=config.pixel_codec.cosine_margin,
            distance_margin=config.pixel_codec.distance_margin,
        )
    weight = adaptive_alignment_weight(
        reconstruction_total,
        vf.total,
        pixel_tokens,
        gradient_ratio=config.pixel_codec.alignment_gradient_ratio,
    )
    total = reconstruction_total + weight * vf.total
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.codec_train.grad_clip)
    optimizer.step()
    optimizer.zero_grad()
    return CodecStepMetrics(
        total.item(),
        reconstruction_total.item(),
        l1.item(),
        perceptual.item(),
        vf.total.item(),
        vf.marginal_cosine.item(),
        vf.marginal_distance_matrix.item(),
        weight.item(),
        float(gradient_norm),
    )


def codec_learning_rate(config: ExperimentConfig, progress: float) -> float:
    settings = config.codec_train
    if progress < settings.warmup_epochs:
        return settings.lr * (progress + 1) / max(settings.warmup_epochs, 1)
    fraction = (progress - settings.warmup_epochs) / (
        settings.epochs - settings.warmup_epochs
    )
    fraction = min(max(fraction, 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * fraction))
    return settings.min_lr + (settings.lr - settings.min_lr) * cosine


@torch.inference_mode()
def validate_codec(
    model: PixelCodec,
    loader: Iterable[Any],
    reconstruction_loss: LPIPSReconstructionLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    for pixels, _labels, _indices in loader:
        prepared = prepare_pixels(pixels, device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            reconstruction, _latent = model(prepared)
            loss, l1, perceptual = reconstruction_loss(reconstruction, prepared)
        mse = (reconstruction.float() - prepared).square().flatten(1).mean(dim=1)
        psnr = -10 * torch.log10(mse.clamp_min(1e-12))
        count = prepared.shape[0]
        totals += torch.tensor(
            [loss.item() * count, l1.item() * count, perceptual.item() * count, psnr.sum().item(), count],
            device=device,
            dtype=torch.float64,
        )
    if distributed_info()[1] > 1:
        torch.distributed.all_reduce(totals)
    count = totals[-1].item()
    return {
        "codec/validation_loss": totals[0].item() / count,
        "codec/validation_l1": totals[1].item() / count,
        "codec/validation_lpips": totals[2].item() / count,
        "codec/validation_psnr": totals[3].item() / count,
    }


@torch.inference_mode()
def compute_pixel_token_statistics(
    model: PixelCodec,
    loader: Iterable[CodecBatch],
    cache: TokenCache,
    device: torch.device,
) -> PixelTokenStatistics:
    model.eval()
    channels = model.config.latent_channels
    total = torch.zeros(channels, device=device, dtype=torch.float64)
    square = torch.zeros_like(total)
    count = torch.zeros((), device=device, dtype=torch.float64)
    for batch in loader:
        pixels = prepare_pixels(batch.pixels, device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            latent = model.encode(pixels)
        values = latent.double()
        total += values.sum(dim=(0, 2, 3))
        square += values.square().sum(dim=(0, 2, 3))
        count += values.shape[0] * values.shape[2] * values.shape[3]
    if distributed_info()[1] > 1:
        for value in (total, square, count):
            torch.distributed.all_reduce(value)
    mean = total / count
    variance = (square / count - mean.square()).clamp_min(0)
    std = variance.sqrt()
    if torch.any(std <= 0):
        raise ValueError("pixel token statistics contain zero variance")

    cross = torch.zeros(channels, device=device, dtype=torch.float64)
    cross_count = torch.zeros((), device=device, dtype=torch.float64)
    for batch in loader:
        pixels = prepare_pixels(batch.pixels, device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            latent = model.encode(pixels)
        pixel = (latent.double() - mean[None, :, None, None]) / std[None, :, None, None]
        rae = normalize_tokens(
            batch.rae_tokens.to(device, non_blocking=True),
            cache.mean.to(device),
            cache.std.to(device),
        ).double()
        cross += (pixel * rae).sum(dim=(0, 2, 3))
        cross_count += pixel.shape[0] * pixel.shape[2] * pixel.shape[3]
    if distributed_info()[1] > 1:
        for value in (cross, cross_count):
            torch.distributed.all_reduce(value)
    correlation = (cross / cross_count).clamp(-1, 1)
    result = PixelTokenStatistics(
        mean.float().cpu(), std.float().cpu(), correlation.float().cpu()
    )
    result.validate(channels)
    return result
