from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .codecs import CompactRAECodec, RAECodec, load_codecs
from .constants import (
    COMPACT_REPO,
    COMPACT_REVISION,
    IMAGENET_REPO,
    IMAGENET_REVISION,
    LATENT_SPECS,
    RAE_REPO,
    RAE_REVISION,
)
from .evaluation import fid_from_statistics

CODEC_NAMES = ("rae", "rae_compact")
LogFunction = Callable[[str], None]


@dataclass(frozen=True)
class ReconstructionParameters:
    batch_size: int = 16
    workers: int = 8
    max_samples: int | None = None
    log_interval: int = 10

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("reconstruction batch size must be positive")
        if self.workers < 0:
            raise ValueError("reconstruction workers cannot be negative")
        if self.max_samples is not None and self.max_samples < 2:
            raise ValueError("reconstruction FID requires at least two samples")
        if self.log_interval < 1:
            raise ValueError("reconstruction log interval must be positive")


def rank_indices(sample_count: int, rank: int, world_size: int) -> range:
    if sample_count < 0:
        raise ValueError("sample count cannot be negative")
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank or world size")
    return range(rank, sample_count, world_size)


class FeatureMoments:
    """Streaming float64 feature moments that can be reduced across ranks."""

    def __init__(self, dimensions: int, device: torch.device) -> None:
        if dimensions < 1:
            raise ValueError("feature dimensions must be positive")
        self.dimensions = dimensions
        self.count = torch.zeros((), dtype=torch.int64, device=device)
        self.total = torch.zeros(dimensions, dtype=torch.float64, device=device)
        self.outer = torch.zeros((dimensions, dimensions), dtype=torch.float64, device=device)

    def update(self, features: Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.dimensions:
            raise ValueError(
                f"features must have shape Nx{self.dimensions}, received {tuple(features.shape)}"
            )
        values = features.detach().to(device=self.total.device, dtype=torch.float64)
        self.total.add_(values.sum(dim=0))
        self.outer.addmm_(values.T, values)
        self.count.add_(values.shape[0])

    def synchronize(self) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for tensor in (self.count, self.total, self.outer):
                torch.distributed.all_reduce(tensor)

    def statistics(self) -> dict[str, np.ndarray]:
        count = int(self.count.item())
        if count < 2:
            raise ValueError("reconstruction FID requires at least two features")
        mean = self.total / count
        covariance = (self.outer - count * torch.outer(mean, mean)) / (count - 1)
        return {
            "mu": mean.detach().cpu().numpy(),
            "sigma": covariance.detach().cpu().numpy(),
        }


def _default_lpips(device: torch.device) -> nn.Module:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    return LearnedPerceptualImagePatchSimilarity(
        net_type="alex", reduction="none", normalize=True
    ).to(device)


def _default_feature_extractor(device: torch.device) -> nn.Module:
    from torch_fidelity.utils import create_feature_extractor

    return create_feature_extractor(
        "inception-v3-compat",
        ["2048"],
        cuda=device.type == "cuda",
        verbose=False,
    ).to(device)


class ReconstructionMetricAccumulator:
    """Accumulate matched reconstruction metrics without retaining image batches."""

    def __init__(
        self,
        device: torch.device,
        *,
        lpips_metric: nn.Module | None = None,
        feature_extractor: nn.Module | None = None,
        feature_dimensions: int = 2048,
    ) -> None:
        self.device = device
        self.lpips_metric = (lpips_metric or _default_lpips(device)).to(device).eval()
        self.feature_extractor = (
            (feature_extractor or _default_feature_extractor(device)).to(device).eval()
        )
        self.count = torch.zeros((), dtype=torch.int64, device=device)
        self.sums = {
            name: torch.zeros(3, dtype=torch.float64, device=device) for name in CODEC_NAMES
        }
        self.moments = {
            "original": FeatureMoments(feature_dimensions, device),
            **{name: FeatureMoments(feature_dimensions, device) for name in CODEC_NAMES},
        }
        self._synchronized = False

    def _lpips(self, reconstructed: Tensor, original: Tensor) -> Tensor:
        values = self.lpips_metric(reconstructed, original).reshape(-1)
        reset = getattr(self.lpips_metric, "reset", None)
        if callable(reset):
            reset()
        if values.numel() != original.shape[0]:
            raise ValueError("LPIPS metric did not return one score per image")
        return values

    def _features(self, images: Tensor) -> Tensor:
        uint8 = images.mul(255).round().to(torch.uint8)
        output = self.feature_extractor(uint8)
        features = output[0] if isinstance(output, (tuple, list)) else output
        if features.ndim != 2 or features.shape[0] != images.shape[0]:
            raise ValueError("Inception feature extractor did not return an NxD matrix")
        return features

    @torch.inference_mode()
    def update(self, original: Tensor, reconstructions: dict[str, Tensor]) -> None:
        from torchmetrics.functional.image import (
            peak_signal_noise_ratio,
            structural_similarity_index_measure,
        )

        if self._synchronized:
            raise RuntimeError("cannot update metrics after distributed synchronization")
        if set(reconstructions) != set(CODEC_NAMES):
            raise ValueError(f"reconstructions must contain {CODEC_NAMES}")
        original = original.to(device=self.device, dtype=torch.float32)
        if original.ndim != 4 or original.shape[1:] != (3, 256, 256):
            raise ValueError(
                f"original images must have shape Nx3x256x256, received {tuple(original.shape)}"
            )
        self.moments["original"].update(self._features(original))
        for name in CODEC_NAMES:
            reconstructed = reconstructions[name].to(device=self.device, dtype=torch.float32)
            if reconstructed.shape != original.shape:
                raise ValueError(f"{name} reconstruction shape does not match the input")
            reconstructed = reconstructed.clamp(0, 1)
            lpips = self._lpips(reconstructed, original)
            psnr = peak_signal_noise_ratio(
                reconstructed,
                original,
                data_range=1.0,
                reduction="none",
                dim=(1, 2, 3),
            )
            ssim = structural_similarity_index_measure(
                reconstructed,
                original,
                data_range=1.0,
                reduction="none",
            )
            self.sums[name].add_(
                torch.stack((lpips.sum(), psnr.sum(), ssim.sum())).to(torch.float64)
            )
            self.moments[name].update(self._features(reconstructed))
        self.count.add_(original.shape[0])

    def synchronize(self) -> None:
        if self._synchronized:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.count)
            for name in CODEC_NAMES:
                torch.distributed.all_reduce(self.sums[name])
        for moments in self.moments.values():
            moments.synchronize()
        self._synchronized = True

    def compute(self) -> dict[str, dict[str, float]]:
        self.synchronize()
        count = int(self.count.item())
        if count < 2:
            raise ValueError("reconstruction FID requires at least two samples")
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0
        metrics: dict[str, dict[str, float]] | None = None
        if rank == 0:
            original_statistics = self.moments["original"].statistics()
            metrics = {}
            for name in CODEC_NAMES:
                values = (self.sums[name] / count).detach().cpu().tolist()
                metrics[name] = {
                    "LPIPS": float(values[0]),
                    "PSNR": float(values[1]),
                    "SSIM": float(values[2]),
                    "rFID": fid_from_statistics(
                        self.moments[name].statistics(), original_statistics
                    ),
                }
        if distributed:
            objects = [metrics]
            torch.distributed.broadcast_object_list(objects, src=0)
            metrics = objects[0]
        if metrics is None:
            raise RuntimeError("rank zero did not produce reconstruction metrics")
        return metrics


class _ImageDataset(Dataset[Tensor]):
    def __init__(self, dataset: Any, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tensor:
        row = self.dataset[self.indices[item]]
        image = np.asarray(row["image"].convert("RGB"), dtype=np.uint8)
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255)


@torch.inference_mode()
def evaluate_reconstructions(
    dataset: Any,
    rae: RAECodec,
    compact: CompactRAECodec,
    parameters: ReconstructionParameters,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    log: LogFunction = print,
    accumulator: ReconstructionMetricAccumulator | None = None,
) -> tuple[dict[str, dict[str, float]], int]:
    parameters.validate()
    dataset_count = len(dataset)
    if parameters.max_samples is not None and parameters.max_samples > dataset_count:
        raise ValueError(
            f"requested {parameters.max_samples} samples, but validation has {dataset_count}"
        )
    sample_count = parameters.max_samples or dataset_count
    if sample_count < 2:
        raise ValueError("reconstruction FID requires at least two samples")
    indices = rank_indices(sample_count, rank, world_size)
    loader = DataLoader(
        _ImageDataset(dataset, indices),
        batch_size=parameters.batch_size,
        shuffle=False,
        num_workers=parameters.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=parameters.workers > 0,
        prefetch_factor=2 if parameters.workers > 0 else None,
    )
    metrics = accumulator or ReconstructionMetricAccumulator(device)
    started = time.perf_counter()
    total_batches = math.ceil(len(indices) / parameters.batch_size)
    processed = 0
    for batch_index, images in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            rae_latent = rae.encode(images)
            compact_latent = compact.compress(rae_latent)
            rae_reconstruction = rae.decode(rae_latent)
            compact_reconstruction = compact.decode(compact_latent)
        metrics.update(
            images,
            {
                "rae": rae_reconstruction,
                "rae_compact": compact_reconstruction,
            },
        )
        processed += images.shape[0]
        if rank == 0 and (
            batch_index % parameters.log_interval == 0 or batch_index == total_batches
        ):
            elapsed = max(time.perf_counter() - started, 1e-9)
            log(
                f"reconstruction: rank0={processed}/{len(indices)} images, "
                f"{processed / elapsed:.2f} images/s, world_size={world_size}"
            )
    return metrics.compute(), sample_count

@torch.autocast("cuda", dtype=torch.bfloat16, enabled=True)
def load_and_evaluate_reconstructions(
    parameters: ReconstructionParameters,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    log: LogFunction = print,
) -> tuple[dict[str, dict[str, float]], int]:
    from datasets import load_dataset

    parameters.validate()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    dataset = load_dataset(
        IMAGENET_REPO,
        split="validation",
        num_proc=8,
        revision=IMAGENET_REVISION,
    )
    loaded = load_codecs("both", str(device))
    if not isinstance(loaded, tuple):
        raise TypeError("combined codec loading did not return both codecs")
    rae, compact = loaded
    return evaluate_reconstructions(
        dataset,
        rae,
        compact,
        parameters,
        rank=rank,
        world_size=world_size,
        device=device,
        log=log,
    )


def build_reconstruction_report(
    metrics: dict[str, dict[str, float]],
    *,
    sample_count: int,
    parameters: ReconstructionParameters,
    world_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if set(metrics) != set(CODEC_NAMES):
        raise ValueError(f"metrics must contain {CODEC_NAMES}")
    provenance = {
        "rae": {
            "repository": RAE_REPO,
            "revision": RAE_REVISION,
        },
        "rae_compact": {
            "repository": COMPACT_REPO,
            "revision": COMPACT_REVISION,
            "base_repository": RAE_REPO,
            "base_revision": RAE_REVISION,
        },
    }
    codecs: dict[str, Any] = {}
    for name in CODEC_NAMES:
        codecs[name] = {
            **provenance[name],
            "latent_shape": list(LATENT_SPECS[name]),
            "metrics": metrics[name],
        }
    deltas = {
        metric: metrics["rae_compact"][metric] - metrics["rae"][metric]
        for metric in ("LPIPS", "PSNR", "SSIM", "rFID")
    }
    return {
        "schema_version": 1,
        "dataset": {
            "repository": IMAGENET_REPO,
            "revision": IMAGENET_REVISION,
            "split": "validation",
            "sample_count": sample_count,
            "preprocessing": ["PIL.convert(RGB)", "tensor", "scale_uint8_to_[0,1]"],
        },
        "metric_config": {
            "LPIPS": {"backbone": "alex", "normalize": True, "better": "lower"},
            "PSNR": {"data_range": 1.0, "aggregation": "mean_per_image", "better": "higher"},
            "SSIM": {"data_range": 1.0, "aggregation": "mean_per_image", "better": "higher"},
            "rFID": {
                "feature_extractor": "inception-v3-compat",
                "features": 2048,
                "image_dtype": "uint8",
                "better": "lower",
            },
            "reconstruction_range": [0.0, 1.0],
        },
        "runtime": {
            "batch_size_per_rank": parameters.batch_size,
            "workers_per_rank": parameters.workers,
            "world_size": world_size,
            "device_type": device.type,
            "codec_autocast": "bfloat16" if device.type == "cuda" else "disabled",
        },
        "codecs": codecs,
        "deltas": {"rae_compact_minus_rae": deltas},
    }


def save_reconstruction_report(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, target)
