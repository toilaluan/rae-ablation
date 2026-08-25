from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .config import IMAGE_SIZE, TOKEN_COUNT, TOKEN_DIM, DataConfig, ExperimentConfig
from .representations import ChannelStatistics, RepresentationManifest, patchify_rgb


def prepare_pixels(
    pixels: Tensor,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> Tensor:
    source_is_uint8 = pixels.dtype == torch.uint8
    prepared = pixels.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    if source_is_uint8:
        prepared.div_(255)
    return prepared


def preprocessing_metadata(config: DataConfig, image_size: int = IMAGE_SIZE) -> dict[str, Any]:
    return {
        "repository": config.repository,
        "revision": config.revision,
        "split": config.split,
        "max_samples": config.max_samples,
        "image_size": image_size,
        "augmentation": "none",
        "preprocessing": [
            "PIL.convert(RGB)",
            f"PIL.resize({image_size}x{image_size}, bicubic) when needed",
            "uint8_to_float32_[0,1]",
        ],
    }


def data_fingerprint(config: DataConfig, image_size: int = IMAGE_SIZE) -> str:
    encoded = json.dumps(
        preprocessing_metadata(config, image_size), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class ImageNetDataset(Dataset[tuple[Tensor | None, int, int]]):
    def __init__(self, dataset: Any, image_size: int, *, load_pixels: bool = True) -> None:
        self.dataset = dataset
        self.image_size = image_size
        self.load_pixels = load_pixels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor | None, int, int]:
        row: Mapping[str, Any] = self.dataset[index]
        if not self.load_pixels:
            return None, int(row["label"]), index
        image = row["image"].convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(pixels).permute(2, 0, 1), int(row["label"]), index


def load_imagenet(
    config: DataConfig,
    image_size: int = IMAGE_SIZE,
    *,
    load_pixels: bool = True,
    split: str | None = None,
) -> ImageNetDataset:
    from datasets import load_dataset

    dataset = load_dataset(
        config.repository,
        split=split or config.split,
        num_proc=12,
        revision=config.revision,
    )
    if config.max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), config.max_samples)))
    if not load_pixels:
        dataset = dataset.select_columns(["label"])
    return ImageNetDataset(dataset, image_size, load_pixels=load_pixels)


def _statistics(manifest: RepresentationManifest, stage: str) -> ChannelStatistics:
    try:
        values = manifest.statistics[stage]
        return ChannelStatistics(
            torch.tensor(values["mean"]),
            torch.tensor(values["std"]),
            int(values["count"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"representation manifest has invalid {stage} statistics") from error


class StageDataset(Dataset[tuple[Tensor, int, int]]):
    """Stage-specific target view rebuilt at every curriculum boundary."""

    def __init__(
        self,
        dataset: ImageNetDataset,
        stage: str,
        manifest: RepresentationManifest,
        root: str | Path,
    ) -> None:
        self.dataset = dataset
        self.stage = stage
        self.statistics = _statistics(manifest, stage)
        self.cache: np.memmap | None = None
        if stage != "rgb":
            cache_path = Path(root) / manifest.cache_files[stage]
            self.cache = np.memmap(
                cache_path,
                mode="r",
                dtype=np.uint16,
                shape=(len(dataset), TOKEN_COUNT, TOKEN_DIM),
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        pixels, label, source_index = self.dataset[index]
        if self.stage == "rgb":
            if pixels is None:
                raise ValueError("RGB stage dataset did not load pixels")
            target = patchify_rgb(pixels[None].float().div(255))[0]
        else:
            assert self.cache is not None
            raw = np.asarray(self.cache[source_index]).copy()
            target = torch.from_numpy(raw).view(torch.bfloat16).float()
        return self.statistics.normalize(target), label, source_index


@dataclass(frozen=True)
class RepresentationBatch:
    tokens: Tensor
    labels: Tensor
    indices: Tensor


def representation_collator(
    samples: Sequence[tuple[Tensor, int, int]],
) -> RepresentationBatch:
    tokens, labels, indices = zip(*samples, strict=True)
    return RepresentationBatch(
        torch.stack(tokens), torch.tensor(labels, dtype=torch.int64), torch.tensor(indices)
    )


def make_stage_loader(
    config: ExperimentConfig,
    stage: str,
    *,
    rank: int = 0,
    world_size: int = 1,
    epoch: int = 0,
) -> DataLoader:
    if stage not in {"dino12", "dino8", "dino4", "rgb"}:
        raise ValueError(f"unknown training stage {stage}")
    root = Path(config.representations.directory)
    manifest = RepresentationManifest.load(
        root / config.representations.manifest,
        require_complete=config.representations.require_complete,
    )
    dataset = load_imagenet(config.data, load_pixels=stage == "rgb")
    if len(dataset) != manifest.dataset_size:
        raise ValueError("dataset size differs from prepared representations")
    stage_dataset = StageDataset(dataset, stage, manifest, root)
    sampler = DistributedSampler(
        stage_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=config.data.shuffle,
        seed=config.experiment.seed,
    )
    sampler.set_epoch(epoch)
    if config.optimizer.global_batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    local_batch = config.optimizer.global_batch_size // world_size
    return DataLoader(
        stage_dataset,
        batch_size=local_batch,
        sampler=sampler,
        collate_fn=representation_collator,
        num_workers=config.data.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.data.workers > 0,
        prefetch_factor=config.data.prefetch_factor if config.data.workers > 0 else None,
        drop_last=True,
    )
