from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .config import DataConfig


def preprocessing_metadata(config: DataConfig, image_size: int) -> dict[str, Any]:
    return {
        "repository": config.repository,
        "revision": config.revision,
        "split": config.split,
        "max_samples": config.max_samples,
        "image_size": image_size,
        "preprocessing": [
            "PIL.convert(RGB)",
            f"PIL.resize({image_size}x{image_size}, bicubic) when needed",
            "uint8_to_float32_[0,1]",
            "device_random_horizontal_flip",
            "model_scale_[0,1]_to_[-1,1]",
        ],
    }


def data_fingerprint(config: DataConfig, image_size: int) -> str:
    encoded = json.dumps(
        preprocessing_metadata(config, image_size), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class ImageNetDataset(Dataset[tuple[Tensor, int, int]]):
    """Small tensor adapter over the pinned Hugging Face ImageNet-256 dataset."""

    def __init__(self, dataset: Any, image_size: int) -> None:
        self.dataset = dataset
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        row: Mapping[str, Any] = self.dataset[index]
        image = row["image"].convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
        pixels = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        return tensor, int(row["label"]), index


def load_imagenet(config: DataConfig, image_size: int) -> ImageNetDataset:
    from datasets import load_dataset

    dataset = load_dataset(
        config.repository,
        split=config.split,
        num_proc=12,
        revision=config.revision,
    )
    if config.max_samples is not None:
        count = min(len(dataset), config.max_samples)
        dataset = dataset.select(range(count))
    return ImageNetDataset(dataset, image_size)


def random_horizontal_flip(
    images: Tensor,
    generator: torch.Generator,
    probability: float = 0.5,
) -> Tensor:
    """Flip complete examples on their current device using a resumable RNG stream."""
    if images.ndim != 4:
        raise ValueError("images must have shape [B, C, H, W]")
    if not 0 <= probability <= 1:
        raise ValueError("flip probability must be in [0, 1]")
    if probability == 0:
        return images
    if probability == 1:
        return images.flip(-1)
    mask = torch.rand(
        images.shape[0], device=images.device, generator=generator
    ) < probability
    return torch.where(mask[:, None, None, None], images.flip(-1), images)
