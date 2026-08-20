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


def prepare_pixels(
    pixels: Tensor,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> Tensor:
    """Move image tensors to float32, scaling uint8 pixels into [0, 1]."""
    source_is_uint8 = pixels.dtype == torch.uint8
    prepared = pixels.to(
        device=device,
        dtype=torch.float32,
        non_blocking=non_blocking,
    )
    if source_is_uint8:
        prepared.div_(255)
    return prepared


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
            "online_frozen_dinov3_b16_imagenet_normalization",
            "shared_final_layernorm_target_and_semantic_patch_tokens",
            "shared_dataset_per_channel_latent_normalization",
        ],
    }


def data_fingerprint(config: DataConfig, image_size: int) -> str:
    encoded = json.dumps(
        preprocessing_metadata(config, image_size), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class ImageNetDataset(Dataset[tuple[Tensor | None, int, int]]):
    """Small tensor adapter over the pinned Hugging Face ImageNet-256 dataset."""

    def __init__(
        self, dataset: Any, image_size: int, *, load_pixels: bool = True
    ) -> None:
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
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
        pixels = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        return tensor, int(row["label"]), index


def load_imagenet(
    config: DataConfig,
    image_size: int,
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
        count = min(len(dataset), config.max_samples)
        dataset = dataset.select(range(count))
    if not load_pixels:
        dataset = dataset.select_columns(["label"])
    return ImageNetDataset(dataset, image_size, load_pixels=load_pixels)
