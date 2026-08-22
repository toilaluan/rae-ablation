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

from .config import DINO_CHANNELS, DINO_GRID_SIZE, DINO_PATCH_SIZE, IMAGE_SIZE, DataConfig


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


def patchify_rgb(pixels: Tensor) -> Tensor:
    """Convert 128px RGB images into the 768x8x8 flow endpoint layout."""
    if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"RGB patchification expects [B,3,{IMAGE_SIZE},{IMAGE_SIZE}], "
            f"received {tuple(pixels.shape)}"
        )
    batch = pixels.shape[0]
    patch = DINO_PATCH_SIZE
    grid = DINO_GRID_SIZE
    return (
        pixels.reshape(batch, 3, grid, patch, grid, patch)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch, grid, grid, DINO_CHANNELS)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def unpatchify_rgb(patches: Tensor) -> Tensor:
    """Invert :func:`patchify_rgb` without changing values."""
    expected = (DINO_CHANNELS, DINO_GRID_SIZE, DINO_GRID_SIZE)
    if patches.ndim != 4 or tuple(patches.shape[1:]) != expected:
        raise ValueError(
            f"RGB unpatchification expects [B,{expected[0]},{expected[1]},{expected[2]}], "
            f"received {tuple(patches.shape)}"
        )
    batch = patches.shape[0]
    patch = DINO_PATCH_SIZE
    grid = DINO_GRID_SIZE
    return (
        patches.permute(0, 2, 3, 1)
        .reshape(batch, grid, grid, patch, patch, 3)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, 3, IMAGE_SIZE, IMAGE_SIZE)
        .contiguous()
    )


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
            "raw_block2_patch_states_before_final_layernorm",
            "final_layernorm_block12_semantic_patch_tokens",
            "separate_target-semantic-rgb-patch_normalization",
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
