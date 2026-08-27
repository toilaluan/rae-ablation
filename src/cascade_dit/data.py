from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .config import IMAGE_SIZE, TOKEN_COUNT, TOKEN_DIM, DataConfig, ExperimentConfig
from .representations import ChannelStatistics, RepresentationManifest


def prepare_pixels(
    pixels: Tensor,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> Tensor:
    source_is_uint8 = pixels.dtype == torch.uint8
    prepared = pixels.to(device=device, non_blocking=non_blocking).float()
    if source_is_uint8:
        prepared.div_(255)
    return prepared


def preprocessing_metadata(
    config: DataConfig, image_size: int = IMAGE_SIZE
) -> dict[str, Any]:
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
        preprocessing_metadata(config, image_size),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class ImageNetDataset(Dataset[tuple[Tensor, int, int]]):
    def __init__(self, dataset: Any, labels: Any, image_size: int) -> None:
        self.dataset = dataset
        self.labels = labels
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.labels)

    def get_item(self, index: int) -> tuple[Tensor, int, int]:
        label_row: Mapping[str, Any] = self.labels[index]
        row: Mapping[str, Any] = self.dataset[index]
        image = row["image"].convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
        pixels = np.asarray(image, dtype=np.uint8).copy()
        return (
            torch.from_numpy(pixels).permute(2, 0, 1),
            int(label_row["label"]),
            index,
        )

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        return self.get_item(index)


def load_imagenet(
    config: DataConfig,
    image_size: int = IMAGE_SIZE,
    *,
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
    labels = dataset.select_columns(["label"])
    return ImageNetDataset(dataset, labels, image_size)


def _statistics(manifest: RepresentationManifest, name: str) -> ChannelStatistics:
    try:
        values = manifest.statistics[name]
        return ChannelStatistics(
            torch.tensor(values["mean"]),
            torch.tensor(values["std"]),
            int(values["count"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"representation manifest has invalid {name} statistics") from error


JointSample = tuple[Tensor, Tensor, int]


class JointDataset(Dataset[JointSample]):
    """Always load cached semantic tokens, raw RGB, and a class label."""

    def __init__(
        self,
        dataset: ImageNetDataset,
        config: ExperimentConfig,
        manifest: RepresentationManifest,
        root: str | Path,
    ) -> None:
        self.dataset = dataset
        name = config.representations.semantic.name
        try:
            cache_path = Path(root) / manifest.cache_files[name]
        except KeyError as error:
            raise ValueError("selected semantic cache is missing from the manifest") from error
        self.semantic_cache = np.memmap(
            cache_path,
            mode="r",
            dtype=np.uint16,
            shape=(len(dataset), TOKEN_COUNT, TOKEN_DIM),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> JointSample:
        pixels, label, source_index = self.dataset.get_item(index)
        raw = np.asarray(self.semantic_cache[source_index]).copy()
        semantic = torch.from_numpy(raw).view(torch.bfloat16)
        return semantic, pixels, label


class JointBatch(NamedTuple):
    semantic: Tensor
    pixels: Tensor
    labels: Tensor


class PreparedJointBatch(NamedTuple):
    semantic: Tensor
    pixels: Tensor
    labels: Tensor


def joint_collator(samples: Sequence[JointSample]) -> JointBatch:
    semantic, pixels, labels = zip(*samples, strict=True)
    return JointBatch(
        torch.stack(semantic),
        torch.stack(pixels),
        torch.tensor(labels, dtype=torch.int64),
    )


class JointBatchPreprocessor:
    """Normalize DINO targets while leaving RGB in its raw [0,1] range."""

    def __init__(
        self,
        config: ExperimentConfig,
        manifest: RepresentationManifest,
        device: torch.device | str,
    ) -> None:
        self.device = torch.device(device)
        self.semantic_statistics = _statistics(
            manifest, config.representations.semantic.name
        )

    @torch.no_grad()
    def __call__(self, batch: JointBatch) -> PreparedJointBatch:
        semantic = self.semantic_statistics.normalize(
            batch.semantic.to(self.device, non_blocking=True).float()
        )
        pixels = prepare_pixels(batch.pixels, self.device, non_blocking=True)
        labels = batch.labels.to(self.device, non_blocking=True)
        return PreparedJointBatch(semantic, pixels, labels)


def validate_joint_manifest(
    config: ExperimentConfig, manifest: RepresentationManifest
) -> None:
    stages = {
        str(stage.get("name")): stage.get("source")
        for stage in manifest.stages
        if isinstance(stage, dict)
    }
    selected = config.representations.semantic
    source = stages.get(selected.name)
    if (
        not isinstance(source, dict)
        or source.get("kind") != "dino"
        or source.get("layer") != selected.layer
    ):
        raise ValueError("representation manifest does not include the selected semantic target")
    if manifest.dataset_fingerprint != data_fingerprint(
        config.data, config.data.image_size
    ):
        raise ValueError("prepared representations use a different dataset")


def load_joint_manifest(config: ExperimentConfig) -> RepresentationManifest:
    root = Path(config.representations.directory)
    manifest = RepresentationManifest.load(
        root / config.representations.manifest,
        require_complete=config.representations.require_complete,
        required_stages=[config.representations.semantic.name],
    )
    validate_joint_manifest(config, manifest)
    return manifest


def make_joint_loader(
    config: ExperimentConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    epoch: int = 0,
) -> DataLoader:
    manifest = load_joint_manifest(config)
    dataset = load_imagenet(config.data)
    if len(dataset) != manifest.dataset_size:
        raise ValueError("dataset size differs from prepared representations")
    joint_dataset = JointDataset(
        dataset,
        config,
        manifest,
        config.representations.directory,
    )
    sampler = DistributedSampler(
        joint_dataset,
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
        joint_dataset,
        batch_size=local_batch,
        sampler=sampler,
        collate_fn=joint_collator,
        num_workers=config.data.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.data.workers > 0,
        prefetch_factor=config.data.prefetch_factor if config.data.workers > 0 else None,
        drop_last=True,
    )


def set_joint_loader_epoch(loader: DataLoader, epoch: int) -> None:
    sampler = loader.sampler
    if not isinstance(loader.dataset, JointDataset):
        raise TypeError("joint loader has an unexpected dataset")
    if not isinstance(sampler, DistributedSampler):
        raise TypeError("joint loader has an unexpected sampler")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    sampler.set_epoch(epoch)
