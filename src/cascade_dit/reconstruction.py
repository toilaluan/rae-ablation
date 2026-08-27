from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from .config import TOKEN_COUNT, TOKEN_DIM, DataConfig, ExperimentConfig
from .data import ImageNetDataset, data_fingerprint, load_imagenet, prepare_pixels
from .representations import ChannelStatistics
from .vision import encoder_fingerprint

RECONSTRUCTION_CACHE_SCHEMA_VERSION = 1


def reconstruction_data_config(config: ExperimentConfig) -> DataConfig:
    return replace(
        config.data,
        split=config.validation.reconstruction.split,
        max_samples=None,
        shuffle=False,
    )


def select_class_balanced_indices(
    dataset: ImageNetDataset,
    *,
    num_classes: int,
    samples_per_class: int,
) -> tuple[list[int], list[int]]:
    selected: list[list[int]] = [[] for _ in range(num_classes)]
    remaining = num_classes * samples_per_class
    for index in range(len(dataset)):
        row = dataset.labels[index]
        try:
            label = int(row["label"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("validation dataset contains an invalid class label") from error
        if not 0 <= label < num_classes:
            raise ValueError("validation dataset contains an out-of-range class label")
        if len(selected[label]) < samples_per_class:
            selected[label].append(index)
            remaining -= 1
            if remaining == 0:
                break
    missing = [index for index, values in enumerate(selected) if len(values) < samples_per_class]
    if missing:
        raise ValueError(
            "validation dataset cannot provide the configured samples per class; "
            f"missing classes: {missing[:16]}"
        )
    source_indices = [index for values in selected for index in values]
    labels = [label for label in range(num_classes) for _ in range(samples_per_class)]
    return source_indices, labels


def _selection_digest(source_indices: list[int], labels: list[int]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(source_indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(labels, dtype=np.int64).tobytes())
    return digest.hexdigest()


def reconstruction_preparation_fingerprint(
    config: ExperimentConfig,
    source_indices: list[int],
    labels: list[int],
) -> str:
    semantic = config.representations.semantic
    data = reconstruction_data_config(config)
    payload = {
        "dataset": data_fingerprint(data, data.image_size),
        "encoder": encoder_fingerprint(config.encoder, [semantic.layer]),
        "representation": {"name": semantic.name, "layer": semantic.layer},
        "selection": {
            "kind": "first_per_class",
            "samples_per_class": config.validation.reconstruction.samples_per_class,
            "sha256": _selection_digest(source_indices, labels),
        },
        "cache_dtype": "bfloat16",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ReconstructionCacheManifest:
    dataset_size: int
    dataset_fingerprint: str
    encoder_fingerprint: str
    preparation_fingerprint: str
    representation_name: str
    layer: int
    source_indices: list[int]
    labels: list[int]
    cache_file: str
    completion_file: str
    complete: bool = False
    schema_version: int = RECONSTRUCTION_CACHE_SCHEMA_VERSION

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + f".partial-{os.getpid()}")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        require_complete: bool = True,
    ) -> ReconstructionCacheManifest:
        target = Path(path)
        try:
            payload = json.loads(target.read_text())
            if payload.get("schema_version") != RECONSTRUCTION_CACHE_SCHEMA_VERSION:
                raise ValueError("reconstruction cache manifest schema is incompatible")
            manifest = cls(**payload)
        except (OSError, json.JSONDecodeError, TypeError) as error:
            raise ValueError(
                f"reconstruction cache manifest is missing or corrupt: {target}"
            ) from error
        manifest.validate(target, require_complete=require_complete)
        return manifest

    def validate(self, path: Path, *, require_complete: bool) -> None:
        if require_complete and not self.complete:
            raise ValueError("reconstruction cache is incomplete")
        if (
            self.dataset_size < 1
            or self.dataset_size != len(self.source_indices)
            or self.dataset_size != len(self.labels)
            or len(set(self.source_indices)) != self.dataset_size
            or any(index < 0 for index in self.source_indices)
            or any(label < 0 for label in self.labels)
            or not all(
                (
                    self.dataset_fingerprint,
                    self.encoder_fingerprint,
                    self.preparation_fingerprint,
                    self.representation_name,
                    self.cache_file,
                    self.completion_file,
                )
            )
            or not 1 <= self.layer <= 12
        ):
            raise ValueError("reconstruction cache manifest is invalid")
        root = path.parent
        cache = root / self.cache_file
        completion_path = root / self.completion_file
        expected = self.dataset_size * TOKEN_COUNT * TOKEN_DIM * 2
        if not cache.is_file() or cache.stat().st_size != expected:
            raise ValueError("reconstruction semantic cache is missing or has the wrong size")
        if (
            not completion_path.is_file()
            or completion_path.stat().st_size != self.dataset_size
        ):
            raise ValueError("reconstruction completion map is missing or has the wrong size")
        completion = np.memmap(
            completion_path, mode="r", dtype=np.uint8, shape=(self.dataset_size,)
        )
        if require_complete and np.any(completion != 1):
            raise ValueError("reconstruction semantic cache has unfinished rows")


def _manifest_path(config: ExperimentConfig) -> Path:
    values = config.validation.reconstruction
    return Path(values.cache_directory) / values.manifest


def validate_reconstruction_manifest(
    config: ExperimentConfig,
    manifest: ReconstructionCacheManifest,
) -> None:
    values = config.validation.reconstruction
    semantic = config.representations.semantic
    data = reconstruction_data_config(config)
    expected_labels = [
        label
        for label in range(config.model.num_classes)
        for _ in range(values.samples_per_class)
    ]
    if (
        manifest.dataset_size != config.model.num_classes * values.samples_per_class
        or manifest.dataset_fingerprint != data_fingerprint(data, data.image_size)
        or manifest.encoder_fingerprint
        != encoder_fingerprint(config.encoder, [semantic.layer])
        or manifest.representation_name != semantic.name
        or manifest.layer != semantic.layer
        or manifest.labels != expected_labels
        or manifest.preparation_fingerprint
        != reconstruction_preparation_fingerprint(
            config, manifest.source_indices, manifest.labels
        )
    ):
        raise ValueError("reconstruction cache provenance does not match the experiment")


def load_reconstruction_manifest(
    config: ExperimentConfig,
    *,
    require_complete: bool = True,
) -> ReconstructionCacheManifest:
    manifest = ReconstructionCacheManifest.load(
        _manifest_path(config), require_complete=require_complete
    )
    validate_reconstruction_manifest(config, manifest)
    return manifest


class ReconstructionSample(NamedTuple):
    semantic: Tensor
    pixels: Tensor
    label: int
    position: int


class ReconstructionBatch(NamedTuple):
    semantic: Tensor
    pixels: Tensor
    labels: Tensor
    positions: Tensor


class PreparedReconstructionBatch(NamedTuple):
    semantic: Tensor
    pixels: Tensor
    labels: Tensor
    positions: Tensor


class ReconstructionDataset(Dataset[ReconstructionSample]):
    def __init__(
        self,
        dataset: ImageNetDataset,
        manifest: ReconstructionCacheManifest,
        root: str | Path,
    ) -> None:
        self.dataset = dataset
        self.manifest = manifest
        self.semantic = np.memmap(
            Path(root) / manifest.cache_file,
            mode="r",
            dtype=np.uint16,
            shape=(manifest.dataset_size, TOKEN_COUNT, TOKEN_DIM),
        )

    def __len__(self) -> int:
        return self.manifest.dataset_size

    def __getitem__(self, position: int) -> ReconstructionSample:
        source_index = self.manifest.source_indices[position]
        pixels, label, _source_index = self.dataset.get_item(source_index)
        if label != self.manifest.labels[position]:
            raise ValueError("reconstruction dataset labels changed after preparation")
        raw = np.asarray(self.semantic[position]).copy()
        return ReconstructionSample(
            torch.from_numpy(raw).view(torch.bfloat16), pixels, label, position
        )


class ReconstructionPreparationDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        dataset: ImageNetDataset,
        source_indices: list[int],
        labels: list[int],
    ) -> None:
        self.dataset = dataset
        self.source_indices = source_indices
        self.labels = labels

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, position: int) -> tuple[Tensor, Tensor]:
        pixels, label, _source_index = self.dataset.get_item(
            self.source_indices[position]
        )
        if label != self.labels[position]:
            raise ValueError("reconstruction dataset labels changed during preparation")
        return pixels, torch.tensor(position, dtype=torch.int64)


def reconstruction_collator(samples: list[ReconstructionSample]) -> ReconstructionBatch:
    semantic, pixels, labels, positions = zip(*samples, strict=True)
    return ReconstructionBatch(
        torch.stack(semantic),
        torch.stack(pixels),
        torch.tensor(labels, dtype=torch.int64),
        torch.tensor(positions, dtype=torch.int64),
    )


class ReconstructionBatchPreprocessor:
    def __init__(
        self,
        statistics: ChannelStatistics,
        device: torch.device | str,
    ) -> None:
        self.statistics = statistics
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, batch: ReconstructionBatch) -> PreparedReconstructionBatch:
        semantic = self.statistics.normalize(
            batch.semantic.to(self.device, non_blocking=True).float()
        )
        return PreparedReconstructionBatch(
            semantic,
            prepare_pixels(batch.pixels, self.device, non_blocking=True),
            batch.labels.to(self.device, non_blocking=True),
            batch.positions,
        )


def make_reconstruction_loader(
    config: ExperimentConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    manifest = load_reconstruction_manifest(config)
    data = reconstruction_data_config(config)
    images = load_imagenet(data, data.image_size)
    if manifest.source_indices and max(manifest.source_indices) >= len(images):
        raise ValueError("reconstruction source index exceeds the validation dataset")
    dataset = ReconstructionDataset(
        images,
        manifest,
        config.validation.reconstruction.cache_directory,
    )
    local = Subset(dataset, range(rank, len(dataset), world_size))
    return DataLoader(
        local,
        batch_size=config.validation.reconstruction.batch_size,
        shuffle=False,
        collate_fn=reconstruction_collator,
        num_workers=config.data.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.data.workers > 0,
        prefetch_factor=config.data.prefetch_factor if config.data.workers > 0 else None,
        drop_last=False,
    )
