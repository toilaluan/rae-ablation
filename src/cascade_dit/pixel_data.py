from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .config import TOKEN_COUNT, TOKEN_DIM, ExperimentConfig
from .data import (
    UnlabeledImageDataset,
    imagenet_compatible_unlabeled_fingerprint,
    load_unlabeled_images,
    prepare_pixels,
    unlabeled_data_fingerprint,
)
from .representations import ChannelStatistics, RepresentationManifest
from .vision import encoder_fingerprint

PIXEL_STATISTICS_SCHEMA_VERSION = 1


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PixelSemanticStatistics:
    dataset_fingerprint: str
    encoder_fingerprint: str
    representation_name: str
    layer: int
    statistics: ChannelStatistics
    schema_version: int = PIXEL_STATISTICS_SCHEMA_VERSION

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        tensors = target / "online-statistics.safetensors"
        metadata = target / "online-statistics.json"
        temporary_tensors = target / f".online-statistics-{os.getpid()}.safetensors"
        temporary_metadata = target / f".online-statistics-{os.getpid()}.json"
        save_file(
            {
                "mean": self.statistics.mean.float().contiguous(),
                "std": self.statistics.std.float().contiguous(),
            },
            temporary_tensors,
        )
        payload = asdict(self)
        payload["statistics"] = {"count": self.statistics.count}
        payload["tensors_sha256"] = _file_sha256(temporary_tensors)
        temporary_metadata.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary_tensors, tensors)
        os.replace(temporary_metadata, metadata)

    @classmethod
    def load(cls, directory: str | Path) -> PixelSemanticStatistics:
        root = Path(directory)
        metadata_path = root / "online-statistics.json"
        tensors_path = root / "online-statistics.safetensors"
        if not metadata_path.is_file() or not tensors_path.is_file():
            raise ValueError("online semantic statistics are missing")
        payload = json.loads(metadata_path.read_text())
        if payload.get("schema_version") != PIXEL_STATISTICS_SCHEMA_VERSION:
            raise ValueError("online semantic statistics schema is incompatible")
        if payload.get("tensors_sha256") != _file_sha256(tensors_path):
            raise ValueError("online semantic statistics checksum mismatch")
        tensors = load_file(tensors_path)
        try:
            mean = tensors["mean"].float()
            std = tensors["std"].float()
            count = int(payload["statistics"]["count"])
            result = cls(
                dataset_fingerprint=str(payload["dataset_fingerprint"]),
                encoder_fingerprint=str(payload["encoder_fingerprint"]),
                representation_name=str(payload["representation_name"]),
                layer=int(payload["layer"]),
                statistics=ChannelStatistics(mean, std, count),
                schema_version=int(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("online semantic statistics are corrupt") from error
        result.validate()
        return result

    def validate(self) -> None:
        values = self.statistics
        if (
            not self.dataset_fingerprint
            or not self.encoder_fingerprint
            or not self.representation_name
            or not 1 <= self.layer <= 12
            or values.mean.shape != (TOKEN_DIM,)
            or values.std.shape != (TOKEN_DIM,)
            or values.count < 2
            or not torch.isfinite(values.mean).all()
            or not torch.isfinite(values.std).all()
            or torch.any(values.std <= 0)
        ):
            raise ValueError("online semantic statistics are invalid")


class PixelPretrainingBatch(NamedTuple):
    pixels: Tensor
    semantic: Tensor | None


class CachedPixelDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        dataset: UnlabeledImageDataset,
        manifest: RepresentationManifest,
        root: str | Path,
        representation_name: str,
    ) -> None:
        self.dataset = dataset
        try:
            cache_path = Path(root) / manifest.cache_files[representation_name]
        except KeyError as error:
            raise ValueError("pixel pretraining semantic cache is missing") from error
        self.semantic = np.memmap(
            cache_path,
            mode="r",
            dtype=np.uint16,
            shape=(len(dataset), TOKEN_COUNT, TOKEN_DIM),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        pixels, source_index = self.dataset.get_item(index)
        raw = np.asarray(self.semantic[source_index]).copy()
        return torch.from_numpy(raw).view(torch.bfloat16), pixels


def _cached_collator(samples: list[tuple[Tensor, Tensor]]) -> PixelPretrainingBatch:
    semantics, pixels = zip(*samples, strict=True)
    return PixelPretrainingBatch(torch.stack(pixels), torch.stack(semantics))


def _online_collator(samples: list[tuple[Tensor, int]]) -> PixelPretrainingBatch:
    pixels, _indices = zip(*samples, strict=True)
    return PixelPretrainingBatch(torch.stack(pixels), None)


def _manifest_statistics(
    manifest: RepresentationManifest, name: str
) -> ChannelStatistics:
    try:
        values = manifest.statistics[name]
        result = ChannelStatistics(
            torch.tensor(values["mean"], dtype=torch.float32),
            torch.tensor(values["std"], dtype=torch.float32),
            int(values["count"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("pixel semantic cache statistics are corrupt") from error
    if result.mean.shape != (TOKEN_DIM,) or result.std.shape != (TOKEN_DIM,):
        raise ValueError("pixel semantic cache statistics have the wrong shape")
    return result


def _split_settings(
    config: ExperimentConfig, split_kind: str
) -> tuple[str, int | None, str]:
    values = config.pixel_pretraining
    if split_kind == "train":
        return (
            values.data.train_split,
            values.data.max_train_samples,
            values.train_cache_directory,
        )
    if split_kind == "validation":
        return (
            values.data.validation_split,
            values.data.max_validation_samples,
            values.validation_cache_directory,
        )
    raise ValueError("pixel pretraining split must be train or validation")


def validate_pixel_manifest(
    config: ExperimentConfig,
    manifest: RepresentationManifest,
    *,
    split_kind: str,
) -> None:
    split, maximum, _directory = _split_settings(config, split_kind)
    semantic = config.representations.semantic
    stages = {
        str(stage.get("name")): stage.get("source")
        for stage in manifest.stages
        if isinstance(stage, dict)
    }
    source = stages.get(semantic.name)
    if (
        not isinstance(source, dict)
        or source.get("kind") != "dino"
        or source.get("layer") != semantic.layer
    ):
        raise ValueError("pixel cache does not contain the selected DINO representation")
    expected = unlabeled_data_fingerprint(
        config.pixel_pretraining.data, split=split, max_samples=maximum
    )
    legacy = imagenet_compatible_unlabeled_fingerprint(
        config.pixel_pretraining.data, split=split, max_samples=maximum
    )
    if manifest.dataset_fingerprint not in {expected, legacy}:
        raise ValueError("pixel cache was prepared from a different image dataset")
    expected_encoder = encoder_fingerprint(config.encoder, [semantic.layer])
    if manifest.encoder_fingerprint != expected_encoder:
        raise ValueError("pixel cache was prepared with a different DINO encoder")


def load_pixel_manifest(
    config: ExperimentConfig, *, split_kind: str
) -> RepresentationManifest:
    _split, _maximum, directory = _split_settings(config, split_kind)
    values = config.pixel_pretraining
    manifest = RepresentationManifest.load(
        Path(directory) / values.manifest,
        required_stages=[config.representations.semantic.name],
    )
    validate_pixel_manifest(config, manifest, split_kind=split_kind)
    return manifest


def load_pixel_semantic_statistics(config: ExperimentConfig) -> ChannelStatistics:
    values = config.pixel_pretraining
    if values.semantic_input_mode == "cached":
        manifest = load_pixel_manifest(config, split_kind="train")
        return _manifest_statistics(manifest, config.representations.semantic.name)
    artifact = PixelSemanticStatistics.load(values.train_cache_directory)
    split = values.data.train_split
    maximum = values.data.max_train_samples
    expected_data = unlabeled_data_fingerprint(
        values.data, split=split, max_samples=maximum
    )
    expected_encoder = encoder_fingerprint(
        config.encoder, [config.representations.semantic.layer]
    )
    if (
        artifact.dataset_fingerprint != expected_data
        or artifact.encoder_fingerprint != expected_encoder
        or artifact.representation_name != config.representations.semantic.name
        or artifact.layer != config.representations.semantic.layer
    ):
        raise ValueError("online semantic statistics provenance mismatch")
    return artifact.statistics


def make_pixel_pretraining_loader(
    config: ExperimentConfig,
    *,
    split_kind: str,
    rank: int = 0,
    world_size: int = 1,
    epoch: int = 0,
) -> DataLoader:
    split, maximum, directory = _split_settings(config, split_kind)
    images = load_unlabeled_images(
        config.pixel_pretraining.data, split=split, max_samples=maximum
    )
    cached = config.pixel_pretraining.semantic_input_mode == "cached"
    if cached:
        manifest = load_pixel_manifest(config, split_kind=split_kind)
        if len(images) != manifest.dataset_size:
            raise ValueError("pixel image dataset and semantic cache sizes differ")
        dataset: Dataset = CachedPixelDataset(
            images,
            manifest,
            directory,
            config.representations.semantic.name,
        )
        collator = _cached_collator
    else:
        dataset = images
        collator = _online_collator
    training = split_kind == "train"
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=training and config.pixel_pretraining.data.shuffle,
        seed=config.experiment.seed,
        drop_last=training,
    )
    sampler.set_epoch(epoch)
    optimizer = config.pixel_pretraining.optimizer
    if optimizer.global_batch_size % world_size:
        raise ValueError("pixel pretraining batch size must be divisible by world size")
    workers = config.pixel_pretraining.data.workers
    return DataLoader(
        dataset,
        batch_size=optimizer.global_batch_size // world_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=(
            config.pixel_pretraining.data.prefetch_factor if workers > 0 else None
        ),
        drop_last=training,
    )


def set_pixel_loader_epoch(loader: DataLoader, epoch: int) -> None:
    if epoch < 0:
        raise ValueError("pixel pretraining epoch must be non-negative")
    sampler = loader.sampler
    if not isinstance(sampler, DistributedSampler):
        raise TypeError("pixel pretraining loader has an unexpected sampler")
    sampler.set_epoch(epoch)


class PixelBatchPreprocessor:
    def __init__(
        self,
        statistics: ChannelStatistics,
        device: torch.device | str,
        encoder: torch.nn.Module | None = None,
        layer: int = 12,
    ) -> None:
        self.statistics = statistics
        self.device = torch.device(device)
        self.encoder = encoder
        self.layer = layer

    @torch.no_grad()
    def __call__(self, batch: PixelPretrainingBatch) -> tuple[Tensor, Tensor]:
        pixels = prepare_pixels(batch.pixels, self.device, non_blocking=True)
        semantic = batch.semantic
        if semantic is None:
            if self.encoder is None:
                raise ValueError("online pixel pretraining requires a DINO encoder")
            outputs = self.encoder(batch.pixels.to(self.device, non_blocking=True))
            semantic = outputs[self.layer]
        semantic = self.statistics.normalize(
            semantic.to(self.device, non_blocking=True).float()
        )
        return semantic, pixels
