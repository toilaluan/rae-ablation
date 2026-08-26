from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .config import IMAGE_SIZE, TOKEN_COUNT, TOKEN_DIM, DataConfig, ExperimentConfig
from .representations import (
    ChannelStatistics,
    RepresentationManifest,
    cached_stages_include,
    patchify_rgb,
)

PURE_CURRENT = 0
ADJACENT_MIX = 1
PURE_NEXT = 2
ALL_LEVELS_MIX = 3
MIXTURE_MODE_NAMES = ("pure_current", "adjacent_mix", "pure_next", "all_levels")


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


class ImageNetDataset(Dataset[tuple[Tensor | None, int, int]]):
    def __init__(
        self,
        dataset: Any,
        labels: Any,
        image_size: int,
        *,
        load_pixels: bool = True,
    ) -> None:
        self.dataset = dataset
        self.labels = labels
        self.image_size = image_size
        self.load_pixels = load_pixels

    def __len__(self) -> int:
        return len(self.labels)

    def get_item(
        self, index: int, *, load_pixels: bool
    ) -> tuple[Tensor | None, int, int]:
        label_row: Mapping[str, Any] = self.labels[index]
        if not load_pixels:
            return None, int(label_row["label"]), index
        if self.dataset is None:
            raise ValueError("ImageNet pixels were not retained by this dataset")
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

    def __getitem__(self, index: int) -> tuple[Tensor | None, int, int]:
        return self.get_item(index, load_pixels=self.load_pixels)


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
    labels = dataset.select_columns(["label"])
    return ImageNetDataset(
        dataset if load_pixels else None,
        labels,
        image_size,
        load_pixels=load_pixels,
    )


def _statistics(
    manifest: RepresentationManifest, stage: str
) -> ChannelStatistics:
    try:
        values = manifest.statistics[stage]
        return ChannelStatistics(
            torch.tensor(values["mean"]),
            torch.tensor(values["std"]),
            int(values["count"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"representation manifest has invalid {stage} statistics") from error


def _item_seed(seed: int, epoch: int, source_index: int) -> int:
    payload = f"{seed}:{epoch}:{source_index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


@dataclass(frozen=True)
class MixtureChoice:
    weights: Tensor
    mode: int


def sample_representation_weights(
    config: ExperimentConfig,
    stage_index: int,
    epoch: int,
    source_index: int,
    *,
    pixel_only: bool = False,
    representation_levels: int | None = None,
) -> MixtureChoice:
    levels = representation_levels or config.num_representation_levels
    if not 0 <= stage_index < levels:
        raise ValueError("stage index is outside the representation list")
    weights = np.zeros(levels, dtype=np.float32)
    if pixel_only or stage_index == levels - 1:
        weights[-1] = 1
        return MixtureChoice(torch.from_numpy(weights), PURE_CURRENT)

    mixing = config.curriculum.mixing
    rng = np.random.default_rng(_item_seed(config.experiment.seed, epoch, source_index))
    choice = float(rng.random())
    current_end = mixing.pure_current_probability
    adjacent_end = current_end + mixing.adjacent.probability
    next_end = adjacent_end + mixing.pure_next_probability
    if choice < current_end:
        weights[stage_index] = 1
        mode = PURE_CURRENT
    elif choice < adjacent_end:
        adjacent = mixing.adjacent
        next_weight = float(rng.uniform(adjacent.minimum, adjacent.maximum))
        weights[stage_index] = 1 - next_weight
        weights[stage_index + 1] = next_weight
        mode = ADJACENT_MIX
    elif choice < next_end:
        weights[stage_index + 1] = 1
        mode = PURE_NEXT
    else:
        alpha = np.full(levels, mixing.all_levels.concentration, dtype=np.float64)
        weights = rng.dirichlet(alpha).astype(np.float32)
        mode = ALL_LEVELS_MIX
    return MixtureChoice(torch.from_numpy(weights), mode)


RawRepresentationSample = tuple[
    tuple[Tensor | None, ...],
    Tensor,
    int,
    int,
    int,
]


class MixedRepresentationDataset(Dataset[RawRepresentationSample]):
    def __init__(
        self,
        dataset: ImageNetDataset,
        stage_index: int,
        epoch: int,
        config: ExperimentConfig,
        manifest: RepresentationManifest,
        root: str | Path,
        *,
        pixel_only: bool = False,
        representation_levels: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.manifest = manifest
        self.pixel_only = pixel_only
        self.representation_levels = representation_levels or config.num_representation_levels
        self._stage_epoch = torch.tensor(0, dtype=torch.int64).share_memory_()
        self.set_epoch(stage_index, epoch)
        self.caches: dict[str, np.memmap] = {}
        if pixel_only:
            return
        for stage in config.stages:
            if stage.source.kind != "dino":
                continue
            cache_path = Path(root) / manifest.cache_files[stage.name]
            self.caches[stage.name] = np.memmap(
                cache_path,
                mode="r",
                dtype=np.uint16,
                shape=(len(dataset), TOKEN_COUNT, TOKEN_DIM),
            )

    @property
    def stage_index(self) -> int:
        return int(self._stage_epoch.item()) % self.representation_levels

    @property
    def epoch(self) -> int:
        return int(self._stage_epoch.item()) // self.representation_levels

    def set_epoch(self, stage_index: int, epoch: int) -> None:
        """Update curriculum state visible to persistent DataLoader workers."""
        if not 0 <= stage_index < self.representation_levels:
            raise ValueError("unknown training stage index")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._stage_epoch.fill_(epoch * self.representation_levels + stage_index)

    def __len__(self) -> int:
        return len(self.dataset)

    def _load_level(
        self, level: int, source_index: int, pixels: Tensor | None
    ) -> Tensor:
        if self.pixel_only and level == self.representation_levels - 1:
            if pixels is None:
                raise ValueError("RGB pixels were not loaded for a nonzero RGB weight")
            return pixels
        stage = self.config.stages[level]
        if stage.source.kind == "rgb":
            if pixels is None:
                raise ValueError("RGB pixels were not loaded for a nonzero RGB weight")
            return pixels
        values = np.asarray(self.caches[stage.name][source_index]).copy()
        return torch.from_numpy(values).view(torch.bfloat16)

    def __getitem__(self, index: int) -> RawRepresentationSample:
        state = int(self._stage_epoch.item())
        stage_index = state % self.representation_levels
        epoch = state // self.representation_levels
        choice = sample_representation_weights(
            self.config,
            stage_index,
            epoch,
            index,
            pixel_only=self.pixel_only,
            representation_levels=self.representation_levels,
        )
        rgb_required = bool(choice.weights[-1] > 0)
        pixels, label, source_index = self.dataset.get_item(
            index, load_pixels=rgb_required
        )
        levels: list[Tensor | None] = [None] * self.representation_levels
        for level in torch.nonzero(choice.weights, as_tuple=False).flatten().tolist():
            levels[level] = self._load_level(level, source_index, pixels)
        return tuple(levels), choice.weights, label, source_index, choice.mode


class RawLevelBatch(NamedTuple):
    batch_indices: Tensor
    values: Tensor


class RepresentationBatch(NamedTuple):
    levels: tuple[RawLevelBatch | None, ...]
    representation_weights: Tensor
    labels: Tensor
    indices: Tensor
    modes: Tensor


class PreparedRepresentationBatch(NamedTuple):
    tokens: Tensor
    representation_weights: Tensor
    labels: Tensor
    indices: Tensor
    modes: Tensor


def representation_collator(
    samples: Sequence[RawRepresentationSample],
) -> RepresentationBatch:
    raw_levels, weights, labels, indices, modes = zip(*samples, strict=True)
    level_count = len(raw_levels[0])
    if any(len(levels) != level_count for levels in raw_levels):
        raise ValueError("representation samples have inconsistent level counts")
    levels: list[RawLevelBatch | None] = []
    for level_index in range(level_count):
        selected = [
            (batch_index, sample_levels[level_index])
            for batch_index, sample_levels in enumerate(raw_levels)
            if sample_levels[level_index] is not None
        ]
        if not selected:
            levels.append(None)
            continue
        levels.append(
            RawLevelBatch(
                torch.tensor(
                    [batch_index for batch_index, _values in selected],
                    dtype=torch.int64,
                ),
                torch.stack([values for _batch_index, values in selected]),
            )
        )
    return RepresentationBatch(
        tuple(levels),
        torch.stack(weights),
        torch.tensor(labels, dtype=torch.int64),
        torch.tensor(indices, dtype=torch.int64),
        torch.tensor(modes, dtype=torch.int64),
    )


class RepresentationBatchPreprocessor:
    """Convert, normalize, and mix sparse raw representations on one device."""

    def __init__(
        self,
        config: ExperimentConfig,
        manifest: RepresentationManifest,
        device: torch.device | str,
        *,
        pixel_only: bool = False,
        representation_levels: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        levels = representation_levels or config.num_representation_levels
        self.source_kinds = (
            tuple(["dino"] * (levels - 1) + ["rgb"])
            if pixel_only
            else tuple(stage.source.kind for stage in config.stages)
        )
        if pixel_only:
            pixel = _statistics(manifest, config.pixel_training.name)
            statistics = [
                ChannelStatistics(torch.zeros(TOKEN_DIM), torch.ones(TOKEN_DIM), pixel.count)
                for _level in range(levels - 1)
            ] + [pixel]
        else:
            statistics = [_statistics(manifest, stage.name) for stage in config.stages]
        self.means = torch.stack([values.mean for values in statistics]).to(self.device)
        self.stds = torch.stack([values.std for values in statistics]).to(self.device)

    @torch.no_grad()
    def __call__(self, batch: RepresentationBatch) -> PreparedRepresentationBatch:
        if len(batch.levels) != len(self.source_kinds):
            raise ValueError("representation batch level count differs from configuration")
        weights = batch.representation_weights.to(self.device, non_blocking=True)
        labels = batch.labels.to(self.device, non_blocking=True)
        indices = batch.indices.to(self.device, non_blocking=True)
        modes = batch.modes.to(self.device, non_blocking=True)
        tokens = torch.zeros(
            weights.shape[0],
            TOKEN_COUNT,
            TOKEN_DIM,
            dtype=torch.float32,
            device=self.device,
        )
        for level_index, level in enumerate(batch.levels):
            if level is None:
                continue
            batch_indices = level.batch_indices.to(self.device, non_blocking=True)
            raw = level.values.to(self.device, non_blocking=True)
            if self.source_kinds[level_index] == "rgb":
                values = patchify_rgb(prepare_pixels(raw, self.device))
            else:
                values = raw.float()
            level_weights = weights[batch_indices, level_index, None, None]
            values.sub_(self.means[level_index]).div_(self.stds[level_index])
            values.mul_(level_weights)
            tokens.index_add_(0, batch_indices, values)
        return PreparedRepresentationBatch(tokens, weights, labels, indices, modes)


def make_stage_loader(
    config: ExperimentConfig,
    stage_index: int,
    *,
    rank: int = 0,
    world_size: int = 1,
    epoch: int = 0,
    pixel_only: bool = False,
    representation_levels: int | None = None,
) -> DataLoader:
    levels = representation_levels or config.num_representation_levels
    if not 0 <= stage_index < levels:
        raise ValueError("unknown training stage index")
    root = Path(config.representations.directory)
    manifest = RepresentationManifest.load(
        root / config.representations.manifest,
        require_complete=config.representations.require_complete,
        required_stages=[config.pixel_training.name] if pixel_only else None,
    )
    requested = [config.pixel_stage] if pixel_only else config.stages
    expected_stages = [
        {"name": stage.name, "source": asdict(stage.source)} for stage in requested
    ]
    if not cached_stages_include(manifest.stages, expected_stages):
        raise ValueError("prepared representations do not include configured stages")
    if manifest.dataset_fingerprint != data_fingerprint(config.data, config.data.image_size):
        raise ValueError("prepared representations use a different dataset")
    dataset = load_imagenet(config.data, load_pixels=True)
    if len(dataset) != manifest.dataset_size:
        raise ValueError("dataset size differs from prepared representations")
    mixed_dataset = MixedRepresentationDataset(
        dataset,
        stage_index,
        epoch,
        config,
        manifest,
        root,
        pixel_only=pixel_only,
        representation_levels=levels,
    )
    sampler = DistributedSampler(
        mixed_dataset,
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
        mixed_dataset,
        batch_size=local_batch,
        sampler=sampler,
        collate_fn=representation_collator,
        num_workers=config.data.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.data.workers > 0,
        prefetch_factor=config.data.prefetch_factor if config.data.workers > 0 else None,
        drop_last=True,
    )


def set_stage_loader_epoch(
    loader: DataLoader,
    stage_index: int,
    epoch: int,
) -> None:
    """Advance a reusable stage loader without recreating its dataset or workers."""
    dataset = loader.dataset
    sampler = loader.sampler
    if not isinstance(dataset, MixedRepresentationDataset):
        raise TypeError("stage loader has an unexpected dataset")
    if not isinstance(sampler, DistributedSampler):
        raise TypeError("stage loader has an unexpected sampler")
    dataset.set_epoch(stage_index, epoch)
    sampler.set_epoch(epoch)
