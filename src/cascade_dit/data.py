from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .config import IMAGE_SIZE, TOKEN_COUNT, TOKEN_DIM, DataConfig, ExperimentConfig
from .representations import ChannelStatistics, RepresentationManifest, patchify_rgb

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
    prepared = pixels.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
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
    pixel_baseline: bool = False,
) -> MixtureChoice:
    levels = config.num_representation_levels
    if not 0 <= stage_index < levels:
        raise ValueError("stage index is outside the representation list")
    weights = np.zeros(levels, dtype=np.float32)
    if pixel_baseline or stage_index == levels - 1:
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


class MixedRepresentationDataset(Dataset[tuple[Tensor, Tensor, int, int, int]]):
    def __init__(
        self,
        dataset: ImageNetDataset,
        stage_index: int,
        epoch: int,
        config: ExperimentConfig,
        manifest: RepresentationManifest,
        root: str | Path,
        *,
        pixel_baseline: bool = False,
    ) -> None:
        self.dataset = dataset
        self.stage_index = stage_index
        self.epoch = epoch
        self.config = config
        self.manifest = manifest
        self.pixel_baseline = pixel_baseline
        self.statistics = {
            name: _statistics(manifest, name) for name in manifest.stage_names
        }
        self.caches: dict[str, np.memmap] = {}
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

    def __len__(self) -> int:
        return len(self.dataset)

    def _load_level(
        self, level: int, source_index: int, pixels: Tensor | None
    ) -> Tensor:
        stage = self.config.stages[level]
        if stage.source.kind == "rgb":
            if pixels is None:
                raise ValueError("RGB pixels were not loaded for a nonzero RGB weight")
            import time
            start = time.time()
            raw = patchify_rgb(pixels[None].float().div(255))[0]
            print(f"patchify_rgb time: {time.time() - start:.4f}s")
        else:
            values = np.asarray(self.caches[stage.name][source_index]).copy()
            raw = torch.from_numpy(values).view(torch.bfloat16).float()
        return self.statistics[stage.name].normalize(raw)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int, int, int]:
        choice = sample_representation_weights(
            self.config,
            self.stage_index,
            self.epoch,
            index,
            pixel_baseline=self.pixel_baseline,
        )
        rgb_required = bool(choice.weights[-1] > 0)
        pixels, label, source_index = self.dataset.get_item(
            index, load_pixels=rgb_required
        )
        target = torch.zeros(TOKEN_COUNT, TOKEN_DIM, dtype=torch.float32)
        for level in torch.nonzero(choice.weights, as_tuple=False).flatten().tolist():
            target.add_(
                self._load_level(level, source_index, pixels),
                alpha=float(choice.weights[level]),
            )
        return target, choice.weights, label, source_index, choice.mode


@dataclass(frozen=True)
class RepresentationBatch:
    tokens: Tensor
    representation_weights: Tensor
    labels: Tensor
    indices: Tensor
    modes: Tensor


def representation_collator(
    samples: Sequence[tuple[Tensor, Tensor, int, int, int]],
) -> RepresentationBatch:
    tokens, weights, labels, indices, modes = zip(*samples, strict=True)
    return RepresentationBatch(
        torch.stack(tokens),
        torch.stack(weights),
        torch.tensor(labels, dtype=torch.int64),
        torch.tensor(indices, dtype=torch.int64),
        torch.tensor(modes, dtype=torch.int64),
    )


def make_stage_loader(
    config: ExperimentConfig,
    stage_index: int,
    *,
    rank: int = 0,
    world_size: int = 1,
    epoch: int = 0,
    pixel_baseline: bool = False,
) -> DataLoader:
    if not 0 <= stage_index < config.num_representation_levels:
        raise ValueError("unknown training stage index")
    root = Path(config.representations.directory)
    manifest = RepresentationManifest.load(
        root / config.representations.manifest,
        require_complete=config.representations.require_complete,
    )
    expected_stages = [
        {"name": stage.name, "source": asdict(stage.source)} for stage in config.stages
    ]
    if manifest.stages != expected_stages:
        raise ValueError("prepared representation stages differ from configuration")
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
        pixel_baseline=pixel_baseline,
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
