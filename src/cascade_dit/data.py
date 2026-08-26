from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .config import (
    IMAGE_SIZE,
    PIXEL_REPRESENTATION_NAME,
    TOKEN_COUNT,
    TOKEN_DIM,
    DataConfig,
    ExperimentConfig,
)
from .representations import ChannelStatistics, RepresentationManifest, patchify_rgb

EndpointPhase = Literal["semantic", "pixel"]
PHASES: tuple[EndpointPhase, ...] = ("semantic", "pixel")


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
    def __init__(
        self,
        dataset: Any,
        labels: Any,
        image_size: int,
    ) -> None:
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


def _statistics(
    manifest: RepresentationManifest, name: str
) -> ChannelStatistics:
    try:
        values = manifest.statistics[name]
        return ChannelStatistics(
            torch.tensor(values["mean"]),
            torch.tensor(values["std"]),
            int(values["count"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"representation manifest has invalid {name} statistics") from error


def _item_seed(seed: int, epoch: int, source_index: int) -> int:
    payload = f"{seed}:{epoch}:{source_index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def sample_pixel_ratio(
    config: ExperimentConfig,
    epoch: int,
    source_index: int,
    *,
    phase: EndpointPhase = "semantic",
) -> float:
    """Deterministically choose a pure-semantic or logit-normal mixed item."""
    if epoch < 0 or source_index < 0:
        raise ValueError("epoch and source index must be non-negative")
    if phase == "pixel":
        return 1.0
    if phase != "semantic":
        raise ValueError("phase must be semantic or pixel")
    settings = config.training.semantic
    rng = np.random.default_rng(_item_seed(config.experiment.seed, epoch, source_index))
    if float(rng.random()) >= settings.mixed_probability:
        return 0.0
    ratio = settings.mix_ratio
    logit = float(rng.normal(ratio.loc, ratio.scale))
    if logit >= 0:
        value = 1 / (1 + math.exp(-logit))
    else:
        exponential = math.exp(logit)
        value = exponential / (1 + exponential)
    epsilon = float(torch.finfo(torch.float32).eps)
    return min(max(value, epsilon), 1 - epsilon)


EndpointSample = tuple[Tensor | None, Tensor, Tensor, int]


class EndpointDataset(Dataset[EndpointSample]):
    """Load exactly the semantic and RGB endpoints required by the active phase."""

    def __init__(
        self,
        dataset: ImageNetDataset,
        phase: EndpointPhase,
        epoch: int,
        config: ExperimentConfig,
        manifest: RepresentationManifest,
        root: str | Path,
        *,
        load_semantic: bool,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.manifest = manifest
        self.load_semantic = load_semantic
        self._phase_epoch = torch.tensor(0, dtype=torch.int64).share_memory_()
        self.set_epoch(phase, epoch)
        self.semantic_cache: np.memmap | None = None
        if load_semantic:
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

    @property
    def phase(self) -> EndpointPhase:
        return PHASES[int(self._phase_epoch.item()) % len(PHASES)]

    @property
    def epoch(self) -> int:
        return int(self._phase_epoch.item()) // len(PHASES)

    def set_epoch(self, phase: EndpointPhase, epoch: int) -> None:
        if phase not in PHASES:
            raise ValueError("phase must be semantic or pixel")
        if phase == "semantic" and not self.load_semantic:
            raise ValueError("semantic phase requires a semantic cache")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._phase_epoch.fill_(epoch * len(PHASES) + PHASES.index(phase))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> EndpointSample:
        state = int(self._phase_epoch.item())
        phase = PHASES[state % len(PHASES)]
        epoch = state // len(PHASES)
        pixels, label, source_index = self.dataset.get_item(index)
        if phase == "pixel":
            return None, pixels, torch.tensor(1.0), label
        if self.semantic_cache is None:
            raise ValueError("semantic phase requires a semantic cache")
        raw = np.asarray(self.semantic_cache[source_index]).copy()
        semantic = torch.from_numpy(raw).view(torch.bfloat16)
        ratio = sample_pixel_ratio(self.config, epoch, source_index, phase=phase)
        return semantic, pixels, torch.tensor(ratio), label


class EndpointBatch(NamedTuple):
    semantic: Tensor | None
    pixels: Tensor
    pixel_ratio: Tensor
    labels: Tensor


class PreparedEndpointBatch(NamedTuple):
    clean: Tensor
    semantic: Tensor | None
    pixel: Tensor
    pixel_ratio: Tensor
    labels: Tensor


def endpoint_collator(samples: Sequence[EndpointSample]) -> EndpointBatch:
    semantic, pixels, ratios, labels = zip(*samples, strict=True)
    available = [values is not None for values in semantic]
    if any(available) and not all(available):
        raise ValueError("endpoint batch mixes semantic and pixel phases")
    semantic_batch = (
        torch.stack([values for values in semantic if values is not None])
        if all(available)
        else None
    )
    return EndpointBatch(
        semantic_batch,
        torch.stack(pixels),
        torch.stack(ratios).float(),
        torch.tensor(labels, dtype=torch.int64),
    )


class EndpointBatchPreprocessor:
    """Normalize semantic and RGB endpoints independently, then interpolate."""

    def __init__(
        self,
        config: ExperimentConfig,
        manifest: RepresentationManifest,
        device: torch.device | str,
        *,
        load_semantic: bool,
    ) -> None:
        self.device = torch.device(device)
        self.pixel_statistics = _statistics(manifest, PIXEL_REPRESENTATION_NAME)
        self.semantic_statistics = (
            _statistics(manifest, config.representations.semantic.name)
            if load_semantic
            else None
        )

    @torch.no_grad()
    def __call__(self, batch: EndpointBatch) -> PreparedEndpointBatch:
        ratios = batch.pixel_ratio.to(self.device, non_blocking=True)
        labels = batch.labels.to(self.device, non_blocking=True)
        pixels = patchify_rgb(
            prepare_pixels(batch.pixels, self.device, non_blocking=True)
        )
        pixel = self.pixel_statistics.normalize(pixels)
        semantic = None
        if batch.semantic is not None:
            if self.semantic_statistics is None:
                raise ValueError("semantic values were provided without semantic statistics")
            semantic = self.semantic_statistics.normalize(
                batch.semantic.to(self.device, non_blocking=True).float()
            )
            clean = torch.lerp(semantic, pixel, ratios[:, None, None].to(pixel))
        else:
            if torch.any(ratios != 1):
                raise ValueError("a pixel-only batch must have pixel_ratio=1")
            clean = pixel
        return PreparedEndpointBatch(clean, semantic, pixel, ratios, labels)


def _required_manifest_stages(config: ExperimentConfig, load_semantic: bool) -> list[str]:
    names = [PIXEL_REPRESENTATION_NAME]
    if load_semantic:
        names.insert(0, config.representations.semantic.name)
    return names


def validate_endpoint_manifest(
    config: ExperimentConfig,
    manifest: RepresentationManifest,
    *,
    load_semantic: bool,
) -> None:
    stages = {
        str(stage.get("name")): stage.get("source")
        for stage in manifest.stages
        if isinstance(stage, dict)
    }
    pixel_source = stages.get(PIXEL_REPRESENTATION_NAME)
    if not isinstance(pixel_source, dict) or pixel_source.get("kind") != "rgb":
        raise ValueError("representation manifest does not include RGB statistics")
    if load_semantic:
        selected = config.representations.semantic
        source = stages.get(selected.name)
        if (
            not isinstance(source, dict)
            or source.get("kind") != "dino"
            or source.get("layer") != selected.layer
        ):
            raise ValueError("representation manifest does not include the selected semantic endpoint")
    if manifest.dataset_fingerprint != data_fingerprint(
        config.data, config.data.image_size
    ):
        raise ValueError("prepared representations use a different dataset")


def load_endpoint_manifest(
    config: ExperimentConfig, *, load_semantic: bool
) -> RepresentationManifest:
    root = Path(config.representations.directory)
    manifest = RepresentationManifest.load(
        root / config.representations.manifest,
        require_complete=config.representations.require_complete,
        required_stages=_required_manifest_stages(config, load_semantic),
    )
    validate_endpoint_manifest(config, manifest, load_semantic=load_semantic)
    return manifest


def make_endpoint_loader(
    config: ExperimentConfig,
    phase: EndpointPhase,
    *,
    rank: int = 0,
    world_size: int = 1,
    epoch: int = 0,
) -> DataLoader:
    load_semantic = config.training.mode == "progressive"
    if phase == "semantic" and not load_semantic:
        raise ValueError("pixel_only training cannot enter the semantic phase")
    manifest = load_endpoint_manifest(config, load_semantic=load_semantic)
    dataset = load_imagenet(config.data)
    if len(dataset) != manifest.dataset_size:
        raise ValueError("dataset size differs from prepared representations")
    endpoint_dataset = EndpointDataset(
        dataset,
        phase,
        epoch,
        config,
        manifest,
        config.representations.directory,
        load_semantic=load_semantic,
    )
    sampler = DistributedSampler(
        endpoint_dataset,
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
        endpoint_dataset,
        batch_size=local_batch,
        sampler=sampler,
        collate_fn=endpoint_collator,
        num_workers=config.data.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.data.workers > 0,
        prefetch_factor=config.data.prefetch_factor if config.data.workers > 0 else None,
        drop_last=True,
    )


def set_endpoint_loader_epoch(
    loader: DataLoader,
    phase: EndpointPhase,
    epoch: int,
) -> None:
    """Advance a reusable loader while preserving its persistent workers."""
    dataset = loader.dataset
    sampler = loader.sampler
    if not isinstance(dataset, EndpointDataset):
        raise TypeError("endpoint loader has an unexpected dataset")
    if not isinstance(sampler, DistributedSampler):
        raise TypeError("endpoint loader has an unexpected sampler")
    dataset.set_epoch(phase, epoch)
    sampler.set_epoch(epoch)
