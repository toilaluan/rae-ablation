from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from .config import TOKEN_DIM, ExperimentConfig
from .data import load_unlabeled_images, unlabeled_data_fingerprint
from .pixel_data import (
    PixelSemanticStatistics,
    load_pixel_manifest,
    load_pixel_semantic_statistics,
)
from .representations import (
    ChannelStatistics,
    ChannelStatisticsAccumulator,
    LayerMemmap,
    RepresentationManifest,
    fingerprint,
    preflight_representation_capacity,
)
from .vision import DINOMultiLayerEncoder, encoder_fingerprint, load_dino_encoder


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _all_reduce(values: Tensor) -> Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        target = values.to(torch.device("cuda") if torch.cuda.is_available() else "cpu")
        torch.distributed.all_reduce(target)
        return target.cpu()
    return values


def _merge_statistics(
    accumulator: ChannelStatisticsAccumulator, epsilon: float
) -> ChannelStatistics:
    count = torch.tensor(float(accumulator.count), dtype=torch.float64)
    total = accumulator.mean * accumulator.count
    square_total = accumulator.m2 + accumulator.mean.square() * accumulator.count
    packed = _all_reduce(torch.cat((count[None], total, square_total)))
    global_count = int(packed[0].item())
    if global_count < 2:
        raise ValueError("pixel semantic statistics require at least two tokens")
    mean = packed[1 : 1 + accumulator.channels] / global_count
    variance = packed[1 + accumulator.channels :] / global_count - mean.square()
    return ChannelStatistics(
        mean.float(), variance.clamp_min(epsilon**2).sqrt().float(), global_count
    )


def _split_settings(
    config: ExperimentConfig, split_kind: str
) -> tuple[str, int | None, Path]:
    values = config.pixel_pretraining
    if split_kind == "train":
        return (
            values.data.train_split,
            values.data.max_train_samples,
            Path(values.train_cache_directory),
        )
    if split_kind == "validation":
        return (
            values.data.validation_split,
            values.data.max_validation_samples,
            Path(values.validation_cache_directory),
        )
    raise ValueError("pixel pretraining split must be train or validation")


def _dataset_loader(
    dataset: object,
    indices: Tensor,
    config: ExperimentConfig,
) -> DataLoader:
    workers = config.pixel_pretraining.data.workers
    return DataLoader(
        Subset(dataset, indices.tolist()),  # type: ignore[arg-type]
        batch_size=config.pixel_pretraining.cache_batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=(
            config.pixel_pretraining.data.prefetch_factor if workers > 0 else None
        ),
    )


def _extract_cache(
    dataset: object,
    encoder: DINOMultiLayerEncoder,
    cache: LayerMemmap,
    layer: int,
    config: ExperimentConfig,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    pending = torch.from_numpy(cache.pending_indices(rank, world_size))
    processed = 0
    for pixels, indices in _dataset_loader(dataset, pending, config):
        if rank == 0:
            processed += len(indices)
            print(
                f"pixel semantic cache: {processed}/{len(pending)}",
                flush=True,
            )
        outputs = encoder(pixels.to(device, non_blocking=True))[layer]
        cache.write(indices, outputs)


def _cached_statistics(
    dataset: object,
    cache: LayerMemmap,
    config: ExperimentConfig,
    rank: int,
    world_size: int,
) -> ChannelStatistics:
    accumulator = ChannelStatisticsAccumulator(TOKEN_DIM)
    indices = torch.arange(rank, len(dataset), world_size)  # type: ignore[arg-type]
    for _pixels, batch_indices in _dataset_loader(dataset, indices, config):
        accumulator.update(cache.lookup(batch_indices))
    return _merge_statistics(
        accumulator, config.pixel_pretraining.normalization_epsilon
    )


def _preparation_fingerprint(
    config: ExperimentConfig,
    *,
    split: str,
    maximum: int | None,
) -> str:
    semantic = config.representations.semantic
    return fingerprint(
        {
            "dataset": unlabeled_data_fingerprint(
                config.pixel_pretraining.data,
                split=split,
                max_samples=maximum,
            ),
            "encoder": encoder_fingerprint(config.encoder, [semantic.layer]),
            "stage": {
                "name": semantic.name,
                "source": {"kind": "dino", "layer": semantic.layer},
            },
            "normalization_epsilon": config.pixel_pretraining.normalization_epsilon,
            "cache_dtype": "bfloat16",
        }
    )


def prepare_cached_pixel_split(
    config: ExperimentConfig,
    *,
    split_kind: str,
    rank: int,
    world_size: int,
    device: torch.device,
) -> RepresentationManifest:
    split, maximum, root = _split_settings(config, split_kind)
    manifest_path = root / config.pixel_pretraining.manifest
    if manifest_path.is_file():
        return load_pixel_manifest(config, split_kind=split_kind)
    dataset = load_unlabeled_images(
        config.pixel_pretraining.data,
        split=split,
        max_samples=maximum,
    )
    semantic = config.representations.semantic
    error: list[str | None] = [None]
    if rank == 0:
        try:
            root.mkdir(parents=True, exist_ok=True)
            needed = preflight_representation_capacity(
                root, len(dataset), [semantic.name]
            )
            print(
                f"pixel cache capacity preflight ({split_kind}): "
                f"{needed / 2**30:.2f} GiB additional",
                flush=True,
            )
            LayerMemmap.initialize(root, semantic.name, len(dataset))
        except Exception as exception:
            error[0] = f"{type(exception).__name__}: {exception}"
    if world_size > 1:
        torch.distributed.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0])
    _barrier()
    cache = LayerMemmap.initialize(root, semantic.name, len(dataset))
    encoder = load_dino_encoder(
        config.encoder,
        [semantic.layer],
        device,
        compile_config=config.compile,
    )
    _extract_cache(
        dataset,
        encoder,
        cache,
        semantic.layer,
        config,
        rank,
        world_size,
        device,
    )
    del encoder
    _barrier()
    statistics = _cached_statistics(dataset, cache, config, rank, world_size)
    dataset_hash = unlabeled_data_fingerprint(
        config.pixel_pretraining.data,
        split=split,
        max_samples=maximum,
    )
    if rank == 0:
        save_file(
            {
                f"statistics.{semantic.name}.mean": statistics.mean,
                f"statistics.{semantic.name}.std": statistics.std,
            },
            root / "statistics.safetensors",
        )
        manifest = RepresentationManifest(
            dataset_size=len(dataset),
            dataset_fingerprint=dataset_hash,
            encoder_fingerprint=encoder_fingerprint(
                config.encoder, [semantic.layer]
            ),
            preparation_fingerprint=_preparation_fingerprint(
                config, split=split, maximum=maximum
            ),
            stages=[
                {
                    "name": semantic.name,
                    "source": {"kind": "dino", "layer": semantic.layer},
                }
            ],
            statistics={
                semantic.name: {
                    "mean": statistics.mean.tolist(),
                    "std": statistics.std.tolist(),
                    "count": statistics.count,
                }
            },
            cache_files={semantic.name: f"{semantic.name}.bf16"},
            completion_files={
                semantic.name: f"{semantic.name}.completion.u8"
            },
            complete=True,
        )
        manifest.save(manifest_path)
    _barrier()
    return load_pixel_manifest(config, split_kind=split_kind)


def prepare_online_pixel_statistics(
    config: ExperimentConfig,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> PixelSemanticStatistics:
    try:
        existing = load_pixel_semantic_statistics(config)
    except ValueError:
        existing = None
    if existing is not None:
        return PixelSemanticStatistics(
            dataset_fingerprint=unlabeled_data_fingerprint(
                config.pixel_pretraining.data,
                split=config.pixel_pretraining.data.train_split,
                max_samples=config.pixel_pretraining.data.max_train_samples,
            ),
            encoder_fingerprint=encoder_fingerprint(
                config.encoder, [config.representations.semantic.layer]
            ),
            representation_name=config.representations.semantic.name,
            layer=config.representations.semantic.layer,
            statistics=existing,
        )
    data = config.pixel_pretraining.data
    dataset = load_unlabeled_images(
        data,
        split=data.train_split,
        max_samples=data.max_train_samples,
    )
    semantic = config.representations.semantic
    encoder = load_dino_encoder(
        config.encoder,
        [semantic.layer],
        device,
        compile_config=config.compile,
    )
    accumulator = ChannelStatisticsAccumulator(TOKEN_DIM)
    indices = torch.arange(rank, len(dataset), world_size)
    for pixels, _batch_indices in _dataset_loader(dataset, indices, config):
        accumulator.update(
            encoder(pixels.to(device, non_blocking=True))[semantic.layer]
        )
    statistics = _merge_statistics(
        accumulator, config.pixel_pretraining.normalization_epsilon
    )
    artifact = PixelSemanticStatistics(
        dataset_fingerprint=unlabeled_data_fingerprint(
            data,
            split=data.train_split,
            max_samples=data.max_train_samples,
        ),
        encoder_fingerprint=encoder_fingerprint(config.encoder, [semantic.layer]),
        representation_name=semantic.name,
        layer=semantic.layer,
        statistics=statistics,
    )
    if rank == 0:
        artifact.save(config.pixel_pretraining.train_cache_directory)
    _barrier()
    return PixelSemanticStatistics.load(
        config.pixel_pretraining.train_cache_directory
    )


def prepare_pixel_pretraining_inputs(
    config: ExperimentConfig,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    if config.pixel_pretraining.semantic_input_mode == "online":
        prepare_online_pixel_statistics(
            config, rank=rank, world_size=world_size, device=device
        )
        return
    for split_kind in ("train", "validation"):
        prepare_cached_pixel_split(
            config,
            split_kind=split_kind,
            rank=rank,
            world_size=world_size,
            device=device,
        )
