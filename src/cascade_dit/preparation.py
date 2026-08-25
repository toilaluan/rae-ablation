from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from .config import TOKEN_DIM, ExperimentConfig
from .data import data_fingerprint, load_imagenet, prepare_pixels
from .representations import (
    ChannelStatistics,
    ChannelStatisticsAccumulator,
    LayerMemmap,
    RepresentationManifest,
    fingerprint,
    patchify_rgb,
    preflight_representation_capacity,
)
from .vision import DINOMultiLayerEncoder, encoder_fingerprint, load_dino_encoder


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _all_reduce(tensor: Tensor) -> Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        target = tensor.to(torch.device("cuda") if torch.cuda.is_available() else "cpu")
        torch.distributed.all_reduce(target)
        return target.cpu()
    return tensor


def _merge_statistics(
    accumulator: ChannelStatisticsAccumulator, epsilon: float
) -> ChannelStatistics:
    count = torch.tensor(float(accumulator.count), dtype=torch.float64)
    total = accumulator.mean * accumulator.count
    square_total = accumulator.m2 + accumulator.mean.square() * accumulator.count
    packed = _all_reduce(torch.cat((count[None], total, square_total)))
    global_count = int(packed[0].item())
    if global_count < 2:
        raise ValueError("representation preparation needs at least two tokens")
    mean = packed[1 : 1 + accumulator.channels] / global_count
    variance = packed[1 + accumulator.channels :] / global_count - mean.square()
    return ChannelStatistics(
        mean.float(), variance.clamp_min(epsilon**2).sqrt().float(), global_count
    )


def _statistics_payload(statistics: ChannelStatistics) -> dict[str, object]:
    return {
        "mean": statistics.mean.tolist(),
        "std": statistics.std.tolist(),
        "count": statistics.count,
    }


def statistics_from_manifest(
    manifest: RepresentationManifest, name: str
) -> ChannelStatistics:
    try:
        values = manifest.statistics[name]
        return ChannelStatistics(
            torch.tensor(values["mean"], dtype=torch.float32),
            torch.tensor(values["std"], dtype=torch.float32),
            int(values["count"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"manifest statistics for {name} are invalid") from error


def representation_manifest_fingerprint(manifest: RepresentationManifest) -> str:
    return fingerprint(manifest.to_dict())


def _stage_payload(config: ExperimentConfig) -> list[dict[str, object]]:
    return [
        {"name": stage.name, "source": asdict(stage.source)} for stage in config.stages
    ]


def preparation_fingerprint(config: ExperimentConfig) -> str:
    return fingerprint(
        {
            "dataset": data_fingerprint(config.data, config.data.image_size),
            "encoder": encoder_fingerprint(config.encoder, config.dino_layers),
            "stages": _stage_payload(config),
            "normalization_epsilon": config.representations.normalization_epsilon,
            "cache_dtype": config.representations.dtype,
        }
    )


def _loader(dataset: object, indices: Tensor, config: ExperimentConfig) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=config.representations.cache_batch_size,
        shuffle=False,
        num_workers=config.data.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.data.workers > 0,
        prefetch_factor=config.data.prefetch_factor if config.data.workers > 0 else None,
    )


def _dino_stage_layers(config: ExperimentConfig) -> dict[str, int]:
    return {
        stage.name: int(stage.source.layer)
        for stage in config.stages
        if stage.source.kind == "dino" and stage.source.layer is not None
    }


def _extract_caches(
    dataset: object,
    encoder: DINOMultiLayerEncoder,
    caches: dict[str, LayerMemmap],
    layers: dict[str, int],
    config: ExperimentConfig,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    pending = sorted(
        set().union(
            *(set(cache.pending_indices(rank, world_size).tolist()) for cache in caches.values())
        )
    )
    if not pending:
        return
    processed = 0
    for pixels, _labels, indices in _loader(dataset, torch.tensor(pending), config):
        if rank == 0:
            processed += len(indices)
            step = processed // config.representations.cache_batch_size
            print(
                f"representation preparation: {processed}/{len(pending)} "
                f"({step})",
                flush=True,
            )
        outputs = encoder(pixels.to(device, non_blocking=True))
        for name, layer in layers.items():
            cache = caches[name]
            missing = torch.from_numpy(cache.completion[indices.numpy()] == 0)
            if missing.any():
                cache.write(indices[missing], outputs[layer][missing.to(device)])


def _collect_statistics(
    dataset: object,
    caches: dict[str, LayerMemmap],
    config: ExperimentConfig,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, ChannelStatistics]:
    accumulators = {
        stage.name: ChannelStatisticsAccumulator(TOKEN_DIM) for stage in config.stages
    }
    rgb_name = config.stages[-1].name
    indices = torch.arange(rank, len(dataset), world_size)
    i = 0
    for pixels, _labels, batch_indices in _loader(dataset, indices, config):
        i+=1
        print(i)
        if i >= 100:
            break
        for name, cache in caches.items():
            accumulators[name].update(cache.lookup(batch_indices))
        rgb = patchify_rgb(prepare_pixels(pixels, device)).cpu()
        accumulators[rgb_name].update(rgb)
    return {
        name: _merge_statistics(
            accumulator, config.representations.normalization_epsilon
        )
        for name, accumulator in accumulators.items()
    }


def _preparation_identity(
    config: ExperimentConfig, dataset_size: int
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "dataset_size": dataset_size,
        "dataset_fingerprint": data_fingerprint(config.data, config.data.image_size),
        "encoder_fingerprint": encoder_fingerprint(config.encoder, config.dino_layers),
        "stages": _stage_payload(config),
        "shape": [dataset_size, 64, TOKEN_DIM],
        "dtype": "bfloat16",
        "preparation_fingerprint": preparation_fingerprint(config),
    }


def _initialize_preparation_state(
    root: Path, config: ExperimentConfig, dataset_size: int
) -> None:
    state_path = root / "preparation.json"
    expected = _preparation_identity(config, dataset_size)
    if state_path.is_file():
        try:
            existing = json.loads(state_path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("representation preparation state is corrupt") from error
        if existing != expected:
            raise ValueError("representation preparation fingerprint mismatch")
        return
    cache_files = list(root.glob("*.bf16")) + list(root.glob("*.completion.u8"))
    if cache_files:
        raise ValueError(
            "untracked representation caches already exist; choose the fresh direct-cache "
            "directory or move them explicitly"
        )
    temporary = state_path.with_suffix(f".json.partial-{os.getpid()}")
    temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, state_path)


def prepare_representations(
    config: ExperimentConfig,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> RepresentationManifest:
    root = Path(config.representations.directory)
    manifest_path = root / config.representations.manifest
    if manifest_path.is_file():
        existing = RepresentationManifest.load(manifest_path)
        if (
            existing.dataset_fingerprint
            != data_fingerprint(config.data, config.data.image_size)
            or existing.encoder_fingerprint
            != encoder_fingerprint(config.encoder, config.dino_layers)
            or existing.preparation_fingerprint != preparation_fingerprint(config)
        ):
            raise ValueError("completed representation manifest fingerprint mismatch")
        return existing

    dataset = load_imagenet(config.data, config.data.image_size, load_pixels=True)
    layers = _dino_stage_layers(config)
    initialization_error: BaseException | None = None
    if rank == 0:
        try:
            root.mkdir(parents=True, exist_ok=True)
            _initialize_preparation_state(root, config, len(dataset))
            needed = preflight_representation_capacity(root, len(dataset), layers)
            print(
                f"representation capacity preflight: {needed / 2**30:.2f} GiB additional",
                flush=True,
            )
            for name in layers:
                LayerMemmap.initialize(root, name, len(dataset))
        except BaseException as error:
            initialization_error = error
    if world_size > 1:
        reported: list[object] = [
            repr(initialization_error)
            if rank == 0 and initialization_error is not None
            else None
        ]
        torch.distributed.broadcast_object_list(reported, src=0)
        if rank != 0 and reported[0] is not None:
            raise RuntimeError(f"representation initialization failed: {reported[0]}")
    if initialization_error is not None:
        raise initialization_error
    _barrier()

    caches = {
        name: LayerMemmap.initialize(root, name, len(dataset)) for name in layers
    }
    encoder = load_dino_encoder(
        config.encoder,
        list(layers.values()),
        device,
        compile_config=config.compile,
    )
    _extract_caches(
        dataset, encoder, caches, layers, config, rank, world_size, device
    )
    _barrier()
    for cache in caches.values():
        cache.values.flush()
        cache.completion.flush()
    statistics = _collect_statistics(
        dataset, caches, config, rank, world_size, device
    )
    if rank == 0:
        save_file(
            {
                f"statistics.{name}.{statistic}": getattr(values, statistic).float()
                for name, values in statistics.items()
                for statistic in ("mean", "std")
            },
            root / "statistics.safetensors",
        )
        manifest = RepresentationManifest(
            dataset_size=len(dataset),
            dataset_fingerprint=data_fingerprint(config.data, config.data.image_size),
            encoder_fingerprint=encoder_fingerprint(config.encoder, config.dino_layers),
            preparation_fingerprint=preparation_fingerprint(config),
            stages=_stage_payload(config),
            statistics={
                name: _statistics_payload(values) for name, values in statistics.items()
            },
            cache_files={name: f"{name}.bf16" for name in caches},
            completion_files={
                name: f"{name}.completion.u8" for name in caches
            },
            complete=True,
        )
        manifest.save(manifest_path)
    _barrier()
    return RepresentationManifest.load(manifest_path)
