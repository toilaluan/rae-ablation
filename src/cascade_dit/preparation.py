from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from .config import PIXEL_REPRESENTATION_NAME, TOKEN_DIM, ExperimentConfig
from .data import (
    data_fingerprint,
    load_imagenet,
    prepare_pixels,
    validate_endpoint_manifest,
)
from .representations import (
    ChannelStatistics,
    ChannelStatisticsAccumulator,
    LayerMemmap,
    RepresentationManifest,
    cached_stages_include,
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
    semantic = config.representations.semantic
    pixel = {
        "name": PIXEL_REPRESENTATION_NAME,
        "source": {"kind": "rgb", "layer": None},
    }
    if config.training.mode == "pixel_only":
        return [pixel]
    return [
        {
            "name": semantic.name,
            "source": {"kind": "dino", "layer": semantic.layer},
        },
        pixel,
    ]


def _dino_layers_from_stages(stages: list[dict[str, object]]) -> list[int]:
    return [
        int(source["layer"])
        for stage in stages
        if isinstance((source := stage.get("source")), dict)
        and source.get("kind") == "dino"
        and source.get("layer") is not None
    ]


def preparation_fingerprint(
    config: ExperimentConfig, stages: list[dict[str, object]] | None = None
) -> str:
    selected_stages = _stage_payload(config) if stages is None else stages
    return fingerprint(
        {
            "dataset": data_fingerprint(config.data, config.data.image_size),
            "encoder": encoder_fingerprint(
                config.encoder, _dino_layers_from_stages(selected_stages)
            ),
            "stages": selected_stages,
            "normalization_epsilon": config.representations.normalization_epsilon,
            "cache_dtype": "bfloat16",
        }
    )


def validate_manifest_compatibility(
    config: ExperimentConfig, manifest: RepresentationManifest
) -> None:
    validate_endpoint_manifest(config, manifest, load_semantic=True)
    requested_stages = _stage_payload(config)
    cached_stages = manifest.stages
    if not cached_stages_include(cached_stages, requested_stages):
        raise ValueError(
            "completed representation manifest does not include the configured stages"
        )
    cached_layers = _dino_layers_from_stages(cached_stages)
    if (
        manifest.dataset_fingerprint
        != data_fingerprint(config.data, config.data.image_size)
        or manifest.encoder_fingerprint
        != encoder_fingerprint(config.encoder, cached_layers)
        or manifest.preparation_fingerprint
        != preparation_fingerprint(config, cached_stages)
    ):
        raise ValueError("completed representation manifest fingerprint mismatch")


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
    if config.training.mode == "pixel_only":
        return {}
    semantic = config.representations.semantic
    return {semantic.name: semantic.layer}


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
        **{name: ChannelStatisticsAccumulator(TOKEN_DIM) for name in caches},
        PIXEL_REPRESENTATION_NAME: ChannelStatisticsAccumulator(TOKEN_DIM),
    }
    indices = torch.arange(rank, len(dataset), world_size)
    for pixels, _labels, batch_indices in _loader(dataset, indices, config):
        for name, cache in caches.items():
            accumulators[name].update(cache.lookup(batch_indices))
        rgb = patchify_rgb(prepare_pixels(pixels, device)).cpu()
        accumulators[PIXEL_REPRESENTATION_NAME].update(rgb)
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
        "schema_version": 3,
        "dataset_size": dataset_size,
        "dataset_fingerprint": data_fingerprint(config.data, config.data.image_size),
        "encoder_fingerprint": encoder_fingerprint(
            config.encoder, list(_dino_stage_layers(config).values())
        ),
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
        if not isinstance(existing, dict):
            raise ValueError("representation preparation state is corrupt")
        cached_stages = existing.get("stages")
        compatible = (
            isinstance(cached_stages, list)
            and cached_stages_include(cached_stages, expected["stages"])
            and existing.get("schema_version") == expected["schema_version"]
            and existing.get("dataset_size") == expected["dataset_size"]
            and existing.get("dataset_fingerprint") == expected["dataset_fingerprint"]
            and existing.get("shape") == expected["shape"]
            and existing.get("dtype") == expected["dtype"]
            and existing.get("encoder_fingerprint")
            == encoder_fingerprint(config.encoder, _dino_layers_from_stages(cached_stages))
            and existing.get("preparation_fingerprint")
            == preparation_fingerprint(config, cached_stages)
        )
        if not compatible:
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
        existing = RepresentationManifest.load(
            manifest_path,
            required_stages=(
                None
                if config.training.mode == "progressive"
                else [PIXEL_REPRESENTATION_NAME]
            ),
        )
        if config.training.mode == "progressive":
            validate_manifest_compatibility(config, existing)
        else:
            validate_endpoint_manifest(config, existing, load_semantic=False)
        return existing

    dataset = load_imagenet(config.data, config.data.image_size)
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
    if layers:
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
            encoder_fingerprint=encoder_fingerprint(
                config.encoder, list(layers.values())
            ),
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
