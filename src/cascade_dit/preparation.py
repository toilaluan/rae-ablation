from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from .alignment import (
    AlignmentDiagnostics,
    StreamingProcrustes,
    cumulative_projectors,
    orthogonality_error,
)
from .config import TOKEN_DIM, ExperimentConfig
from .data import data_fingerprint, load_imagenet, prepare_pixels
from .representations import (
    ChannelStatistics,
    ChannelStatisticsAccumulator,
    LayerMemmap,
    RepresentationManifest,
    deterministic_subset_indices,
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


def _merge_statistics(accumulator: ChannelStatisticsAccumulator, epsilon: float) -> ChannelStatistics:
    count = torch.tensor(float(accumulator.count), dtype=torch.float64)
    total = accumulator.mean * accumulator.count
    square_total = accumulator.m2 + accumulator.mean.square() * accumulator.count
    packed = torch.cat((count[None], total, square_total))
    packed = _all_reduce(packed)
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


def preparation_fingerprint(config: ExperimentConfig) -> str:
    return fingerprint(
        {
            "dataset": data_fingerprint(config.data, config.data.image_size),
            "encoder": encoder_fingerprint(config.encoder),
            "normalization_epsilon": config.representations.normalization_epsilon,
            "cache_dtype": config.representations.dtype,
            "alignment": asdict(config.alignment),
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


def _extract_caches(
    dataset: object,
    encoder: DINOMultiLayerEncoder,
    caches: dict[str, LayerMemmap],
    config: ExperimentConfig,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    pending = sorted(
        set().union(*(set(cache.pending_indices(rank, world_size).tolist()) for cache in caches.values()))
    )
    if not pending:
        return
    loader = _loader(dataset, torch.tensor(pending), config)
    for pixels, _labels, indices in loader:
        outputs = encoder(pixels.to(device, non_blocking=True))
        for layer in config.encoder.layers:
            name = f"dino{layer}"
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
        name: ChannelStatisticsAccumulator(TOKEN_DIM)
        for name in ("dino12", "dino8", "dino4", "rgb")
    }
    indices = torch.arange(rank, len(dataset), world_size)
    for pixels, _labels, batch_indices in _loader(dataset, indices, config):
        for name, cache in caches.items():
            accumulators[name].update(cache.lookup(batch_indices))
        rgb = patchify_rgb(prepare_pixels(pixels, device)).cpu()
        accumulators["rgb"].update(rgb)
    return {
        name: _merge_statistics(accumulator, config.representations.normalization_epsilon)
        for name, accumulator in accumulators.items()
    }


def _fit_alignments(
    dataset: object,
    caches: dict[str, LayerMemmap],
    statistics: dict[str, ChannelStatistics],
    config: ExperimentConfig,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, AlignmentDiagnostics]]:
    total_needed = min(
        len(dataset), config.alignment.subset_size + config.alignment.heldout_size
    )
    chosen = deterministic_subset_indices(len(dataset), total_needed, config.alignment.subset_seed)
    train_count = min(config.alignment.subset_size, total_needed - 1)
    train_indices, heldout_indices = chosen[:train_count], chosen[train_count:]
    if heldout_indices.numel() == 0:
        heldout_indices = chosen[-1:]
    pairs = list(config.alignment.pairs)
    if config.alignment.direct_layer12_to_rgb and "dino12->rgb" not in pairs:
        pairs.append("dino12->rgb")
    accumulators = {pair: StreamingProcrustes(TOKEN_DIM) for pair in pairs}

    def normalized_batch(pixels: Tensor, indices: Tensor) -> dict[str, Tensor]:
        values = {
            name: statistics[name].normalize(cache.lookup(indices).float())
            for name, cache in caches.items()
        }
        values["rgb"] = statistics["rgb"].normalize(
            patchify_rgb(prepare_pixels(pixels, device)).cpu()
        )
        return values

    local_train = train_indices[rank::world_size]
    for pixels, _labels, indices in _loader(dataset, local_train, config):
        values = normalized_batch(pixels, indices)
        for pair, accumulator in accumulators.items():
            source, target = pair.split("->")
            accumulator.update(values[source], values[target])
    for accumulator in accumulators.values():
        accumulator.cross_covariance = _all_reduce(accumulator.cross_covariance)
        count = _all_reduce(torch.tensor([accumulator.count], dtype=torch.int64))
        accumulator.count = int(count.item())
    matrices = {
        pair: accumulator.solve(
            identity=config.alignment.identity,
            driver=config.alignment.svd_driver,
            device=device,
        )
        for pair, accumulator in accumulators.items()
    }

    heldout_sums = {pair: torch.zeros(2, dtype=torch.float64) for pair in matrices}
    local_heldout = heldout_indices[rank::world_size]
    for pixels, _labels, indices in _loader(dataset, local_heldout, config):
        values = normalized_batch(pixels, indices)
        for pair, matrix in matrices.items():
            source, target = pair.split("->")
            error = (values[source].float() @ matrix - values[target].float()).square()
            heldout_sums[pair] += torch.tensor([error.sum().item(), error.numel()])
    diagnostics = {}
    for pair, values in heldout_sums.items():
        values = _all_reduce(values)
        diagnostics[pair] = AlignmentDiagnostics(
            heldout_mse=float(values[0] / values[1]),
            orthogonality_error=orthogonality_error(matrices[pair]),
            count=accumulators[pair].count,
        )
    return matrices, diagnostics


def _preparation_identity(config: ExperimentConfig, dataset_size: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_size": dataset_size,
        "dataset_fingerprint": data_fingerprint(config.data, config.data.image_size),
        "encoder_fingerprint": encoder_fingerprint(config.encoder),
        "layers": config.encoder.layers,
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
    cache_files = list(root.glob("dino*.bf16")) + list(root.glob("dino*.completion.u8"))
    if cache_files:
        raise ValueError("untracked representation caches already exist; refusing to reuse them")
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
        try:
            existing = RepresentationManifest.load(manifest_path)
            expected_data = data_fingerprint(config.data, config.data.image_size)
            expected_encoder = encoder_fingerprint(config.encoder)
            if (
                existing.dataset_fingerprint != expected_data
                or existing.encoder_fingerprint != expected_encoder
                or existing.preparation_fingerprint != preparation_fingerprint(config)
            ):
                raise ValueError("completed representation manifest fingerprint mismatch")
            return existing
        except ValueError:
            # A corrupt or incompatible final manifest is never silently overwritten.
            raise
    dataset = load_imagenet(config.data, config.data.image_size, load_pixels=True)
    initialization_error: BaseException | None = None
    if rank == 0:
        try:
            root.mkdir(parents=True, exist_ok=True)
            _initialize_preparation_state(root, config, len(dataset))
            needed = preflight_representation_capacity(root, len(dataset))
            print(
                f"representation capacity preflight: {needed / 2**30:.2f} GiB additional",
                flush=True,
            )
            for name in ("dino12", "dino8", "dino4"):
                LayerMemmap.initialize(root, name, len(dataset))
        except BaseException as error:
            initialization_error = error
    if world_size > 1:
        reported: list[object] = [
            repr(initialization_error) if rank == 0 and initialization_error is not None else None
        ]
        torch.distributed.broadcast_object_list(reported, src=0)
        if rank != 0 and reported[0] is not None:
            raise RuntimeError(f"representation initialization failed: {reported[0]}")
    if initialization_error is not None:
        raise initialization_error
    _barrier()
    caches = {
        name: LayerMemmap.initialize(root, name, len(dataset))
        for name in ("dino12", "dino8", "dino4")
    }
    encoder = load_dino_encoder(config.encoder, device, compile_config=config.compile)
    _extract_caches(dataset, encoder, caches, config, rank, world_size, device)
    _barrier()
    for cache in caches.values():
        cache.finalize()
    statistics = _collect_statistics(dataset, caches, config, rank, world_size, device)
    if config.alignment.enabled:
        matrices, diagnostics = _fit_alignments(
            dataset, caches, statistics, config, rank, world_size, device
        )
    else:
        matrices = {pair: torch.eye(TOKEN_DIM) for pair in config.alignment.pairs}
        diagnostics = {
            pair: AlignmentDiagnostics(0.0, 0.0, 0) for pair in config.alignment.pairs
        }
    cumulative = cumulative_projectors(matrices)
    if config.alignment.direct_layer12_to_rgb:
        cumulative["rgb"] = matrices["dino12->rgb"]
    if rank == 0:
        save_file(
            {
                **{f"pairwise.{name}": matrix.float() for name, matrix in matrices.items()},
                **{f"cumulative.{name}": matrix.float() for name, matrix in cumulative.items()},
                **{
                    f"statistics.{name}.mean": values.mean.float()
                    for name, values in statistics.items()
                },
                **{
                    f"statistics.{name}.std": values.std.float()
                    for name, values in statistics.items()
                },
            },
            root / "representations.safetensors",
        )
        manifest = RepresentationManifest(
            dataset_size=len(dataset),
            dataset_fingerprint=data_fingerprint(config.data, config.data.image_size),
            encoder_fingerprint=encoder_fingerprint(config.encoder),
            preparation_fingerprint=preparation_fingerprint(config),
            statistics={name: _statistics_payload(values) for name, values in statistics.items()},
            cache_files={name: f"{name}.bf16" for name in caches},
            completion_files={name: f"{name}.completion.u8" for name in caches},
            alignments={name: vars(values) | {"matrix": f"pairwise.{name}"} for name, values in diagnostics.items()},
            cumulative_projectors={name: f"cumulative.{name}" for name in cumulative},
            complete=True,
        )
        manifest.save(manifest_path)
    _barrier()
    return RepresentationManifest.load(manifest_path)
