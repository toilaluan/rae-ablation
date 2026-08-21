from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from jit_reg.config import load_config
from jit_reg.data import data_fingerprint, load_imagenet
from jit_reg.distributed import initialize_distributed
from jit_reg.endpoint_cache import (
    AsyncEndpointCacheWriter,
    EndpointCacheWriter,
    endpoint_cache_build_lock,
    endpoint_cache_path,
    endpoint_cache_spec,
    preflight_endpoint_cache_capacity,
)
from jit_reg.latent_statistics import (
    EndpointLatentNormalizer,
    latent_statistics_sha256,
    load_latent_statistics,
)
from jit_reg.vision import load_dino_encoder, vision_fingerprint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache normalized block-2 and block-12 DINO endpoints"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dino-waypoint.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.override)
    rank, world_size, device = initialize_distributed()
    statistics_path = Path(config.vision.statistics_path).resolve()
    statistics = load_latent_statistics(
        statistics_path,
        expected_channels=config.vision.hidden_size,
        expected_vision_fingerprint=vision_fingerprint(config.vision),
        expected_data_fingerprint=data_fingerprint(config.data, config.vision.image_size),
    )
    statistics_digest = latent_statistics_sha256(statistics_path)
    dataset = load_imagenet(config.data, config.vision.image_size, load_pixels=True)
    spec = endpoint_cache_spec(config, len(dataset), statistics_digest)
    path = endpoint_cache_path(config, spec)
    lock = (
        endpoint_cache_build_lock(config.vision.cache.directory)
        if rank == 0
        else contextlib.nullcontext()
    )
    with lock:
        if rank == 0:
            required = preflight_endpoint_cache_capacity(
                config.vision.cache.directory, spec, path
            )
            print(
                f"endpoint-cache capacity preflight: {required / 2**30:.2f} GiB additional",
                flush=True,
            )
            complete = EndpointCacheWriter.initialize(path, spec)
        else:
            complete = False
        if world_size > 1:
            values: list[object] = [complete]
            torch.distributed.broadcast_object_list(values, src=0)
            complete = bool(values[0])
            torch.distributed.barrier()
        if complete:
            if rank == 0:
                print(f"endpoint cache is already complete: {path}", flush=True)
            return

        writer = EndpointCacheWriter(path, spec)
        pending = writer.pending_indices(rank, world_size)
        pipeline_error: BaseException | None = None
        if len(pending):
            loader = DataLoader(
                Subset(dataset, pending.tolist()),
                batch_size=config.vision.cache.batch_size,
                shuffle=False,
                num_workers=config.data.workers,
                pin_memory=device.type == "cuda",
                persistent_workers=config.data.workers > 0,
                prefetch_factor=(
                    config.data.prefetch_factor if config.data.workers > 0 else None
                ),
            )
            encoder = load_dino_encoder(
                config.vision,
                device,
                include_semantic=True,
                normalizer=EndpointLatentNormalizer(statistics),
            )
            pipeline = AsyncEndpointCacheWriter(
                writer, config.vision.cache.batch_size, device
            )
            started = time.perf_counter()
            processed = 0
            try:
                for batch_index, (pixels, _labels, indices) in enumerate(loader, start=1):
                    buffer = pipeline.acquire_buffer()
                    try:
                        target, semantic = encoder.encode_pair(
                            pixels.to(device, non_blocking=True)
                        )
                    except BaseException:
                        pipeline.release_buffer(buffer)
                        raise
                    pipeline.submit(indices, target, semantic, buffer=buffer)
                    processed += indices.numel()
                    if rank == 0 and batch_index % config.vision.cache.log_interval == 0:
                        elapsed = max(time.perf_counter() - started, 1e-9)
                        print(
                            f"cached {processed}/{len(pending)} local examples "
                            f"({processed / elapsed:.1f} images/s)",
                            flush=True,
                        )
                pipeline.drain()
            except BaseException as error:
                pipeline_error = error
            finally:
                try:
                    pipeline.close()
                except BaseException as error:
                    if pipeline_error is None:
                        pipeline_error = error
            del encoder
        if world_size > 1:
            errors: list[object] = [None] * world_size
            torch.distributed.all_gather_object(
                errors, repr(pipeline_error) if pipeline_error is not None else None
            )
            failures = [error for error in errors if error is not None]
            if failures:
                raise RuntimeError(f"endpoint-cache pipeline failed: {failures}") from pipeline_error
            torch.distributed.barrier()
        elif pipeline_error is not None:
            raise pipeline_error
        if rank == 0:
            writer.finalize()
            print(
                f"endpoint cache complete: {path} "
                f"fingerprint={path.name}",
                flush=True,
            )
        if world_size > 1:
            torch.distributed.barrier()


if __name__ == "__main__":
    main()
