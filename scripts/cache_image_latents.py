from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from cascade_dit.config import load_config
from cascade_dit.data import load_imagenet
from cascade_dit.distributed import initialize_distributed
from cascade_dit.image_latent_cache import (
    AsyncImageLatentCacheWriter,
    ImageLatentCacheWriter,
    image_latent_cache_path,
    image_latent_cache_spec,
    preflight_image_latent_cache_capacity,
)
from cascade_dit.mage import file_sha256, load_mage_vae_codec
from cascade_dit.representation_cache import cache_build_lock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache deterministic native Mage-VAE image latents"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/latent-baseline.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main() -> None:
    from huggingface_hub import hf_hub_download

    args = build_parser().parse_args()
    config = load_config(args.config, args.override)
    rank, world_size, device = initialize_distributed()
    weights = hf_hub_download(
        repo_id=config.image_latents.repository,
        filename=config.image_latents.weight_file,
        revision=config.image_latents.revision,
    )
    weight_digest = file_sha256(weights)
    dataset = load_imagenet(config.data, config.image_latents.image_size, load_pixels=True)
    spec = image_latent_cache_spec(config, len(dataset), weight_digest)
    path = image_latent_cache_path(config, spec)
    lock = (
        cache_build_lock(config.image_latents.cache.directory)
        if rank == 0
        else contextlib.nullcontext()
    )
    with lock:
        if rank == 0:
            required = preflight_image_latent_cache_capacity(
                config.image_latents.cache.directory, spec, path
            )
            print(
                f"image-latent capacity preflight: {required / 2**30:.2f} GiB additional",
                flush=True,
            )
            complete = ImageLatentCacheWriter.initialize(path, spec)
        else:
            complete = False
        if world_size > 1:
            values: list[object] = [complete]
            torch.distributed.broadcast_object_list(values, src=0)
            complete = bool(values[0])
            torch.distributed.barrier()
        if complete:
            if rank == 0:
                print(f"image latent cache is already complete: {path}", flush=True)
            return
        writer = ImageLatentCacheWriter(path, spec)
        pending = writer.pending_indices(rank, world_size)
        pipeline_error: BaseException | None = None
        if len(pending):
            loader = DataLoader(
                Subset(dataset, pending.tolist()),
                batch_size=config.image_latents.cache.batch_size,
                shuffle=False,
                num_workers=config.data.workers,
                pin_memory=device.type == "cuda",
                persistent_workers=config.data.workers > 0,
                prefetch_factor=(
                    config.data.prefetch_factor if config.data.workers > 0 else None
                ),
            )
            codec = load_mage_vae_codec(config.image_latents, device)
            pipeline = AsyncImageLatentCacheWriter(
                writer,
                config.image_latents.cache.batch_size,
                device,
                enabled=config.image_latents.cache.asynchronous_writes,
            )
            started = time.perf_counter()
            processed = 0
            try:
                for batch_index, (pixels, _labels, indices) in enumerate(loader, start=1):
                    buffer = pipeline.acquire_buffer()
                    try:
                        image_latents = codec.encode_pixels(
                            pixels.to(device, non_blocking=True)
                        )
                    except BaseException:
                        pipeline.release_buffer(buffer)
                        raise
                    pipeline.submit(indices, image_latents, buffer=buffer)
                    processed += indices.numel()
                    if rank == 0 and batch_index % config.image_latents.cache.log_interval == 0:
                        elapsed = max(time.perf_counter() - started, 1e-9)
                        print(
                            f"cached {processed}/{len(pending)} local image latents "
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
        if world_size > 1:
            errors: list[object] = [None] * world_size
            torch.distributed.all_gather_object(
                errors, repr(pipeline_error) if pipeline_error is not None else None
            )
            failures = [error for error in errors if error is not None]
            if failures:
                raise RuntimeError(f"image latent cache failed: {failures}") from pipeline_error
            torch.distributed.barrier()
        elif pipeline_error is not None:
            raise pipeline_error
        if rank == 0:
            writer.finalize()
            print(f"image latent cache complete: {path}", flush=True)
        if world_size > 1:
            torch.distributed.barrier()


if __name__ == "__main__":
    main()
