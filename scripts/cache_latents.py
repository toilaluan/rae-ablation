from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from jit_reg.config import load_config
from jit_reg.data import load_imagenet
from jit_reg.distributed import initialize_distributed
from jit_reg.latent_cache import (
    AsyncLatentCacheWriter,
    LatentCacheWriter,
    latent_cache_build_lock,
    latent_cache_path,
    latent_cache_spec,
    preflight_latent_cache_capacity,
)
from jit_reg.vae import load_flux2_codec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute normalized FLUX.2 posterior-mode latents"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/latent-baseline-p16.yaml"),
        help="schema-v9 latent experiment YAML",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="repeatable structured configuration override",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.override)
    if config.model.input_space != "flux2_latents":
        raise ValueError("latent-cache building requires FLUX.2 latent input space")
    rank, world_size, device = initialize_distributed()
    dataset = load_imagenet(config.data, config.model.image_size)
    spec = latent_cache_spec(config, len(dataset))
    path = latent_cache_path(config, len(dataset))
    lock = (
        latent_cache_build_lock(config.vae.cache.directory)
        if rank == 0
        else contextlib.nullcontext()
    )
    with lock:
        if rank == 0:
            required = preflight_latent_cache_capacity(
                config.vae.cache.directory, spec, path
            )
            print(
                f"latent-cache capacity preflight: {required / 2**30:.1f} GiB "
                "additional space required",
                flush=True,
            )
            complete = LatentCacheWriter.initialize(path, spec)
        else:
            complete = False
        if world_size > 1:
            values: list[object] = [complete]
            torch.distributed.broadcast_object_list(values, src=0)
            complete = bool(values[0])
            torch.distributed.barrier()
        if complete:
            if rank == 0:
                print(f"latent cache is already complete: {path}", flush=True)
            return

        writer = LatentCacheWriter(path, spec)
        pending = writer.pending_indices(rank, world_size)
        loader = DataLoader(
            Subset(dataset, pending.tolist()),
            batch_size=config.vae.cache.batch_size,
            shuffle=False,
            num_workers=config.data.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.data.workers > 0,
            prefetch_factor=(
                config.data.prefetch_factor if config.data.workers > 0 else None
            ),
        )
        codec = load_flux2_codec(
            config.vae,
            image_size=config.model.image_size,
            device=device,
        )
        pipeline = AsyncLatentCacheWriter(
            writer, config.vae.cache.batch_size, device
        )
        started = time.perf_counter()
        processed = 0
        pipeline_error: Exception | None = None
        try:
            for batch_index, batch in enumerate(loader, start=1):
                pixels, _labels, indices = batch
                buffer = pipeline.acquire_buffer()
                try:
                    pixels = pixels.to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    ).div_(255)
                    latents = codec.encode(pixels)
                except BaseException:
                    pipeline.release_buffer(buffer)
                    raise
                pipeline.submit(indices, latents, buffer=buffer)
                processed += indices.numel()
                if (
                    rank == 0
                    and batch_index % config.vae.cache.log_interval == 0
                ):
                    metrics = pipeline.metrics()
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(
                        f"cached {processed}/{len(pending)} local examples "
                        f"({processed / elapsed:.1f} images/s; "
                        f"buffer backpressure "
                        f"{metrics.buffer_backpressure_seconds:.2f}s, "
                        f"copy {metrics.copy_seconds:.2f}s, "
                        f"write/flush {metrics.write_flush_seconds:.2f}s)",
                        flush=True,
                    )
            pipeline.drain()
        except Exception as error:
            pipeline_error = error
        finally:
            try:
                pipeline.close()
            except Exception as error:
                if pipeline_error is None:
                    pipeline_error = error
        if world_size > 1:
            errors: list[object] = [None] * world_size
            torch.distributed.all_gather_object(
                errors,
                repr(pipeline_error) if pipeline_error is not None else None,
            )
            failed = [error for error in errors if error is not None]
            if failed:
                if pipeline_error is not None:
                    raise RuntimeError(
                        f"latent-cache pipeline failed on ranks: {failed}"
                    ) from pipeline_error
                raise RuntimeError(
                    f"latent-cache pipeline failed on ranks: {failed}"
                )
        elif pipeline_error is not None:
            raise pipeline_error
        if world_size > 1:
            torch.distributed.barrier()
        if rank == 0:
            writer.finalize()
            metrics = pipeline.metrics()
            print(
                f"latent cache complete: {path} "
                f"({metrics.images_per_second:.1f} images/s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
