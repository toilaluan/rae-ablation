from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from jit_reg.config import load_config
from jit_reg.data import load_imagenet, prepare_pixels
from jit_reg.distributed import initialize_distributed
from jit_reg.rae import load_rae_codec
from jit_reg.token_cache import (
    AsyncTokenCacheWriter,
    TokenCacheWriter,
    preflight_token_cache_capacity,
    token_cache_build_lock,
    token_cache_path,
    token_cache_spec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache raw BF16 RAEv2 compressor tokens")
    parser.add_argument("--config", type=Path, default=Path("configs/ddt-token-mix.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.override)
    rank, world_size, device = initialize_distributed()
    dataset = load_imagenet(config.data, 256, load_pixels=True)
    spec = token_cache_spec(config, len(dataset))
    path = token_cache_path(config, len(dataset))
    lock = token_cache_build_lock(config.rae.cache.directory) if rank == 0 else contextlib.nullcontext()
    with lock:
        if rank == 0:
            required = preflight_token_cache_capacity(config.rae.cache.directory, spec, path)
            print(f"token-cache capacity preflight: {required / 2**30:.2f} GiB additional space required", flush=True)
            complete = TokenCacheWriter.initialize(path, spec)
        else:
            complete = False
        if world_size > 1:
            values: list[object] = [complete]
            torch.distributed.broadcast_object_list(values, src=0)
            complete = bool(values[0])
            torch.distributed.barrier()
        if complete:
            if rank == 0:
                print(f"token cache is already complete: {path}", flush=True)
            return

        writer = TokenCacheWriter(path, spec)
        pending = writer.pending_indices(rank, world_size)
        pipeline_error: BaseException | None = None
        if len(pending):
            loader = DataLoader(
                Subset(dataset, pending.tolist()),
                batch_size=config.rae.cache.batch_size,
                shuffle=False,
                num_workers=config.data.workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=config.data.workers > 0,
                prefetch_factor=config.data.prefetch_factor if config.data.workers > 0 else None,
            )
            codec = load_rae_codec(config.rae, device)
            pipeline = AsyncTokenCacheWriter(writer, config.rae.cache.batch_size, device)
            started = time.perf_counter()
            processed = 0
            try:
                for batch_index, batch in enumerate(loader, start=1):
                    pixels, _labels, indices = batch
                    buffer = pipeline.acquire_buffer()
                    try:
                        tokens = codec.encode(
                            prepare_pixels(pixels, device, non_blocking=True)
                        )
                    except BaseException:
                        pipeline.release_buffer(buffer)
                        raise
                    pipeline.submit(indices, tokens, buffer=buffer)
                    processed += indices.numel()
                    if rank == 0 and batch_index % config.rae.cache.log_interval == 0:
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
            del codec
        if world_size > 1:
            errors: list[object] = [None] * world_size
            torch.distributed.all_gather_object(
                errors, repr(pipeline_error) if pipeline_error is not None else None
            )
            failures = [error for error in errors if error is not None]
            if failures:
                raise RuntimeError(
                    f"token-cache pipeline failed on ranks: {failures}"
                ) from pipeline_error
        elif pipeline_error is not None:
            raise pipeline_error
        if world_size > 1:
            torch.distributed.barrier()

        # This is deliberately a separate resumable phase: no codec is loaded when
        # the bitmap was already complete at process start.
        mean, std = writer.compute_statistics(rank, world_size, device=device)
        if rank == 0:
            writer.finalize(mean, std)
            print(f"token cache complete with channel statistics: {path}", flush=True)
        if world_size > 1:
            torch.distributed.barrier()


if __name__ == "__main__":
    main()
