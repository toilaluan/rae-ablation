from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from cascade_dit.config import load_config
from cascade_dit.data import data_fingerprint, load_imagenet
from cascade_dit.distributed import initialize_distributed
from cascade_dit.representation_cache import cache_build_lock
from cascade_dit.semantic_statistics import (
    SemanticTokenNormalizer,
    load_semantic_statistics,
    semantic_statistics_sha256,
)
from cascade_dit.semantic_token_cache import (
    AsyncSemanticTokenCacheWriter,
    SemanticTokenCacheWriter,
    preflight_semantic_token_cache_capacity,
    semantic_token_cache_path,
    semantic_token_cache_spec,
)
from cascade_dit.vision import (
    load_dino_semantic_encoder,
    semantic_encoder_fingerprint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache normalized DINO semantic tokens")
    parser.add_argument("--config", type=Path, default=Path("configs/cascade-shared.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.override)
    rank, world_size, device = initialize_distributed()
    statistics_path = Path(config.semantic_tokens.statistics_path).resolve()
    statistics = load_semantic_statistics(
        statistics_path,
        expected_channels=config.semantic_tokens.channels,
        expected_semantic_encoder_fingerprint=semantic_encoder_fingerprint(
            config.semantic_tokens
        ),
        expected_data_fingerprint=data_fingerprint(
            config.data, config.image_latents.image_size
        ),
    )
    statistics_digest = semantic_statistics_sha256(statistics_path)
    dataset = load_imagenet(config.data, config.image_latents.image_size, load_pixels=True)
    spec = semantic_token_cache_spec(config, len(dataset), statistics_digest)
    path = semantic_token_cache_path(config, spec)
    lock = (
        cache_build_lock(config.semantic_tokens.cache.directory)
        if rank == 0
        else contextlib.nullcontext()
    )
    with lock:
        if rank == 0:
            required = preflight_semantic_token_cache_capacity(
                config.semantic_tokens.cache.directory, spec, path
            )
            print(
                f"semantic-token capacity preflight: {required / 2**30:.2f} GiB additional",
                flush=True,
            )
            complete = SemanticTokenCacheWriter.initialize(path, spec)
        else:
            complete = False
        if world_size > 1:
            values: list[object] = [complete]
            torch.distributed.broadcast_object_list(values, src=0)
            complete = bool(values[0])
            torch.distributed.barrier()
        if complete:
            if rank == 0:
                print(f"semantic token cache is already complete: {path}", flush=True)
            return
        writer = SemanticTokenCacheWriter(path, spec)
        pending = writer.pending_indices(rank, world_size)
        pipeline_error: BaseException | None = None
        if len(pending):
            loader = DataLoader(
                Subset(dataset, pending.tolist()),
                batch_size=config.semantic_tokens.cache.batch_size,
                shuffle=False,
                num_workers=config.data.workers,
                pin_memory=device.type == "cuda",
                persistent_workers=config.data.workers > 0,
                prefetch_factor=(
                    config.data.prefetch_factor if config.data.workers > 0 else None
                ),
            )
            encoder = load_dino_semantic_encoder(
                config.semantic_tokens,
                device,
                normalizer=SemanticTokenNormalizer(statistics),
            )
            pipeline = AsyncSemanticTokenCacheWriter(
                writer,
                config.semantic_tokens.cache.batch_size,
                device,
                enabled=config.semantic_tokens.cache.asynchronous_writes,
            )
            started = time.perf_counter()
            processed = 0
            try:
                for batch_index, (pixels, _labels, indices) in enumerate(loader, start=1):
                    buffer = pipeline.acquire_buffer()
                    try:
                        semantic_tokens = encoder.encode(
                            pixels.to(device, non_blocking=True)
                        )
                    except BaseException:
                        pipeline.release_buffer(buffer)
                        raise
                    pipeline.submit(indices, semantic_tokens, buffer=buffer)
                    processed += indices.numel()
                    if rank == 0 and batch_index % config.semantic_tokens.cache.log_interval == 0:
                        elapsed = max(time.perf_counter() - started, 1e-9)
                        print(
                            f"cached {processed}/{len(pending)} local semantic tokens "
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
                raise RuntimeError(f"semantic token cache failed: {failures}") from pipeline_error
            torch.distributed.barrier()
        elif pipeline_error is not None:
            raise pipeline_error
        if rank == 0:
            writer.finalize()
            print(f"semantic token cache complete: {path}", flush=True)
        if world_size > 1:
            torch.distributed.barrier()


if __name__ == "__main__":
    main()
