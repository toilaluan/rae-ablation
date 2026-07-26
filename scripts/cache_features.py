from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from jit_reg.config import load_config
from jit_reg.data import load_imagenet
from jit_reg.distributed import initialize_distributed
from jit_reg.feature_cache import (
    AsyncFeatureCacheWriter,
    FeatureComponentCacheWriter,
    cache_build_lock,
    component_path,
    component_spec,
    preflight_cache_capacity,
)
from jit_reg.reg import DINOv3FeatureExtractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute independent DINO REG and REPA feature components"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/baseline-p16.yaml")
    )
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.override)
    if not config.model.features.enabled:
        raise ValueError("feature-cache building requires REG or REPA to be enabled")
    rank, world_size, device = initialize_distributed()
    dataset = load_imagenet(config.data, config.model.image_size)
    components = [
        name
        for name, enabled in (
            ("reg", config.model.features.reg.enabled),
            ("repa", config.model.features.repa.enabled),
        )
        if enabled
    ]
    specs = {
        name: component_spec(config, len(dataset), name) for name in components
    }
    paths = {
        name: component_path(config, len(dataset), name) for name in components
    }

    lock_context = (
        cache_build_lock(config.feature_cache.directory)
        if rank == 0
        else contextlib.nullcontext()
    )
    with lock_context:
        if rank == 0:
            required = preflight_cache_capacity(
                config.feature_cache.directory,
                specs,  # type: ignore[arg-type]
                paths,  # type: ignore[arg-type]
            )
            print(
                f"feature-cache capacity preflight: {required / 2**30:.1f} GiB "
                "additional space required",
                flush=True,
            )
        complete = (
            {
                name: FeatureComponentCacheWriter.initialize(paths[name], specs[name])
                for name in components
            }
            if rank == 0
            else {}
        )
        if world_size > 1:
            values: list[object] = [complete]
            torch.distributed.broadcast_object_list(values, src=0)
            complete = dict(values[0])  # type: ignore[arg-type]
            torch.distributed.barrier()
        missing = [name for name in components if not complete[name]]
        if not missing:
            if rank == 0:
                print("all requested feature-cache components are complete", flush=True)
            return

        writers = {
            name: FeatureComponentCacheWriter(paths[name], specs[name])
            for name in missing
        }
        pending = np.unique(
            np.concatenate(
                [
                    writer.pending_indices(rank, world_size)
                    for writer in writers.values()
                ]
            )
        )
        loader = DataLoader(
            Subset(dataset, pending.tolist()),
            batch_size=config.feature_cache.batch_size,
            shuffle=False,
            num_workers=config.data.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.data.workers > 0,
            prefetch_factor=(
                config.data.prefetch_factor if config.data.workers > 0 else None
            ),
        )
        extractor = DINOv3FeatureExtractor(config.model.features).to(device).bfloat16()
        extractor.eval()
        if config.compile.enabled:
            extractor = torch.compile(
                extractor,
                backend=config.compile.backend,
                mode=config.compile.mode,
                fullgraph=config.compile.fullgraph,
                dynamic=config.compile.dynamic,
            )
        started = time.perf_counter()
        processed = 0
        accumulated_data_wait = 0.0
        accumulated_gpu_extraction = 0.0
        gpu_extraction_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        loader_iterator = iter(loader)
        pipeline = AsyncFeatureCacheWriter(
            writers, config.feature_cache.batch_size, device  # type: ignore[arg-type]
        )
        pipeline_error: Exception | None = None
        try:
            with torch.inference_mode():
                for batch_index in range(1, len(loader) + 1):
                    data_started = time.perf_counter()
                    pixels, _labels, indices = next(loader_iterator)
                    accumulated_data_wait += time.perf_counter() - data_started
                    buffer = pipeline.acquire_buffer()
                    extraction_started = time.perf_counter()
                    gpu_started = None
                    gpu_finished = None
                    if device.type == "cuda":
                        gpu_started = torch.cuda.Event(enable_timing=True)
                        gpu_finished = torch.cuda.Event(enable_timing=True)
                        gpu_started.record()
                    try:
                        pixels = pixels.to(device, non_blocking=True)
                        with torch.autocast(
                            device.type,
                            dtype=torch.bfloat16,
                            enabled=device.type == "cuda",
                        ):
                            targets = extractor.targets(
                                pixels, config.model.grid_size
                            )
                    except BaseException:
                        pipeline.release_buffer(buffer)
                        raise
                    if gpu_finished is not None:
                        gpu_finished.record()
                    else:
                        accumulated_gpu_extraction += (
                            time.perf_counter() - extraction_started
                        )
                    pipeline.submit(indices, targets, buffer=buffer)
                    if gpu_finished is not None:
                        assert gpu_started is not None
                        gpu_extraction_events.append((gpu_started, gpu_finished))
                    processed += indices.numel()
                    if (
                        rank == 0
                        and batch_index % config.feature_cache.log_interval == 0
                    ):
                        if gpu_extraction_events:
                            gpu_extraction_events[-1][1].synchronize()
                            accumulated_gpu_extraction += sum(
                                event_started.elapsed_time(event_finished)
                                for event_started, event_finished in gpu_extraction_events
                            ) / 1_000
                            gpu_extraction_events.clear()
                        metrics = pipeline.metrics()
                        elapsed = max(time.perf_counter() - started, 1e-9)
                        print(
                            f"cached {processed}/{len(pending)} local examples "
                            f"({processed / elapsed:.1f} images/s total; "
                            f"data {accumulated_data_wait:.2f}s, "
                            f"GPU extraction {accumulated_gpu_extraction:.2f}s, "
                            f"buffer backpressure "
                            f"{metrics.buffer_backpressure_seconds:.2f}s, "
                            f"copy {metrics.copy_seconds:.2f}s, "
                            f"write/flush {metrics.write_flush_seconds:.2f}s)",
                            flush=True,
                        )
            pipeline.drain()
            if gpu_extraction_events:
                gpu_extraction_events[-1][1].synchronize()
                accumulated_gpu_extraction += sum(
                    event_started.elapsed_time(event_finished)
                    for event_started, event_finished in gpu_extraction_events
                ) / 1_000
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
                        f"feature-cache pipeline failed on ranks: {failed}"
                    ) from pipeline_error
                raise RuntimeError(f"feature-cache pipeline failed on ranks: {failed}")
        elif pipeline_error is not None:
            raise pipeline_error
        if rank == 0:
            metrics = pipeline.metrics()
            print(
                f"cache pipeline drained ({metrics.images_per_second:.1f} images/s "
                f"total; data {accumulated_data_wait:.2f}s, "
                f"GPU extraction {accumulated_gpu_extraction:.2f}s, "
                f"buffer backpressure {metrics.buffer_backpressure_seconds:.2f}s, "
                f"copy {metrics.copy_seconds:.2f}s, "
                f"write/flush {metrics.write_flush_seconds:.2f}s)",
                flush=True,
            )
        if world_size > 1:
            torch.distributed.barrier()
        if rank == 0:
            for name, writer in writers.items():
                writer.finalize()
                print(f"{name.upper()} cache complete: {paths[name]}", flush=True)


if __name__ == "__main__":
    main()
