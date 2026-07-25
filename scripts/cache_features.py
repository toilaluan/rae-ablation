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
    FeatureComponentCacheWriter,
    cache_build_lock,
    component_path,
    component_spec,
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
        with torch.inference_mode():
            for batch_index, (pixels, _labels, indices) in enumerate(loader, start=1):
                pixels = pixels.to(device, non_blocking=True)
                with torch.autocast(
                    device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    targets = extractor.targets(pixels, config.model.grid_size)
                if "reg" in writers:
                    if targets.reg is None:
                        raise RuntimeError("extractor did not return active REG targets")
                    writers["reg"].write(indices, targets.reg)
                if "repa" in writers:
                    if targets.repa is None:
                        raise RuntimeError("extractor did not return active REPA targets")
                    writers["repa"].write(indices, targets.repa)
                processed += indices.numel()
                if (
                    rank == 0
                    and batch_index % config.feature_cache.log_interval == 0
                ):
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(
                        f"cached {processed}/{len(pending)} local examples "
                        f"({processed / elapsed:.1f} images/s)",
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
