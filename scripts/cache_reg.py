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
from jit_reg.reg import DINOv3FeatureExtractor
from jit_reg.reg_cache import (
    REGTokenCacheWriter,
    cache_build_lock,
    cache_path,
    cache_spec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute configured DINO REG tokens without full spatial features"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/canonical.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.log_interval < 1:
        raise ValueError("cache batch size and log interval must be positive")
    config = load_config(args.config, args.override)
    config.model.reg.repa_enabled = False
    rank, world_size, device = initialize_distributed()
    dataset = load_imagenet(config.data, config.model.image_size)
    directory = cache_path(config, len(dataset))
    spec = cache_spec(config, len(dataset))

    lock_context = cache_build_lock(directory) if rank == 0 else contextlib.nullcontext()
    with lock_context:
        complete = REGTokenCacheWriter.initialize(directory, spec) if rank == 0 else False
        if world_size > 1:
            values: list[object] = [complete]
            torch.distributed.broadcast_object_list(values, src=0)
            complete = bool(values[0])
            torch.distributed.barrier()
        if complete:
            if rank == 0:
                print(f"REG cache already complete: {directory}", flush=True)
            return

        writer = REGTokenCacheWriter(directory, spec)
        pending = writer.pending_indices(rank, world_size)
        subset = Subset(dataset, pending.tolist())
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.data.workers > 0,
            prefetch_factor=config.data.prefetch_factor if config.data.workers > 0 else None,
        )
        extractor = DINOv3FeatureExtractor(config.model.reg).to(device).bfloat16()
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
                    tokens = extractor.targets(pixels, config.model.grid_size).reg_tokens
                writer.write(indices, tokens)
                processed += indices.numel()
                if rank == 0 and batch_index % args.log_interval == 0:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(
                        f"cached {processed}/{len(pending)} local examples "
                        f"({processed / elapsed:.1f} images/s)",
                        flush=True,
                    )

        if world_size > 1:
            torch.distributed.barrier()
        if rank == 0:
            writer.finalize()
            print(f"REG cache complete: {directory}", flush=True)


if __name__ == "__main__":
    main()
