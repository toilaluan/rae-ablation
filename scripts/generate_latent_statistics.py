from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jit_reg.config import load_config
from jit_reg.data import data_fingerprint, load_imagenet, patchify_rgb, prepare_pixels
from jit_reg.distributed import initialize_distributed
from jit_reg.latent_statistics import (
    LatentStatisticsAccumulator,
    latent_statistics_sha256,
    save_latent_statistics,
)
from jit_reg.vision import load_dino_encoder, vision_fingerprint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute raw target, semantic, and RGB-patch statistics"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dino-cascade.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=32, help="batch size per process")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.log_interval < 1:
        raise ValueError("batch size and log interval must be positive")
    config = load_config(args.config, args.override)
    output = args.output or Path(config.vision.statistics_path)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"statistics file already exists: {output}; pass --overwrite")

    rank, world_size, device = initialize_distributed()
    dataset = load_imagenet(
        config.data,
        config.vision.image_size,
        load_pixels=True,
        split=config.data.split,
    )
    indices = range(rank, len(dataset), world_size)
    workers = config.data.workers
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=indices,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=config.data.prefetch_factor if workers > 0 else None,
    )
    encoder = load_dino_encoder(config.vision, device, include_semantic=True)
    accumulator = LatentStatisticsAccumulator(
        config.vision.hidden_size,
        device,
    )
    started = time.perf_counter()
    for batch_index, (pixels, _labels, _sample_indices) in enumerate(loader, start=1):
        raw_pixels = pixels.to(device, non_blocking=True)
        endpoints = encoder.encode_pair(raw_pixels)
        pixel_patches = patchify_rgb(prepare_pixels(raw_pixels, device))
        accumulator.update(
            endpoints.target,
            endpoints.semantic,
            pixel_patches,
        )
        if rank == 0 and batch_index % args.log_interval == 0:
            print(
                f"statistics batches={batch_index}/{len(loader)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    accumulator.all_reduce()

    expected_count = len(dataset) * config.vision.grid_size**2
    if not torch.all(accumulator.count == expected_count):
        raise RuntimeError(
            "statistics observation count mismatch: "
            f"spatial={accumulator.count.tolist()}, expected={expected_count}"
        )
    if rank == 0:
        statistics = accumulator.finalize(
            vision_fingerprint=vision_fingerprint(config.vision),
            data_fingerprint=data_fingerprint(config.data, config.vision.image_size),
        )
        save_latent_statistics(statistics, output)
        print(
            f"saved {output}: patch_observations={statistics.patch_count}, "
            f"channels={statistics.channels}, sha256={latent_statistics_sha256(output)}",
            flush=True,
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
