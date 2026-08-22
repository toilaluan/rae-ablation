from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cascade_dit.config import load_config
from cascade_dit.data import data_fingerprint, load_imagenet
from cascade_dit.distributed import initialize_distributed
from cascade_dit.semantic_statistics import (
    SemanticStatisticsAccumulator,
    save_semantic_statistics,
    semantic_statistics_sha256,
)
from cascade_dit.vision import (
    load_dino_semantic_encoder,
    semantic_encoder_fingerprint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute per-channel statistics for unpooled DINO semantic tokens"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/cascade-shared.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.log_interval < 1:
        raise ValueError("batch size and log interval must be positive")
    config = load_config(args.config, args.override)
    output = args.output or Path(config.semantic_tokens.statistics_path)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"statistics already exist: {output}; pass --overwrite")
    rank, world_size, device = initialize_distributed()
    dataset = load_imagenet(config.data, config.image_latents.image_size, load_pixels=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=range(rank, len(dataset), world_size),
        num_workers=config.data.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.data.workers > 0,
        prefetch_factor=(
            config.data.prefetch_factor if config.data.workers > 0 else None
        ),
    )
    encoder = load_dino_semantic_encoder(config.semantic_tokens, device)
    accumulator = SemanticStatisticsAccumulator(config.semantic_tokens.channels, device)
    started = time.perf_counter()
    for batch_index, (pixels, _labels, _indices) in enumerate(loader, start=1):
        accumulator.update(encoder.encode(pixels.to(device, non_blocking=True)))
        if rank == 0 and batch_index % args.log_interval == 0:
            print(
                f"semantic statistics batches={batch_index}/{len(loader)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    accumulator.all_reduce()
    expected_count = len(dataset) * config.semantic_tokens.grid_size**2
    if int(accumulator.count.item()) != expected_count:
        raise RuntimeError(
            f"semantic observation count={int(accumulator.count.item())}, expected={expected_count}"
        )
    if rank == 0:
        statistics = accumulator.finalize(
            semantic_encoder_fingerprint=semantic_encoder_fingerprint(
                config.semantic_tokens
            ),
            data_fingerprint=data_fingerprint(
                config.data, config.image_latents.image_size
            ),
        )
        save_semantic_statistics(statistics, output)
        print(
            f"saved {output}: observations={statistics.count}, "
            f"channels={statistics.channels}, sha256={semantic_statistics_sha256(output)}",
            flush=True,
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
