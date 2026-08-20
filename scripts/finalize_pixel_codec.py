from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from jit_reg.codec_training import (
    CodecBatch,
    compute_pixel_token_statistics,
    make_statistics_loader,
)
from jit_reg.config import IMAGE_SIZE, load_config
from jit_reg.data import data_fingerprint, load_imagenet, preprocessing_metadata
from jit_reg.distributed import initialize_distributed
from jit_reg.pixel_codec import load_pixel_codec_checkpoint, save_pixel_codec_artifact
from jit_reg.token_cache import TokenCache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute token statistics and turn a codec checkpoint into a portable artifact"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pixel-codec.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="global statistics batch size (defaults to codec_train.global_batch_size)",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


class ProgressLoader:
    def __init__(self, loader: Any, log_interval: int, *, enabled: bool) -> None:
        self.loader = loader
        self.log_interval = log_interval
        self.enabled = enabled
        self.pass_index = 0

    def __iter__(self) -> Iterator[CodecBatch]:
        self.pass_index += 1
        name = "moments" if self.pass_index == 1 else "pixel/RAE correlation"
        total = len(self.loader)
        for index, batch in enumerate(self.loader, start=1):
            yield batch
            if self.enabled and (index % self.log_interval == 0 or index == total):
                print(f"statistics ({name}): {index}/{total} batches", flush=True)


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "min": values.min().item(),
        "max": values.max().item(),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be positive")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"output artifact already exists: {args.output.resolve()}")

    config = load_config(args.config, args.override)
    if args.batch_size is not None:
        config.codec_train.global_batch_size = args.batch_size
        config.validate()
    rank, world_size, device = initialize_distributed()
    checkpoint = load_pixel_codec_checkpoint(args.checkpoint, config.pixel_codec, device)
    dataset = load_imagenet(config.data, IMAGE_SIZE, load_pixels=True)
    cache = TokenCache(config, len(dataset))
    loader = ProgressLoader(
        make_statistics_loader(dataset, cache, config),
        args.log_interval,
        enabled=rank == 0,
    )
    if rank == 0:
        print(
            f"computing codec statistics over {len(dataset)} images on {world_size} rank(s)",
            flush=True,
        )
    statistics = compute_pixel_token_statistics(checkpoint.model, loader, cache, device)

    if rank == 0:
        artifact = save_pixel_codec_artifact(
            args.output,
            checkpoint.model,
            statistics,
            metadata={
                "data_fingerprint": data_fingerprint(config.data, IMAGE_SIZE),
                "preprocessing": preprocessing_metadata(config.data, IMAGE_SIZE),
                "token_cache_fingerprint": cache.fingerprint,
                "source_checkpoint": str(checkpoint.source),
                "source_checkpoint_weights_sha256": checkpoint.weights_sha256,
            },
        )
        print(
            json.dumps(
                {
                    "artifact": str(artifact.resolve()),
                    "source_checkpoint": str(checkpoint.source),
                    "examples": len(dataset),
                    "pixel_token_mean": _summary(statistics.mean),
                    "pixel_token_std": _summary(statistics.std),
                    "pixel_rae_correlation": _summary(statistics.rae_correlation),
                },
                indent=2,
            ),
            flush=True,
        )
    if world_size > 1:
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
