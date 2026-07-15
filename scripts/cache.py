from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rae_ablation.cache import build_cache, verify_cache
from rae_ablation.config import load_config
from rae_ablation.distributed import initialize_distributed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache pinned ImageNet RAE latents")
    parser.add_argument("--config", type=Path, default=Path("configs/canonical.yaml"))
    parser.add_argument("--codec", choices=("rae", "rae_compact", "both"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "overrides", nargs="*", help="OmegaConf dot-list overrides, e.g. cache.workers=0"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.overrides)
    if args.codec:
        config.cache.codec = args.codec
    if args.output:
        config.cache.directory = str(args.output)
    if args.max_samples is not None:
        config.cache.max_samples = args.max_samples
    if args.verify_only:
        completed = verify_cache(config.cache.directory)
        print(f"verified {len(completed)} shards")
        return
    initialize_distributed()
    manifest = build_cache(config.cache)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"cache fingerprint: {manifest.fingerprint}")


if __name__ == "__main__":
    main()
