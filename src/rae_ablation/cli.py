from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .cache import build_cache, verify_cache
from .compare import compare_runs
from .config import load_config
from .distributed import initialize_distributed


def _base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=Path("configs/canonical.yaml"))
    parser.add_argument(
        "overrides", nargs="*", help="OmegaConf dot-list overrides, e.g. cache.workers=0"
    )
    return parser


def cache_main() -> None:
    parser = _base_parser("Cache pinned ImageNet RAE latents")
    parser.add_argument("--codec", choices=("rae", "rae_compact", "both"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
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


def compare_main() -> None:
    parser = argparse.ArgumentParser(description="Compare matched RAE ablation results")
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    comparison: dict[str, Any] = compare_runs(args.first, args.second)
    rendered = json.dumps(comparison, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)
