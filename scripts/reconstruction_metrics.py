from __future__ import annotations

import argparse
import json
from pathlib import Path

from rae_ablation.distributed import initialize_distributed
from rae_ablation.reconstruction import (
    ReconstructionParameters,
    build_reconstruction_report,
    load_and_evaluate_reconstructions,
    save_reconstruction_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare RAE and compact RAE reconstruction metrics on ImageNet validation"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    parameters = ReconstructionParameters(
        batch_size=args.batch_size,
        workers=args.workers,
        max_samples=args.max_samples,
        log_interval=args.log_interval,
    )
    parameters.validate()
    rank, world_size, device = initialize_distributed()
    if rank == 0:
        print(f"initializing reconstruction evaluation on {device}, world_size={world_size}")
    metrics, sample_count = load_and_evaluate_reconstructions(
        parameters,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    if rank == 0:
        report = build_reconstruction_report(
            metrics,
            sample_count=sample_count,
            parameters=parameters,
            world_size=world_size,
            device=device,
        )
        save_reconstruction_report(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
