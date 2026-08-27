from __future__ import annotations

import argparse
from pathlib import Path

from cascade_dit.config import load_config
from cascade_dit.distributed import initialize_distributed
from cascade_dit.pixel_preparation import prepare_pixel_pretraining_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare cached or online-statistics DINO inputs for pixel pretraining"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    config = load_config(arguments.config, arguments.override)
    rank, world_size, device = initialize_distributed()
    prepare_pixel_pretraining_inputs(
        config, rank=rank, world_size=world_size, device=device
    )
    if rank == 0:
        print("pixel decoder semantic inputs are ready", flush=True)


if __name__ == "__main__":
    main()
