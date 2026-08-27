from __future__ import annotations

import argparse
from pathlib import Path

from cascade_dit.config import load_config
from cascade_dit.distributed import initialize_distributed
from cascade_dit.preparation import prepare_representations, representation_manifest_fingerprint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and normalize the selected DINO semantic representation"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    config = load_config(arguments.config, arguments.override)
    rank, world_size, device = initialize_distributed()
    manifest = prepare_representations(
        config, rank=rank, world_size=world_size, device=device
    )
    if rank == 0:
        print(
            "representation preparation complete: "
            f"{representation_manifest_fingerprint(manifest)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
