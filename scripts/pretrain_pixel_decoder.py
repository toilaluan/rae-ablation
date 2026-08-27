from __future__ import annotations

import argparse
import json
from pathlib import Path

from cascade_dit.config import load_config
from cascade_dit.distributed import initialize_distributed
from cascade_dit.pixel_training import PixelPretrainingRunner
from cascade_dit.wandb_utils import RunLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain the semantic-to-RGB pixel decoder"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path, default=Path("artifacts/pixel-decoder"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    config = load_config(arguments.config, arguments.override)
    config.experiment.output_directory = str(arguments.output)
    rank, world_size, device = initialize_distributed()
    logger = RunLogger(config.wandb, config.as_dict(), rank == 0)

    def log(metrics: dict[str, float | int | str]) -> None:
        step = int(metrics["step"])
        validation = any(key.startswith("validation/") for key in metrics)
        prefix = "pixel-validation" if validation else "pixel-pretrain"
        logger.log(
            {
                key if key.startswith("validation/") else f"{prefix}/{key}": value
                for key, value in metrics.items()
            },
            step=step,
        )
        if rank == 0 and step % arguments.log_interval == 0:
            print(json.dumps(metrics, sort_keys=True), flush=True)

    runner = PixelPretrainingRunner(
        config,
        rank=rank,
        world_size=world_size,
        device=device,
        log=log,
        log_image=logger.log_grid,
        resume=arguments.resume,
    )
    try:
        runner.run(max_steps=arguments.max_steps)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
