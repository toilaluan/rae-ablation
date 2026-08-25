from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .checkpoint import CHECKPOINT_SCHEMA_VERSION
from .config import ExperimentConfig, load_checkpoint_config


@dataclass(frozen=True)
class CheckpointBundle:
    directory: Path
    config: ExperimentConfig
    weights: Path
    metadata: dict[str, object]


def resolve_checkpoint_directory(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {candidate}")
    if not (candidate / "config.json").is_file() and (candidate / "last").is_dir():
        candidate = candidate / "last"
    resolved = candidate.resolve()
    if not (resolved / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint config is missing: {resolved / 'config.json'}")
    return resolved


def load_checkpoint_bundle(path: str | Path, *, require_rgb_complete: bool = False) -> CheckpointBundle:
    directory = resolve_checkpoint_directory(path)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError("checkpoint predates the adjacent-flow DDT schema")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema is incompatible")
    config = load_checkpoint_config(directory / "config.json")
    if require_rgb_complete and (
        metadata.get("stage") != config.stage_names[-1]
        or not metadata.get("training_complete")
    ):
        raise ValueError("sampling requires a completed RGB-stage checkpoint")
    weights = directory / "ema.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint EMA weights are missing: {weights}")
    return CheckpointBundle(directory, config, weights, metadata)
