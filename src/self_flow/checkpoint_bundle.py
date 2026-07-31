from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .checkpoint import CHECKPOINT_SCHEMA_VERSION
from .config import ExperimentConfig, load_checkpoint_config
from .validation import ValidationPlan, create_validation_plan


@dataclass(frozen=True)
class CheckpointBundle:
    directory: Path
    config: ExperimentConfig
    weights: Path
    plan: ValidationPlan
    validation_plan_hash: str


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


def _expected_plan_hash(directory: Path) -> str:
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError("checkpoint predates schema v11; old checkpoints are not supported")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema is incompatible; old checkpoints cannot be migrated")
    value = metadata.get("validation_plan_hash")
    if not isinstance(value, str) or not value:
        raise ValueError("checkpoint has no validation-plan hash")
    return value


def _plan_candidates(directory: Path, config: ExperimentConfig) -> list[Path]:
    configured = Path(config.validation.plan_path)
    candidates = [directory / "validation-plan.pt"]
    if configured.is_absolute():
        candidates.append(configured)
    else:
        candidates.extend(
            [
                configured,
                directory / configured,
                directory.parent.parent / configured.name,
            ]
        )
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(candidate)
    return unique


def _load_plan(
    directory: Path,
    config: ExperimentConfig,
    expected_hash: str,
) -> ValidationPlan:
    mismatches: list[Path] = []
    for path in _plan_candidates(directory, config):
        if not path.is_file():
            continue
        plan = ValidationPlan.load(path)
        if plan.plan_hash == expected_hash:
            return plan
        mismatches.append(path)
    reconstructed = create_validation_plan(
        config.validation.final_samples,
        config.validation.periodic_samples,
        config.model.num_classes,
        config.train.seed + 7,
    )
    if reconstructed.plan_hash == expected_hash:
        return reconstructed
    suffix = f"; mismatched candidates: {mismatches}" if mismatches else ""
    raise ValueError("could not locate or reconstruct the checkpoint validation plan" + suffix)


def load_checkpoint_bundle(path: str | Path) -> CheckpointBundle:
    directory = resolve_checkpoint_directory(path)
    config = load_checkpoint_config(directory / "config.json")
    expected_hash = _expected_plan_hash(directory)
    plan = _load_plan(directory, config, expected_hash)
    weights = directory / (
        "ema.safetensors" if config.objective.kind == "ema" else "model.safetensors"
    )
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint sampling weights are missing: {weights}")
    return CheckpointBundle(directory, config, weights, plan, expected_hash)
