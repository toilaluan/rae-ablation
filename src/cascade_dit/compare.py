from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _summary(path: str | Path) -> tuple[str, dict[str, float]]:
    source = Path(path)
    if source.is_file():
        values = json.loads(source.read_text())
        plan_hash = values.get("sampling_plan_hash", values.get("validation_plan_hash"))
        return str(plan_hash), {name: float(value) for name, value in values["metrics"].items()}
    if not source.is_dir():
        raise FileNotFoundError(source)
    training_path = source / "training-metrics.json"
    if not training_path.is_file():
        raise FileNotFoundError(f"run training metrics are missing: {training_path}")
    training = json.loads(training_path.read_text())
    evaluations = sorted(source.glob("checkpoints/**/evaluation.json"))
    if not evaluations:
        raise FileNotFoundError(f"run has no completed evaluation: {source}")
    evaluation = json.loads(evaluations[-1].read_text())
    metrics = {
        f"training/{name}": float(value)
        for name, value in training.items()
        if isinstance(value, int | float) and name != "complete"
    }
    metrics.update(
        {f"evaluation/{name}": float(value) for name, value in evaluation["metrics"].items()}
    )
    return str(evaluation["sampling_plan_hash"]), metrics


def compare_runs(first: str | Path, second: str | Path) -> dict[str, Any]:
    left_hash, left_metrics = _summary(first)
    right_hash, right_metrics = _summary(second)
    if left_hash != right_hash:
        raise ValueError("runs use different deterministic sampling-plan hashes")
    metrics = sorted(set(left_metrics) & set(right_metrics))
    return {
        "sampling_plan_hash": left_hash,
        "first": str(first),
        "second": str(second),
        "metrics": {
            name: {
                "first": left_metrics[name],
                "second": right_metrics[name],
                "second_minus_first": right_metrics[name] - left_metrics[name],
            }
            for name in metrics
        },
    }
