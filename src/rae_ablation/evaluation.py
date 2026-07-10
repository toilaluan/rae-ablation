from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def evaluate_image_directories(
    generated: str | Path,
    real: str | Path,
    *,
    cuda: bool = True,
    inception_score: bool = True,
) -> dict[str, float]:
    import torch_fidelity

    real_path = Path(real).resolve()
    real_cache_name = "real-" + hashlib.sha256(str(real_path).encode()).hexdigest()[:16]
    cache_root = real_path.parent / ".fidelity-cache"
    metrics: dict[str, Any] = torch_fidelity.calculate_metrics(
        input1=str(generated),
        input2=str(real_path),
        input2_cache_name=real_cache_name,
        cache_root=str(cache_root),
        cache=True,
        cuda=cuda,
        fid=True,
        isc=inception_score,
        kid=False,
        verbose=False,
    )
    result = {"FID": float(metrics["frechet_inception_distance"])}
    if inception_score:
        result["Inception Score"] = float(metrics["inception_score_mean"])
        result["Inception Score std"] = float(metrics["inception_score_std"])
    return result


def save_evaluation(path: str | Path, metrics: dict[str, Any], plan_hash: str) -> None:
    Path(path).write_text(
        json.dumps(
            {"validation_plan_hash": plan_hash, "metrics": metrics}, indent=2, sort_keys=True
        )
        + "\n"
    )
