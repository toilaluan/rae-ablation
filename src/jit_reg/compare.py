from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_runs(first: str | Path, second: str | Path) -> dict[str, Any]:
    left = json.loads(Path(first).read_text())
    right = json.loads(Path(second).read_text())
    if left["validation_plan_hash"] != right["validation_plan_hash"]:
        raise ValueError("runs use different validation-plan hashes")
    left_protocol = dict(left.get("protocol", {}))
    right_protocol = dict(right.get("protocol", {}))
    left_endpoint = left_protocol.pop("endpoint", None)
    right_endpoint = right_protocol.pop("endpoint", None)
    if left_protocol != right_protocol:
        raise ValueError("runs use different evaluation protocols")
    metrics = sorted(set(left["metrics"]) & set(right["metrics"]))
    return {
        "validation_plan_hash": left["validation_plan_hash"],
        "first": str(first),
        "second": str(second),
        "first_endpoint": left_endpoint,
        "second_endpoint": right_endpoint,
        "protocol": left_protocol,
        "metrics": {
            name: {
                "first": left["metrics"][name],
                "second": right["metrics"][name],
                "second_minus_first": right["metrics"][name] - left["metrics"][name],
            }
            for name in metrics
        },
    }
