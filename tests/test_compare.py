from __future__ import annotations

import json

import pytest

from jit_reg.compare import compare_runs
from jit_reg.validation import ValidationPlan, create_validation_plan


def _write(path, endpoint: str, *, steps: int = 40) -> None:
    path.write_text(
        json.dumps(
            {
                "validation_plan_hash": "paired-plan",
                "protocol": {
                    "endpoint": endpoint,
                    "sample_count": 10,
                    "steps": steps,
                },
                "metrics": {"validation/lpips_40step": 0.2 if endpoint == "target" else 0.1},
            }
        )
    )


def test_comparison_allows_only_the_endpoint_to_change(tmp_path) -> None:
    target = tmp_path / "target.json"
    pixel = tmp_path / "pixel.json"
    _write(target, "target")
    _write(pixel, "pixel")
    result = compare_runs(target, pixel)
    assert result["first_endpoint"] == "target"
    assert result["second_endpoint"] == "pixel"
    assert result["metrics"]["validation/lpips_40step"]["second_minus_first"] == pytest.approx(-0.1)

    _write(pixel, "pixel", steps=1)
    with pytest.raises(ValueError, match="different evaluation protocols"):
        compare_runs(target, pixel)


def test_validation_plan_is_deterministic_and_hash_checked(tmp_path) -> None:
    first = create_validation_plan(8, 4, seed=3, dataset_size=20)
    second = create_validation_plan(8, 4, seed=3, dataset_size=20)
    assert first.plan_hash == second.plan_hash
    assert first.indices.unique().numel() == 8
    path = tmp_path / "plan.pt"
    first.save(path)
    assert ValidationPlan.load(path).plan_hash == first.plan_hash
    with pytest.raises(ValueError, match="outside"):
        first.prefix(9)
