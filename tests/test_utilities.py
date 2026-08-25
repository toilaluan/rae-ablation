from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cascade_dit.compare import compare_runs
from cascade_dit.evaluation import inception_score
from cascade_dit.feature_pca import feature_pca_rgb
from cascade_dit.rng import RNGStreams
from cascade_dit.validation import ValidationPlan, create_validation_plan


def test_feature_pca_is_deterministic_and_bounded() -> None:
    features = torch.randn(2, 12, 8, 8, generator=torch.Generator().manual_seed(7))
    first = feature_pca_rgb(features, 32)
    second = feature_pca_rgb(features, 32)
    assert first.shape == (2, 3, 32, 32)
    assert torch.equal(first, second)
    assert torch.all((first >= 0) & (first <= 1))


def test_inception_score_and_run_comparison() -> None:
    uniform = np.full((20, 4), 0.25)
    assert inception_score(uniform) == pytest.approx(1)
    classes = np.eye(4)[np.arange(20) % 4]
    assert inception_score(classes) == pytest.approx(2, rel=0.1)


def test_comparison_requires_matched_sampling_plan(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"sampling_plan_hash": "same", "metrics": {"FID": 2}}))
    second.write_text(json.dumps({"sampling_plan_hash": "same", "metrics": {"FID": 1}}))
    comparison = compare_runs(first, second)
    assert comparison["metrics"]["FID"]["second_minus_first"] == -1
    second.write_text(json.dumps({"sampling_plan_hash": "different", "metrics": {"FID": 1}}))
    with pytest.raises(ValueError, match="sampling-plan"):
        compare_runs(first, second)


def test_rng_checkpoint_includes_global_dropout_state() -> None:
    streams = RNGStreams.create(3)
    torch.manual_seed(99)
    state = streams.state_dict()
    expected = torch.rand(4)
    streams.load_state_dict(state)
    assert torch.equal(torch.rand(4), expected)


def test_validation_plan_is_immutable_and_uses_a_class_balanced_prefix(tmp_path) -> None:
    plan = create_validation_plan(
        final_samples=20, periodic_samples=10, num_classes=5, seed=7
    )
    labels, sample_ids, seeds = plan.prefix(10)
    assert torch.bincount(labels, minlength=5).tolist() == [2, 2, 2, 2, 2]
    assert sample_ids.tolist() == list(range(10))
    assert seeds.shape == (10,)

    path = tmp_path / "plan.pt"
    plan.save(path)
    assert ValidationPlan.load(path).plan_hash == plan.plan_hash
    different = create_validation_plan(
        final_samples=20, periodic_samples=10, num_classes=5, seed=8
    )
    with pytest.raises(ValueError, match="immutable"):
        different.save(path)
