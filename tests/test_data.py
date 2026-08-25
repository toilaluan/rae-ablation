from __future__ import annotations

from collections import Counter

import pytest
import torch

from cascade_dit.config import (
    ExperimentConfig,
    RepresentationSourceConfig,
    RepresentationStageConfig,
)
from cascade_dit.data import (
    ADJACENT_MIX,
    ALL_LEVELS_MIX,
    PURE_CURRENT,
    PURE_NEXT,
    sample_representation_weights,
)


def test_default_mixture_probabilities_and_determinism() -> None:
    config = ExperimentConfig()
    choices = [
        sample_representation_weights(config, 0, 3, index) for index in range(20_000)
    ]
    counts = Counter(choice.mode for choice in choices)
    assert counts[PURE_CURRENT] / len(choices) == pytest.approx(0.85, abs=0.015)
    assert counts[ADJACENT_MIX] / len(choices) == pytest.approx(0.10, abs=0.01)
    assert counts[PURE_NEXT] / len(choices) == pytest.approx(0.025, abs=0.006)
    assert counts[ALL_LEVELS_MIX] / len(choices) == pytest.approx(0.025, abs=0.006)
    repeated = sample_representation_weights(config, 0, 3, 17)
    assert repeated.mode == choices[17].mode
    assert torch.equal(repeated.weights, choices[17].weights)


def test_adjacent_and_uniform_simplex_weights_are_well_formed() -> None:
    config = ExperimentConfig()
    adjacent = all_levels = None
    for index in range(10_000):
        choice = sample_representation_weights(config, 1, 0, index)
        if choice.mode == ADJACENT_MIX and adjacent is None:
            adjacent = choice.weights
        if choice.mode == ALL_LEVELS_MIX and all_levels is None:
            all_levels = choice.weights
        if adjacent is not None and all_levels is not None:
            break
    assert adjacent is not None and all_levels is not None
    assert torch.count_nonzero(adjacent) == 2
    assert torch.count_nonzero(adjacent[[0, 3]]) == 0
    assert torch.all(all_levels > 0)
    assert torch.isclose(adjacent.sum(), torch.tensor(1.0))
    assert torch.isclose(all_levels.sum(), torch.tensor(1.0))


def test_dynamic_level_count_and_final_stage_are_supported() -> None:
    config = ExperimentConfig()
    config.representations.stages = [
        RepresentationStageConfig("deep", 1, RepresentationSourceConfig("dino", 12)),
        RepresentationStageConfig("shallow", 1, RepresentationSourceConfig("dino", 3)),
        RepresentationStageConfig("rgb", 1, RepresentationSourceConfig("rgb")),
    ]
    config.validate()
    all_levels = None
    for index in range(20_000):
        choice = sample_representation_weights(config, 0, 1, index)
        if choice.mode == ALL_LEVELS_MIX:
            all_levels = choice
            break
    assert all_levels is not None and all_levels.weights.shape == (3,)
    final = sample_representation_weights(config, 2, 1, 0)
    baseline = sample_representation_weights(
        config, 0, 1, 0, pixel_baseline=True
    )
    assert final.mode == baseline.mode == PURE_CURRENT
    assert torch.equal(final.weights, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(baseline.weights, final.weights)
