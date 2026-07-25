from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from jit_reg.config import ExperimentConfig
from jit_reg.feature_cache import (
    CACHE_SCHEMA_VERSION,
    FeatureCache,
    FeatureComponentCache,
    FeatureComponentCacheWriter,
    component_fingerprint,
    component_path,
    component_spec,
)


def cache_config(tmp_path: Path) -> ExperimentConfig:
    config = ExperimentConfig()
    config.feature_cache.enabled = True
    config.feature_cache.directory = str(tmp_path)
    config.model.image_size = 16
    config.model.patch_size = 8
    config.model.features.encoder.feature_dim = 8
    config.model.features.reg.enabled = True
    config.model.features.repa.enabled = True
    config.model.features.repa.block = 1
    config.model.features.repa.token_scope = "image_and_reg"
    config.train.horizontal_flip_probability = 0
    return config


def fingerprints(config: ExperimentConfig) -> tuple[str, str]:
    return (
        component_fingerprint(component_spec(config, 4, "reg")),
        component_fingerprint(component_spec(config, 4, "repa")),
    )


def test_component_fingerprints_are_minimal_and_independent(tmp_path: Path) -> None:
    config = cache_config(tmp_path)
    original = fingerprints(config)

    irrelevant = copy.deepcopy(config)
    irrelevant.model.features.reg.head.kind = "wide_mlp"
    irrelevant.model.features.reg.loss_weight = 99
    irrelevant.model.features.reg.timestep.sample_ratio = 0.25
    irrelevant.model.features.repa.block = 2
    irrelevant.model.features.repa.loss_weight = 99
    irrelevant.train.optimizer.lr = 1
    irrelevant.compile.enabled = False
    irrelevant.sampling.sampling_steps = 3
    irrelevant.evaluation.batch_size = 1
    assert fingerprints(irrelevant) == original

    token_change = copy.deepcopy(config)
    token_change.model.features.reg.token_mode = "cls_spatial"
    changed = fingerprints(token_change)
    assert changed[0] != original[0]
    assert changed[1] == original[1]

    grid_change = copy.deepcopy(config)
    grid_change.model.patch_size = 4
    changed = fingerprints(grid_change)
    assert changed[0] == original[0]
    assert changed[1] != original[1]

    encoder_change = copy.deepcopy(config)
    encoder_change.model.features.encoder.model_name = "other"
    changed = fingerprints(encoder_change)
    assert changed[0] != original[0] and changed[1] != original[1]


def test_environment_is_diagnostic_not_fingerprinted(tmp_path: Path) -> None:
    spec = component_spec(cache_config(tmp_path), 4, "reg")
    changed = copy.deepcopy(spec)
    changed["environment"]["torch"] = "different"
    assert component_fingerprint(changed) == component_fingerprint(spec)


@pytest.mark.parametrize("component", ["reg", "repa"])
def test_component_writer_is_resumable(
    tmp_path: Path, component: str
) -> None:
    config = cache_config(tmp_path)
    spec = component_spec(config, 4, component)  # type: ignore[arg-type]
    path = component_path(config, 4, component)  # type: ignore[arg-type]
    assert not FeatureComponentCacheWriter.initialize(path, spec)
    writer = FeatureComponentCacheWriter(path, spec)
    values = torch.arange(
        2 * writer.shape[1] * writer.shape[2], dtype=torch.float32
    ).reshape(2, writer.shape[1], writer.shape[2])
    writer.write(torch.tensor([0, 2]), values)
    assert np.array_equal(writer.pending_indices(), np.array([1, 3]))
    with pytest.raises(ValueError, match="incomplete"):
        writer.finalize()
    remaining = torch.zeros(2, writer.shape[1], writer.shape[2])
    writer.write(torch.tensor([1, 3]), remaining)
    writer.finalize()
    assert FeatureComponentCacheWriter.initialize(path, spec)
    cache = FeatureComponentCache(config, 4, component)  # type: ignore[arg-type]
    assert torch.equal(cache.lookup(torch.tensor([0, 2])).float(), values)


def test_combined_cache_reuses_independent_components(tmp_path: Path) -> None:
    config = cache_config(tmp_path)
    for component in ("reg", "repa"):
        spec = component_spec(config, 3, component)
        path = component_path(config, 3, component)
        FeatureComponentCacheWriter.initialize(path, spec)
        writer = FeatureComponentCacheWriter(path, spec)
        writer.write(torch.arange(3), torch.ones(writer.shape))
        writer.finalize()
    cache = FeatureCache(config, 3)
    targets = cache.lookup(torch.tensor([2, 0]))
    assert targets.reg is not None and targets.reg.shape == (2, 1, 8)
    assert targets.repa is not None and targets.repa.shape == (2, 4, 8)
    assert set(cache.fingerprints) == {"reg", "repa"}


def test_incompatible_and_old_manifests_are_rejected(tmp_path: Path) -> None:
    config = cache_config(tmp_path)
    spec = component_spec(config, 2, "reg")
    path = component_path(config, 2, "reg")
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps({"schema_version": CACHE_SCHEMA_VERSION - 1})
    )
    with pytest.raises(ValueError, match="old caches must be rebuilt"):
        FeatureComponentCache(config, 2, "reg")

    other = tmp_path / "other"
    FeatureComponentCacheWriter.initialize(other, spec)
    changed = copy.deepcopy(spec)
    changed["fingerprint_inputs"]["target"]["feature_schema"] = "incompatible"
    with pytest.raises(ValueError, match="incompatible"):
        FeatureComponentCacheWriter.initialize(other, changed)
