from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from jit_reg.config import ExperimentConfig
from jit_reg.feature_cache import (
    CACHE_SCHEMA_VERSION,
    AsyncFeatureCacheWriter,
    FeatureCache,
    FeatureComponentCache,
    FeatureComponentCacheWriter,
    component_fingerprint,
    component_path,
    component_spec,
    preflight_cache_capacity,
    required_cache_bytes,
)
from jit_reg.reg import FeatureTargets
from jit_reg.rng import RNGStreams
from jit_reg.training import TrainBatch, make_train_loader


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


def write_active_caches(config: ExperimentConfig, dataset_size: int) -> None:
    components = []
    if config.model.features.reg.enabled:
        components.append("reg")
    if config.model.features.repa.enabled:
        components.append("repa")
    for component in components:
        spec = component_spec(config, dataset_size, component)
        path = component_path(config, dataset_size, component)
        FeatureComponentCacheWriter.initialize(path, spec)
        writer = FeatureComponentCacheWriter(path, spec)
        values = (
            torch.arange(dataset_size, dtype=torch.float32)
            .reshape(dataset_size, 1, 1)
            .expand(writer.shape)
        )
        writer.write(torch.arange(dataset_size), values)
        writer.finalize()


def test_component_fingerprints_are_minimal_and_independent(tmp_path: Path) -> None:
    config = cache_config(tmp_path)
    original = fingerprints(config)

    irrelevant = copy.deepcopy(config)
    irrelevant.model.features.reg.head.kind = "wide_mlp"
    irrelevant.model.features.reg.loss_weight = 99
    irrelevant.model.features.reg.target_std = 99
    irrelevant.model.features.reg.timestep.sample_ratio = 0.25
    irrelevant.model.features.repa.block = 2
    irrelevant.model.features.repa.loss_weight = 99
    irrelevant.train.optimizer.lr = 1
    irrelevant.compile.enabled = False
    irrelevant.sampling.sampling_steps = 3
    irrelevant.evaluation.batch_size = 1
    assert fingerprints(irrelevant) == original

    changed_pool = copy.deepcopy(config)
    changed_pool.model.features.reg.pool_size = 2
    changed = fingerprints(changed_pool)
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
    detached = cache.lookup(torch.tensor([0]))
    detached.zero_()
    assert torch.equal(cache.lookup(torch.tensor([0])).float(), values[:1])


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
    assert targets.reg is not None and targets.reg.shape == (2, 16, 8)
    assert targets.repa is not None and targets.repa.shape == (2, 4, 8)
    assert set(cache.fingerprints) == {"reg", "repa"}


@pytest.mark.parametrize(
    "components", [("reg",), ("repa",), ("reg", "repa")]
)
def test_cache_pipeline_cpu_fallback_writes_partial_batches_in_order(
    tmp_path: Path, components: tuple[str, ...]
) -> None:
    config = cache_config(tmp_path)
    writers = {}
    for component in components:
        spec = component_spec(config, 3, component)  # type: ignore[arg-type]
        path = component_path(config, 3, component)  # type: ignore[arg-type]
        FeatureComponentCacheWriter.initialize(path, spec)
        writers[component] = FeatureComponentCacheWriter(path, spec)
    pipeline = AsyncFeatureCacheWriter(writers, batch_size=2, device="cpu")  # type: ignore[arg-type]
    assert not pipeline.asynchronous
    for indices in (torch.tensor([2, 0]), torch.tensor([1])):
        count = indices.numel()
        reg = (
            indices.float()
            .reshape(count, 1, 1)
            .expand(count, writers["reg"].shape[1], writers["reg"].shape[2])
            if "reg" in writers
            else None
        )
        repa = (
            (indices.float() + 10)
            .reshape(count, 1, 1)
            .expand(count, writers["repa"].shape[1], writers["repa"].shape[2])
            if "repa" in writers
            else None
        )
        pipeline.submit(indices, FeatureTargets(reg, repa))
    pipeline.close()
    metrics = pipeline.metrics()
    assert metrics.examples == 3 and metrics.batches == 2
    assert metrics.buffer_backpressure_seconds == 0
    for component, writer in writers.items():
        writer.finalize()
        cache = FeatureComponentCache(config, 3, component)  # type: ignore[arg-type]
        expected = torch.arange(3) + (10 if component == "repa" else 0)
        assert torch.equal(
            cache.lookup(torch.arange(3))[:, 0, 0].float(), expected.float()
        )


def test_cache_capacity_preflight_counts_unallocated_bytes(tmp_path: Path) -> None:
    config = cache_config(tmp_path)
    specs = {
        component: component_spec(config, 3, component)
        for component in ("reg", "repa")
    }
    paths = {
        component: component_path(config, 3, component)
        for component in ("reg", "repa")
    }
    expected = sum(required_cache_bytes(spec) for spec in specs.values())
    assert preflight_cache_capacity(tmp_path, specs, paths) == expected
    FeatureComponentCacheWriter.initialize(paths["reg"], specs["reg"])
    remaining = preflight_cache_capacity(tmp_path, specs, paths)
    assert required_cache_bytes(specs["repa"]) <= remaining < expected


@pytest.mark.parametrize(
    ("workers", "reg_enabled", "repa_enabled"),
    [
        (0, True, False),
        (0, False, True),
        (0, True, True),
        (2, True, True),
    ],
)
def test_train_loader_prefetches_index_aligned_cached_targets(
    tmp_path: Path,
    workers: int,
    reg_enabled: bool,
    repa_enabled: bool,
) -> None:
    config = cache_config(tmp_path)
    config.model.features.reg.enabled = reg_enabled
    config.model.features.repa.enabled = repa_enabled
    if not reg_enabled:
        config.model.features.repa.token_scope = "image"
    config.data.workers = workers
    config.data.prefetch_factor = 1
    config.train.micro_batch_size = 2
    write_active_caches(config, 4)
    dataset = TensorDataset(
        torch.randint(0, 256, (4, 3, 16, 16), dtype=torch.uint8),
        torch.arange(4),
        torch.arange(4),
    )

    loader, _sampler = make_train_loader(dataset, config, RNGStreams.create(7))
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert isinstance(batch, TrainBatch)
        assert batch.pixels.dtype == torch.uint8
        assert batch.cached_targets is not None
        expected = batch.indices.to(torch.float32)
        if reg_enabled:
            assert batch.cached_targets.reg is not None
            assert batch.cached_targets.reg.dtype == torch.bfloat16
            assert torch.equal(batch.cached_targets.reg[:, 0, 0].float(), expected)
        else:
            assert batch.cached_targets.reg is None
        if repa_enabled:
            assert batch.cached_targets.repa is not None
            assert batch.cached_targets.repa.dtype == torch.bfloat16
            assert torch.equal(batch.cached_targets.repa[:, 0, 0].float(), expected)
        else:
            assert batch.cached_targets.repa is None


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
