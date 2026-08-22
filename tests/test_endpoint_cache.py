from __future__ import annotations

import json

import pytest
import torch

from jit_reg.config import ExperimentConfig
from jit_reg.endpoint_cache import (
    AsyncEndpointCacheWriter,
    EndpointCache,
    EndpointCacheWriter,
    endpoint_cache_fingerprint,
    endpoint_cache_path,
    endpoint_cache_spec,
    required_endpoint_cache_bytes,
)


def test_endpoint_cache_roundtrip_resume_and_size(tmp_path) -> None:
    config = ExperimentConfig()
    config.data.max_samples = 3
    config.validation.periodic_samples = 2
    config.validation.final_samples = 3
    config.vision.cache.directory = str(tmp_path)
    spec = endpoint_cache_spec(config, 3, "statistics-digest")
    path = endpoint_cache_path(config, spec)
    spatial_values = 3 * 768 * 8 * 8
    assert required_endpoint_cache_bytes(spec) == 2 * (2 * spatial_values) + 3
    assert EndpointCacheWriter.initialize(path, spec) is False
    writer = EndpointCacheWriter(path, spec)
    target = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1, 1).expand(3, 768, 8, 8)
    semantic = target + 10
    writer.write(
        torch.tensor([0, 2]),
        target[[0, 2]],
        semantic[[0, 2]],
    )
    assert writer.pending_indices().tolist() == [1]

    resumed = EndpointCacheWriter(path, spec)
    pipeline = AsyncEndpointCacheWriter(resumed, batch_size=1, device="cpu")
    pipeline.submit(torch.tensor([1]), target[[1]], semantic[[1]])
    pipeline.close()
    resumed.finalize()

    cache = EndpointCache(config, 3, "statistics-digest")
    restored_target, restored_semantic = cache.lookup(torch.tensor([2, 0]))
    assert restored_target.dtype == restored_semantic.dtype == torch.bfloat16
    assert torch.equal(restored_target.float(), target[[2, 0]])
    assert torch.equal(restored_semantic.float(), semantic[[2, 0]])
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["complete"] is True


def test_endpoint_cache_fingerprint_ignores_flow_route_but_tracks_sources(tmp_path) -> None:
    config = ExperimentConfig()
    config.vision.cache.directory = str(tmp_path)
    first = endpoint_cache_spec(config, 2, "statistics")
    config.flow.endpoint = "pixel"
    second = endpoint_cache_spec(config, 2, "statistics")
    assert endpoint_cache_fingerprint(first) == endpoint_cache_fingerprint(second)
    config.data.revision = "different"
    third = endpoint_cache_spec(config, 2, "statistics")
    assert endpoint_cache_fingerprint(first) != endpoint_cache_fingerprint(third)


def test_incomplete_endpoint_cache_is_rejected(tmp_path) -> None:
    config = ExperimentConfig()
    config.vision.cache.directory = str(tmp_path)
    spec = endpoint_cache_spec(config, 1, "statistics")
    EndpointCacheWriter.initialize(endpoint_cache_path(config, spec), spec)
    with pytest.raises(ValueError, match="incomplete"):
        EndpointCache(config, 1, "statistics")


def test_legacy_endpoint_cache_schema_is_rejected(tmp_path) -> None:
    config = ExperimentConfig()
    config.vision.cache.directory = str(tmp_path)
    spec = endpoint_cache_spec(config, 2, "statistics")
    path = endpoint_cache_path(config, spec)
    EndpointCacheWriter.initialize(path, spec)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incompatible schema"):
        EndpointCacheWriter(path, spec)
