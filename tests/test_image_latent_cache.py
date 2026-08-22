from __future__ import annotations

import json

import pytest
import torch

from cascade_dit.config import ExperimentConfig
from cascade_dit.image_latent_cache import (
    AsyncImageLatentCacheWriter,
    ImageLatentCache,
    ImageLatentCacheWriter,
    image_latent_cache_fingerprint,
    image_latent_cache_path,
    image_latent_cache_spec,
    required_image_latent_cache_bytes,
)


def test_image_latent_cache_roundtrip_resume_and_exact_size(tmp_path) -> None:
    config = ExperimentConfig()
    config.data.max_samples = 3
    config.image_latents.cache.directory = str(tmp_path)
    spec = image_latent_cache_spec(config, 3, "mage-digest")
    path = image_latent_cache_path(config, spec)
    assert required_image_latent_cache_bytes(spec) == 3 * 128 * 16 * 16 * 2 + 3
    assert ImageLatentCacheWriter.initialize(path, spec) is False
    writer = ImageLatentCacheWriter(path, spec)
    values = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1, 1).expand(3, 128, 16, 16)
    writer.write(torch.tensor([0, 2]), values[[0, 2]])
    assert writer.pending_indices().tolist() == [1]
    resumed = ImageLatentCacheWriter(path, spec)
    pipeline = AsyncImageLatentCacheWriter(resumed, batch_size=1, device="cpu")
    pipeline.submit(torch.tensor([1]), values[[1]])
    pipeline.close()
    resumed.finalize()
    cache = ImageLatentCache(config, 3, "mage-digest")
    restored = cache.lookup(torch.tensor([2, 0]))
    assert restored.dtype == torch.bfloat16
    assert torch.equal(restored.float(), values[[2, 0]])
    assert json.loads((path / "manifest.json").read_text())["complete"] is True


def test_image_cache_fingerprint_ignores_ddt_model_and_rejects_corruption(tmp_path) -> None:
    config = ExperimentConfig()
    config.image_latents.cache.directory = str(tmp_path)
    first = image_latent_cache_spec(config, 2, "mage")
    config.model.architecture = "latent"
    config.model.hidden_size = [32, 48]
    config.model.depth = [2, 1]
    config.model.num_heads = [4, 3]
    config.model.image_target = "pixels"
    second = image_latent_cache_spec(config, 2, "mage")
    assert image_latent_cache_fingerprint(first) == image_latent_cache_fingerprint(second)
    path = image_latent_cache_path(config, first)
    ImageLatentCacheWriter.initialize(path, first)
    (path / "completion.u8").write_bytes(b"\x02\x00")
    with pytest.raises(ValueError, match="corrupt"):
        ImageLatentCacheWriter(path, first)
