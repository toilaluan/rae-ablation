from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jit_reg.config import ExperimentConfig
from jit_reg.feature_cache import (
    AsyncFeatureCacheWriter,
    FeatureCache,
    FeatureComponentCacheWriter,
    component_path,
    component_spec,
)
from jit_reg.model.sit import SiTOutput
from jit_reg.repa import resize_spatial_features
from jit_reg.sample_runtime import SamplingParameters, sampling_output_name
from jit_reg.sampling import ode_sample


class ConstantXModel(nn.Module):
    def __init__(self, value: float = 0.25) -> None:
        super().__init__()
        self.value = value
        self.config = SimpleNamespace(num_classes=3)

    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        labels: torch.Tensor,
        *,
        return_repa: bool = False,
    ) -> SiTOutput:
        del timestep, labels, return_repa
        return SiTOutput(torch.full_like(noisy, self.value))


@pytest.mark.parametrize("solver", ["euler", "heun"])
def test_sampler_converts_x_prediction_to_velocity(solver: str) -> None:
    generated = ode_sample(
        ConstantXModel(),
        torch.randn(2, 1, 2, 2),
        torch.tensor([0, 1]),
        steps=4,
        solver=solver,
        guidance_scale=1,
    )
    assert torch.allclose(generated, torch.full_like(generated, 0.25), atol=1e-6)


def test_sampling_identity_has_no_retired_reg_fields() -> None:
    parameters = SamplingParameters(sampling_steps=4, guidance_scale=2)
    name = sampling_output_name(1000, parameters)
    assert "steps4" in name and "cfg2p0" in name
    assert "reg" not in name


def test_repa_resize_normalizes_target_tokens() -> None:
    spatial = torch.randn(2, 16, 8)
    resized = resize_spatial_features(spatial, 2)
    assert resized.shape == (2, 4, 8)
    assert torch.allclose(resized.norm(dim=-1), torch.ones(2, 4), atol=1e-6)


def test_repa_cache_roundtrip_and_schema_v4_path(tmp_path) -> None:
    config = ExperimentConfig()
    config.model.image_size = 8
    config.model.patch_size = 2
    config.repa.enabled = True
    config.repa.cache.enabled = True
    config.repa.cache.directory = str(tmp_path)
    config.repa.encoder.feature_dim = 8
    spec = component_spec(config, 3, "repa")
    inputs = spec["fingerprint_inputs"]
    assert inputs["schema_version"] == 4
    assert inputs["component"] == "repa"
    path = component_path(config, 3, "repa")
    FeatureComponentCacheWriter.initialize(path, spec)
    writer = FeatureComponentCacheWriter(path, spec)
    values = torch.randn(3, 16, 8)
    with AsyncFeatureCacheWriter({"repa": writer}, 3, "cpu") as pipeline:
        pipeline.submit(torch.arange(3), values)
        pipeline.drain()
    writer.finalize()
    cache = FeatureCache(config, 3)
    actual = cache.lookup(torch.tensor([2, 0]))
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, values[[2, 0]].bfloat16())
