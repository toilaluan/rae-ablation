from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jit_reg.config import ExperimentConfig
from jit_reg.data import ImageNetDataset
from jit_reg.model import DDTOutput
from jit_reg.rae import load_rae_codec
from jit_reg.sampling import ode_sample
from jit_reg.token_cache import (
    TokenCache,
    TokenCacheWriter,
    denormalize_tokens,
    normalize_tokens,
    required_token_cache_bytes,
    token_cache_path,
    token_cache_spec,
)


class GuidanceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_classes=2)

    def forward(self, noisy: torch.Tensor, timestep: torch.Tensor, labels: torch.Tensor) -> DDTOutput:
        del timestep
        shape = (labels.shape[0], *((1,) * (noisy.ndim - 1)))
        conditional = labels.reshape(shape).to(noisy.dtype)
        full = torch.ones_like(noisy) * (2 + conditional)
        base = torch.ones_like(noisy) * conditional
        return DDTOutput(full, base)


def test_pure_internal_guidance_and_combined_cfg() -> None:
    model = GuidanceModel()
    noise = torch.zeros(1, 1, 1, 1)
    pure = ode_sample(model, noise, torch.tensor([1]), steps=1, timestep_shift=1, ig_scale=2, num_classes=2)
    assert pure.item() == pytest.approx(5.0)
    combined = ode_sample(
        model,
        noise,
        torch.tensor([1]),
        steps=1,
        timestep_shift=1,
        ig_scale=2,
        cfg_scale=2,
        num_classes=2,
    )
    # IG(cond)=5, IG(uncond label token 2)=6, then CFG=6+2*(5-6)=4.
    assert combined.item() == pytest.approx(4.0)
    outside = ode_sample(
        model,
        noise,
        torch.tensor([1]),
        steps=1,
        timestep_shift=1,
        ig_scale=99,
        ig_interval=(0.1, 0.9),
        num_classes=2,
    )
    assert outside.item() == pytest.approx(3.0)


def test_cache_roundtrip_statistics_resume_and_size(tmp_path) -> None:
    config = ExperimentConfig()
    config.data.max_samples = 3
    config.rae.cache.directory = str(tmp_path)
    spec = token_cache_spec(config, 3)
    path = token_cache_path(config, 3)
    assert required_token_cache_bytes(spec) == 3 * 1024 * 8 * 8 * 2 + 3 + 8192
    assert TokenCacheWriter.initialize(path, spec) is False
    writer = TokenCacheWriter(path, spec)
    values = torch.empty(3, 1024, 8, 8)
    values[0].fill_(0)
    values[1].fill_(1)
    values[2].fill_(3)
    writer.write(torch.arange(3), values)
    assert writer.pending_indices().size == 0
    mean, std = writer.compute_statistics(chunk_size=1)
    writer.finalize(mean, std)
    assert TokenCacheWriter.initialize(path, spec) is True
    cache = TokenCache(config, 3)
    raw = cache.raw_lookup(torch.tensor([0, 2]))
    assert raw.dtype == torch.bfloat16
    restored = cache.denormalize(cache.lookup(torch.tensor([0, 2])))
    assert torch.allclose(restored, raw.float(), atol=0.02)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["fingerprint_inputs"]["statistics"]["accumulator_dtype"] == "float64"


def test_normalization_roundtrip_and_fingerprint_mismatch(tmp_path) -> None:
    tokens = torch.randn(2, 3, 2, 2)
    mean = torch.randn(3)
    std = torch.rand(3).add(0.1)
    assert torch.allclose(denormalize_tokens(normalize_tokens(tokens, mean, std), mean, std), tokens)
    config = ExperimentConfig()
    first = token_cache_spec(config, 2)
    config.data.revision = "different"
    second = token_cache_spec(config, 2)
    from jit_reg.token_cache import token_cache_fingerprint

    assert token_cache_fingerprint(first) != token_cache_fingerprint(second)


def test_label_only_dataset_does_not_read_rgb() -> None:
    class LabelRow(dict[str, int]):
        def __getitem__(self, key: str) -> int:
            if key == "image":
                raise AssertionError("training attempted to load RGB")
            return super().__getitem__(key)

    dataset = ImageNetDataset([LabelRow(label=7)], 256, load_pixels=False)
    assert dataset[0] == (None, 7, 0)


def test_rae_loader_loads_base_then_injects_it_into_compressor(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class DummyRAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def encode(self, pixels: torch.Tensor) -> torch.Tensor:
            return torch.zeros(pixels.shape[0], 1024, 8, 8)

        def decode(self, tokens: torch.Tensor) -> torch.Tensor:
            return torch.zeros(tokens.shape[0], 3, 256, 256)

    class Loader:
        @staticmethod
        def from_pretrained(repository: str, **kwargs: object) -> DummyRAE:
            calls.append((repository, kwargs))
            return DummyRAE()

    import transformers

    monkeypatch.setattr(transformers, "AutoModel", Loader)
    config = ExperimentConfig().rae
    codec = load_rae_codec(config, "cpu")
    assert [call[0] for call in calls] == [config.base_repository, config.compressor_repository]
    assert calls[1][1]["rae"].__class__ is DummyRAE
    assert calls[0][1]["revision"] == config.base_revision
    assert calls[1][1]["revision"] == config.compressor_revision
    assert calls[0][1]["trust_remote_code"] is True
    assert codec.encode(torch.zeros(1, 3, 256, 256)).shape == (1, 1024, 8, 8)
    assert codec.decode(torch.zeros(1, 1024, 8, 8)).shape == (1, 3, 256, 256)
    assert all(not parameter.requires_grad for parameter in codec.parameters())
