from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch
from torch import nn

from jit_reg.config import ExperimentConfig
from jit_reg.prior import (
    CheckpointPrior,
    PriorProvenance,
    load_checkpoint_prior,
    prior_provenance,
    source_from_gaussian,
    validate_prior_provenance,
)


class TinyBasePredictor(nn.Module):
    def __init__(self, _config: object | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.calls: list[tuple[torch.Tensor, torch.Tensor, bool]] = []

    def forward_base_prediction(
        self,
        value: torch.Tensor,
        timestep: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append((timestep.detach().clone(), labels.detach().clone(), torch.is_grad_enabled()))
        shape = (labels.shape[0], *((1,) * (value.ndim - 1)))
        return self.weight * value + labels.reshape(shape)


def test_prior_configuration_defaults_and_k_restriction() -> None:
    config = ExperimentConfig()
    assert config.prior.checkpoint_dir is None
    assert config.prior.weight_source == "ema"
    config.validate()

    config.prior.weight_source = "other"
    with pytest.raises(ValueError, match="model or ema"):
        config.validate()

    config = ExperimentConfig()
    config.prior.checkpoint_dir = "checkpoint"
    config.exploration.k_early = 2
    with pytest.raises(ValueError, match="require exploration K=1"):
        config.validate()


def test_checkpoint_prior_uses_true_labels_timestep_one_and_no_grad() -> None:
    model = TinyBasePredictor()
    prior = CheckpointPrior(
        model,  # type: ignore[arg-type]
        PriorProvenance("checkpoint", "ema", "digest"),
    )
    gaussian = torch.tensor([[[[2.0]]], [[[3.0]]]], requires_grad=True)
    labels = torch.tensor([4, 5])
    source = source_from_gaussian(prior, gaussian, labels)

    assert torch.equal(source, torch.tensor([[[[6.0]]], [[[8.0]]]]))
    timestep, seen_labels, grad_enabled = model.calls[0]
    assert torch.equal(timestep, torch.ones(2))
    assert torch.equal(seen_labels, labels)
    assert grad_enabled is False
    assert source.requires_grad is False
    assert all(not parameter.requires_grad for parameter in prior.parameters())
    prior.train()
    assert prior.training is False


def test_gaussian_fallback_is_exact() -> None:
    gaussian = torch.randn(2, 3, dtype=torch.bfloat16)
    source = source_from_gaussian(None, gaussian, torch.tensor([0, 1]))
    assert source is gaussian


def test_checkpoint_loader_selects_requested_weights_and_records_digest(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "prior"
    checkpoint.mkdir()
    source_config = ExperimentConfig()
    (checkpoint / "config.json").write_text(json.dumps(source_config.as_dict()))
    (checkpoint / "metadata.json").write_text(
        json.dumps({"token_cache_fingerprint": "cache"})
    )
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "ema.safetensors").write_bytes(b"ema")

    loaded: list[str] = []

    def fake_load(path: str, *, device: str) -> dict[str, torch.Tensor]:
        del device
        loaded.append(path)
        value = 2.0 if path.endswith("model.safetensors") else 3.0
        return {"weight": torch.tensor(value)}

    monkeypatch.setattr("jit_reg.prior.DDT", TinyBasePredictor)
    monkeypatch.setattr("jit_reg.prior.load_file", fake_load)

    config = deepcopy(source_config)
    config.prior.checkpoint_dir = str(checkpoint)
    config.prior.weight_source = "model"
    prior = load_checkpoint_prior(config, "cpu", token_cache_fingerprint="cache")
    assert prior is not None
    assert loaded == [str(checkpoint / "model.safetensors")]
    assert prior.model.weight.item() == 2
    assert prior.provenance.weight_source == "model"
    assert prior.provenance.checkpoint_dir == str(checkpoint.resolve())
    assert len(prior.provenance.weights_sha256) == 64


def test_prior_provenance_matches_weight_identity_not_directory() -> None:
    prior = CheckpointPrior(
        TinyBasePredictor(),  # type: ignore[arg-type]
        PriorProvenance("first", "ema", "same"),
    )
    actual = prior_provenance(prior)
    validate_prior_provenance(actual, {"checkpoint_dir": "moved", "weight_source": "ema", "weights_sha256": "same"})
    with pytest.raises(ValueError, match="weights_sha256"):
        validate_prior_provenance(
            actual,
            {"checkpoint_dir": "first", "weight_source": "ema", "weights_sha256": "changed"},
        )
