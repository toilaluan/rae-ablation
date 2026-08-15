from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from .config import ExperimentConfig, load_checkpoint_config
from .model import DDT


@dataclass(frozen=True)
class PriorProvenance:
    checkpoint_dir: str
    weight_source: str
    weights_sha256: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_directory(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_dir():
        raise FileNotFoundError(f"prior checkpoint directory does not exist: {candidate}")
    if not (candidate / "config.json").is_file() and (candidate / "last").is_dir():
        candidate = candidate / "last"
    resolved = candidate.resolve()
    if not (resolved / "config.json").is_file():
        raise FileNotFoundError(f"prior checkpoint config is missing: {resolved / 'config.json'}")
    return resolved


def _weights_path(directory: Path, source: str) -> Path:
    name = "model.safetensors" if source == "model" else "ema.safetensors"
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(f"prior checkpoint {source} weights are missing: {path}")
    return path


def _metadata(directory: Path) -> dict[str, Any]:
    path = directory / "metadata.json"
    if not path.is_file():
        raise ValueError(f"prior checkpoint metadata is missing: {path}")
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("prior checkpoint metadata must contain an object")
    return values


class CheckpointPrior(nn.Module):
    """Frozen DDT base head that maps Gaussian tensors to flow sources."""

    def __init__(self, model: DDT, provenance: PriorProvenance) -> None:
        super().__init__()
        self.model = model.eval()
        self.provenance = provenance
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> CheckpointPrior:
        del mode
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(self, gaussian: Tensor, labels: Tensor) -> Tensor:
        timestep = torch.ones(gaussian.shape[0], device=gaussian.device, dtype=torch.float32)
        return self.model.forward_base_prediction(gaussian, timestep, labels).float()


def load_checkpoint_prior(
    config: ExperimentConfig,
    device: torch.device | str,
    *,
    token_cache_fingerprint: str,
) -> CheckpointPrior | None:
    if config.prior.checkpoint_dir is None:
        return None

    directory = _checkpoint_directory(config.prior.checkpoint_dir)
    prior_config = load_checkpoint_config(directory / "config.json")
    if prior_config.prior.checkpoint_dir is not None:
        raise ValueError("prior generator must be a normally trained Gaussian-prior checkpoint")
    if prior_config.model != config.model:
        raise ValueError("prior checkpoint model configuration does not match the student")
    if prior_config.rae != config.rae:
        raise ValueError("prior checkpoint RAE configuration does not match the student")
    metadata = _metadata(directory)
    if metadata.get("token_cache_fingerprint") != token_cache_fingerprint:
        raise ValueError("prior checkpoint token-cache fingerprint does not match the student")

    weights = _weights_path(directory, config.prior.weight_source)
    model = DDT(prior_config.model).to(device)
    model.load_state_dict(load_file(str(weights), device=str(torch.device(device))))
    provenance = PriorProvenance(
        checkpoint_dir=str(directory),
        weight_source=config.prior.weight_source,
        weights_sha256=_sha256(weights),
    )
    return CheckpointPrior(model, provenance).to(device)


def prior_provenance(prior: CheckpointPrior | None) -> dict[str, str] | None:
    return None if prior is None else prior.provenance.as_dict()


def validate_prior_provenance(
    actual: Mapping[str, str] | None,
    expected: Mapping[str, object] | None,
) -> None:
    if actual is None and expected is None:
        return
    if actual is None or expected is None:
        raise ValueError("checkpoint prior provenance does not match the configured prior")
    for key in ("weight_source", "weights_sha256"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"checkpoint prior provenance mismatch for {key}")


def source_from_gaussian(
    prior: CheckpointPrior | None,
    gaussian: Tensor,
    labels: Tensor,
) -> Tensor:
    return gaussian if prior is None else prior(gaussian, labels)
