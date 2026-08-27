from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .checkpoint import EMA
from .config import CONFIG_SCHEMA_VERSION, ExperimentConfig, ModelConfig, load_checkpoint_config
from .model import SemanticPixelDecoder
from .model.compile import clean_state_dict
from .representations import ChannelStatistics, fingerprint

PIXEL_DECODER_CHECKPOINT_SCHEMA_VERSION = 1
PIXEL_DECODER_TRAINING_SCHEMA_VERSION = 1


def pixel_decoder_architecture_payload(config: ModelConfig) -> dict[str, object]:
    pixel = config.pixel_decoder
    return {
        "token_dim": config.token_dim,
        "pixel_decoder": {
            "width": pixel.width,
            "depth": pixel.depth,
            "heads": pixel.heads,
        },
        "feedforward": asdict(config.feedforward),
        "attention": asdict(config.attention),
    }


def pixel_decoder_architecture_fingerprint(config: ModelConfig) -> str:
    return fingerprint(pixel_decoder_architecture_payload(config))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PixelDecoderResumeState:
    schema_version: int
    completed_epoch: int
    global_step: int
    training_complete: bool
    optimizer: dict[str, Any] | None
    scheduler: dict[str, Any] | None
    rng_state: dict[str, Any]
    wall_clock_seconds: float
    data_fingerprint: str
    encoder_fingerprint: str


@dataclass(frozen=True)
class PixelDecoderArtifact:
    directory: Path
    model: SemanticPixelDecoder
    statistics: ChannelStatistics
    weights_kind: str
    weights_sha256: str
    metadata: dict[str, object]


def resolve_pixel_decoder_checkpoint(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_dir():
        raise FileNotFoundError(f"pixel decoder checkpoint does not exist: {candidate}")
    if not (candidate / "metadata.json").is_file() and (candidate / "last").is_dir():
        candidate = candidate / "last"
    resolved = candidate.resolve()
    if not (resolved / "metadata.json").is_file():
        raise ValueError("pixel decoder checkpoint metadata is missing")
    return resolved


class PixelDecoderCheckpointManager:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        model: SemanticPixelDecoder,
        ema: EMA,
        state: PixelDecoderResumeState,
        config: ExperimentConfig,
        statistics: ChannelStatistics,
    ) -> Path:
        name = f"epoch-{state.completed_epoch:04d}"
        target = self.directory / name
        if target.exists():
            raise FileExistsError(f"pixel decoder checkpoint already exists: {target}")
        temporary = self.directory / f".{name}.partial-{os.getpid()}"
        temporary.mkdir()
        save_file(
            {
                key: value.detach().contiguous()
                for key, value in clean_state_dict(model).items()
                if isinstance(value, torch.Tensor)
            },
            temporary / "model.safetensors",
        )
        save_file(ema.state_dict(), temporary / "ema.safetensors")
        save_file(
            {
                "mean": statistics.mean.float().contiguous(),
                "std": statistics.std.float().contiguous(),
            },
            temporary / "semantic-statistics.safetensors",
        )
        torch.save(asdict(state), temporary / "training.pt")
        (temporary / "config.json").write_text(
            json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n"
        )
        metadata = {
            "schema_version": PIXEL_DECODER_CHECKPOINT_SCHEMA_VERSION,
            "config_schema_version": CONFIG_SCHEMA_VERSION,
            "name": name,
            "completed_epoch": state.completed_epoch,
            "global_step": state.global_step,
            "training_complete": state.training_complete,
            "data_fingerprint": state.data_fingerprint,
            "encoder_fingerprint": state.encoder_fingerprint,
            "architecture_fingerprint": pixel_decoder_architecture_fingerprint(
                config.model
            ),
            "semantic_count": statistics.count,
            "model_sha256": file_sha256(temporary / "model.safetensors"),
            "ema_sha256": file_sha256(temporary / "ema.safetensors"),
            "statistics_sha256": file_sha256(
                temporary / "semantic-statistics.safetensors"
            ),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, target)
        link = self.directory / "last"
        temporary_link = self.directory / f".last-{os.getpid()}"
        temporary_link.unlink(missing_ok=True)
        temporary_link.symlink_to(name, target_is_directory=True)
        os.replace(temporary_link, link)
        return target

    def load_training(
        self,
        checkpoint: str | Path,
        model: SemanticPixelDecoder,
        ema: EMA,
    ) -> tuple[PixelDecoderResumeState, ExperimentConfig, ChannelStatistics]:
        path = resolve_pixel_decoder_checkpoint(checkpoint)
        metadata = _load_metadata(path)
        config = load_checkpoint_config(path / "config.json")
        expected = pixel_decoder_architecture_fingerprint(model.config)
        if metadata.get("architecture_fingerprint") != expected:
            raise ValueError("pixel decoder checkpoint architecture mismatch")
        for kind in ("model", "ema"):
            weights = path / f"{kind}.safetensors"
            if metadata.get(f"{kind}_sha256") != file_sha256(weights):
                raise ValueError(f"pixel decoder {kind} weight checksum mismatch")
        raw = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != PIXEL_DECODER_TRAINING_SCHEMA_VERSION
        ):
            raise ValueError("pixel decoder training-state schema is incompatible")
        state = PixelDecoderResumeState(**raw)
        if (
            metadata.get("completed_epoch") != state.completed_epoch
            or metadata.get("global_step") != state.global_step
            or metadata.get("training_complete") != state.training_complete
        ):
            raise ValueError("pixel decoder checkpoint progress metadata is inconsistent")
        model.load_state_dict(load_file(path / "model.safetensors"))
        ema.load_state_dict(load_file(path / "ema.safetensors"))
        statistics = _load_statistics(path, metadata)
        return state, config, statistics


def _load_metadata(path: Path) -> dict[str, object]:
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("schema_version") != PIXEL_DECODER_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("pixel decoder checkpoint schema is incompatible")
    return metadata


def _load_statistics(
    path: Path, metadata: dict[str, object]
) -> ChannelStatistics:
    statistics_path = path / "semantic-statistics.safetensors"
    if metadata.get("statistics_sha256") != file_sha256(statistics_path):
        raise ValueError("pixel decoder semantic statistics checksum mismatch")
    tensors = load_file(statistics_path)
    try:
        result = ChannelStatistics(
            tensors["mean"].float(),
            tensors["std"].float(),
            int(metadata["semantic_count"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("pixel decoder semantic statistics are corrupt") from error
    if (
        result.mean.ndim != 1
        or result.mean.shape != result.std.shape
        or result.count < 2
        or torch.any(result.std <= 0)
        or not torch.isfinite(result.mean).all()
        or not torch.isfinite(result.std).all()
    ):
        raise ValueError("pixel decoder semantic statistics are invalid")
    return result


def load_pixel_decoder_artifact(
    checkpoint: str | Path,
    *,
    expected_config: ModelConfig,
    weights_kind: str = "ema",
    device: torch.device | str = "cpu",
) -> PixelDecoderArtifact:
    if weights_kind not in {"ema", "model"}:
        raise ValueError("pixel decoder weights kind must be ema or model")
    path = resolve_pixel_decoder_checkpoint(checkpoint)
    metadata = _load_metadata(path)
    expected = pixel_decoder_architecture_fingerprint(expected_config)
    if metadata.get("architecture_fingerprint") != expected:
        raise ValueError("pretrained pixel decoder architecture mismatch")
    weights = path / f"{weights_kind}.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"pixel decoder weights are missing: {weights}")
    digest = file_sha256(weights)
    if metadata.get(f"{weights_kind}_sha256") != digest:
        raise ValueError("pixel decoder weight checksum mismatch")
    decoder = SemanticPixelDecoder(expected_config).to(device)
    decoder.load_state_dict(load_file(weights, device=str(device)))
    statistics = _load_statistics(path, metadata)
    return PixelDecoderArtifact(
        path,
        decoder,
        statistics,
        weights_kind,
        digest,
        metadata,
    )
