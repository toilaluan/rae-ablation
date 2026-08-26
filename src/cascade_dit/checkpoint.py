from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from .config import CONFIG_SCHEMA_VERSION, ExperimentConfig
from .model.compile import clean_state_dict

CHECKPOINT_SCHEMA_VERSION = 5
TRAINING_STATE_SCHEMA_VERSION = 5
TrainingPhase = Literal["semantic", "pixel"]


def _tensor_state(state: dict[str, object]) -> dict[str, Tensor]:
    return {
        name: value.detach().contiguous()
        for name, value in state.items()
        if isinstance(value, Tensor)
    }


class EMA:
    """State-dict EMA for the canonical uncompiled DDT."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = decay
        self.shadow = {
            name: value.detach().clone()
            for name, value in clean_state_dict(model).items()
            if isinstance(value, Tensor)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = _tensor_state(clean_state_dict(model))
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA keys changed during training")
        for name, source in state.items():
            target = self.shadow[name]
            if source.is_floating_point():
                target.lerp_(source.to(target), 1 - self.decay)
            else:
                target.copy_(source.to(target))

    def state_dict(self) -> dict[str, Tensor]:
        return self.shadow

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA state keys do not match model")
        for name, value in state.items():
            self.shadow[name].copy_(value.to(self.shadow[name]))

    def to_model(self, template: nn.Module) -> Any:
        result = copy.deepcopy(template)
        result.load_state_dict(self.shadow)
        return result


@dataclass(frozen=True)
class ResumeState:
    schema_version: int
    completed_epoch: int
    global_step: int
    phase: TrainingPhase
    epoch_within_phase: int
    training_complete: bool
    images_seen: int
    estimated_flops: int
    wall_clock_seconds: float
    optimizer: dict[str, Any] | None
    scheduler: dict[str, Any] | None
    rng_streams: dict[str, Any]
    data_fingerprint: str
    representation_fingerprint: str


class CheckpointManager:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _save_named(
        self,
        name: str,
        *,
        model: nn.Module,
        ema: EMA,
        state: ResumeState,
        config: ExperimentConfig,
        idempotent: bool,
    ) -> Path:
        target = self.directory / name
        metadata = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "config_schema_version": CONFIG_SCHEMA_VERSION,
            "name": name,
            "completed_epoch": state.completed_epoch,
            "global_step": state.global_step,
            "phase": state.phase,
            "epoch_within_phase": state.epoch_within_phase,
            "training_complete": state.training_complete,
            "images_seen": state.images_seen,
            "estimated_flops": state.estimated_flops,
            "wall_clock_seconds": state.wall_clock_seconds,
            "data_fingerprint": state.data_fingerprint,
            "representation_fingerprint": state.representation_fingerprint,
            "model_format": "plain",
            "optimizer_restorable": state.optimizer is not None,
        }
        if target.exists():
            existing_path = target / "metadata.json"
            existing = json.loads(existing_path.read_text()) if existing_path.is_file() else None
            stable = (
                "schema_version",
                "name",
                "completed_epoch",
                "global_step",
                "phase",
                "data_fingerprint",
                "representation_fingerprint",
            )
            if idempotent and isinstance(existing, dict) and all(
                existing.get(key) == metadata.get(key) for key in stable
            ):
                return target
            raise FileExistsError(f"checkpoint already exists: {target}")
        temporary = self.directory / f".{name}.partial-{os.getpid()}"
        temporary.mkdir()
        save_file(_tensor_state(clean_state_dict(model)), temporary / "model.safetensors")
        save_file(_tensor_state(ema.state_dict()), temporary / "ema.safetensors")
        torch.save(asdict(state), temporary / "training.pt")
        (temporary / "config.json").write_text(
            json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n"
        )
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, target)
        self._point_last(target.name)
        return target

    def _point_last(self, name: str) -> None:
        link = self.directory / "last"
        temporary = self.directory / f".last-{os.getpid()}"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(name, target_is_directory=True)
        os.replace(temporary, link)

    def save(
        self,
        *,
        model: nn.Module,
        ema: EMA,
        state: ResumeState,
        config: ExperimentConfig,
    ) -> Path:
        return self._save_named(
            f"epoch-{state.completed_epoch:04d}",
            model=model,
            ema=ema,
            state=state,
            config=config,
            idempotent=False,
        )

    def save_phase_boundary(
        self,
        *,
        model: nn.Module,
        ema: EMA,
        state: ResumeState,
        config: ExperimentConfig,
    ) -> Path:
        if state.phase != "pixel" or state.epoch_within_phase != 0:
            raise ValueError("phase boundary must be the beginning of the pixel phase")
        return self._save_named(
            "boundary-pixel",
            model=model,
            ema=ema,
            state=state,
            config=config,
            idempotent=True,
        )

    def load(
        self,
        checkpoint: str | Path,
        model: nn.Module,
        ema: EMA,
        *,
        expected_data_fingerprint: str | None = None,
        expected_representation_fingerprint: str | None = None,
    ) -> ResumeState:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = self.directory / path
        if path.name == "last" and path.is_symlink():
            path = path.resolve()
        metadata_path = path / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError("checkpoint predates the semantic-pixel DDT schema")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("checkpoint schema is incompatible; old checkpoints are rejected")
        raw = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
        if not isinstance(raw, dict) or raw.get("schema_version") != TRAINING_STATE_SCHEMA_VERSION:
            raise ValueError("checkpoint training-state schema is incompatible")
        state = ResumeState(**raw)
        for name in (
            "completed_epoch",
            "global_step",
            "phase",
            "epoch_within_phase",
            "training_complete",
            "data_fingerprint",
            "representation_fingerprint",
        ):
            if metadata.get(name) != getattr(state, name):
                raise ValueError(f"checkpoint {name} metadata is inconsistent")
        if (
            state.completed_epoch < 0
            or state.global_step < 0
            or state.epoch_within_phase < 0
            or state.phase not in {"semantic", "pixel"}
        ):
            raise ValueError("checkpoint training progress is invalid")
        for name, expected in (
            ("data_fingerprint", expected_data_fingerprint),
            ("representation_fingerprint", expected_representation_fingerprint),
        ):
            if expected is not None and getattr(state, name) != expected:
                raise ValueError(f"checkpoint {name} mismatch")
        model.load_state_dict(load_file(path / "model.safetensors"))
        ema.load_state_dict(load_file(path / "ema.safetensors"))
        return state
