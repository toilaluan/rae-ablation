from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from .config import CONFIG_SCHEMA_VERSION, ExperimentConfig, load_checkpoint_config
from .model.compile import clean_state_dict

CHECKPOINT_SCHEMA_VERSION = 3
TRAINING_STATE_SCHEMA_VERSION = 3


def _tensor_state(state: dict[str, object]) -> dict[str, Tensor]:
    return {
        name: value.detach().contiguous()
        for name, value in state.items()
        if isinstance(value, Tensor)
    }


class EMA:
    """State-dict EMA that can safely pass through structural LoRA transitions."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = decay
        self.shadow = {
            name: value.detach().clone() for name, value in clean_state_dict(model).items()
            if isinstance(value, Tensor)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = _tensor_state(clean_state_dict(model))
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA keys changed; complete the stage transition first")
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

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)

    def to_model(self, template: nn.Module) -> Any:
        result = copy.deepcopy(template)
        result.load_state_dict(self.shadow)
        return result

    def replace_from_model(self, model: nn.Module) -> None:
        self.shadow = {
            name: value.detach().clone()
            for name, value in _tensor_state(clean_state_dict(model)).items()
        }


@dataclass(frozen=True)
class ResumeState:
    schema_version: int
    completed_epoch: int
    global_step: int
    stage_index: int
    stage: str
    epoch_within_stage: int
    transition_state: str
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
    def __init__(self, directory: str | Path, keep_last_n: int = 8) -> None:
        self.directory = Path(directory)
        self.keep_last_n = keep_last_n
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
            "stage_index": state.stage_index,
            "stage": state.stage,
            "epoch_within_stage": state.epoch_within_stage,
            "transition_state": state.transition_state,
            "training_complete": state.training_complete,
            "images_seen": state.images_seen,
            "estimated_flops": state.estimated_flops,
            "wall_clock_seconds": state.wall_clock_seconds,
            "data_fingerprint": state.data_fingerprint,
            "representation_fingerprint": state.representation_fingerprint,
        }
        if target.exists():
            existing_path = target / "metadata.json"
            existing = json.loads(existing_path.read_text()) if existing_path.is_file() else None
            if idempotent and isinstance(existing, dict):
                stable_keys = (
                    "schema_version",
                    "name",
                    "completed_epoch",
                    "global_step",
                    "stage_index",
                    "stage",
                    "transition_state",
                    "data_fingerprint",
                    "representation_fingerprint",
                )
                if all(existing.get(key) == metadata.get(key) for key in stable_keys):
                    return target
            raise FileExistsError(f"checkpoint already exists: {target}")
        temporary = self.directory / f".{name}.partial-{os.getpid()}"
        temporary.mkdir()
        try:
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
        except BaseException:
            # Leave partial directories as evidence; subsequent runs never erase them.
            raise
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
        name = f"epoch-{state.completed_epoch:04d}"
        return self._save_named(name, model=model, ema=ema, state=state, config=config, idempotent=False)

    def save_boundary(
        self,
        *,
        target_stage_index: int,
        position: str,
        model: nn.Module,
        ema: EMA,
        state: ResumeState,
        config: ExperimentConfig,
    ) -> Path:
        if position not in {"before", "after"}:
            raise ValueError("boundary position must be before or after")
        if state.transition_state != position:
            raise ValueError("boundary state and requested position disagree")
        name = f"boundary-{target_stage_index:02d}-{position}"
        return self._save_named(name, model=model, ema=ema, state=state, config=config, idempotent=True)

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
            raise ValueError(
                "checkpoint predates the progressive LightningDiT schema; DDT/Mage "
                "checkpoints are not supported"
            )
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("checkpoint schema is incompatible; old checkpoints cannot be migrated")
        config = load_checkpoint_config(path / "config.json")
        raw = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
        if raw.get("schema_version") != TRAINING_STATE_SCHEMA_VERSION:
            raise ValueError("checkpoint training-state schema is incompatible")
        state = ResumeState(**raw)
        for name in (
            "completed_epoch",
            "global_step",
            "stage_index",
            "stage",
            "transition_state",
            "training_complete",
            "data_fingerprint",
            "representation_fingerprint",
        ):
            if metadata.get(name) != getattr(state, name):
                raise ValueError(f"checkpoint {name} metadata is inconsistent")
        if state.stage not in config.curriculum.stages:
            raise ValueError("checkpoint curriculum stage is invalid")
        if config.curriculum.stages[state.stage_index] != state.stage:
            raise ValueError("checkpoint curriculum stage index is invalid")
        if state.training_complete and (
            state.stage != "rgb" or state.completed_epoch != config.total_epochs
        ):
            raise ValueError("completed checkpoint is not at the end of the RGB stage")
        for name, expected in (
            ("data_fingerprint", expected_data_fingerprint),
            ("representation_fingerprint", expected_representation_fingerprint),
        ):
            if expected is not None and getattr(state, name) != expected:
                raise ValueError(f"checkpoint {name} mismatch")
        model.load_state_dict(load_file(path / "model.safetensors"))
        ema.load_state_dict(load_file(path / "ema.safetensors"))
        return state

    def export_ema(self, ema: EMA, path: str | Path) -> None:
        target = Path(path)
        temporary = target.with_suffix(target.suffix + f".partial-{os.getpid()}")
        save_file(_tensor_state(ema.state_dict()), temporary)
        os.replace(temporary, target)
