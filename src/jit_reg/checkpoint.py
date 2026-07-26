from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from .model.compile import clean_state_dict

TRAINING_STATE_SCHEMA_VERSION = 3


def _contiguous(state: dict[str, object]) -> dict[str, Tensor]:
    return {
        key: value.detach().contiguous()
        for key, value in state.items()
        if isinstance(value, Tensor)
    }


@dataclass
class ResumeState:
    schema_version: int
    epoch: int
    global_step: int
    last_completed_validation_epoch: int | None
    wandb_run_id: str | None
    data_fingerprint: str
    validation_plan_hash: str
    optimizer: dict[str, Any]
    rng_streams: dict[str, Tensor]
    feature_cache_fingerprints: dict[str, str] | None = None


def should_checkpoint(completed_epoch: int, total_epochs: int, interval: int) -> bool:
    return completed_epoch == total_epochs or (interval > 0 and completed_epoch % interval == 0)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995) -> None:
        self.decay = decay
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            target = self.shadow[key]
            if target.is_floating_point():
                target.lerp_(value.detach().to(target.device), 1 - self.decay)
            else:
                target.copy_(value)

    def state_dict(self) -> dict[str, Tensor]:
        return self.shadow

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA state keys do not match model")
        for key, value in state.items():
            self.shadow[key].copy_(value.to(self.shadow[key].device))

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


class CheckpointManager:
    def __init__(self, directory: str | Path, keep_last_n: int = 3) -> None:
        self.directory = Path(directory)
        self.keep_last_n = keep_last_n
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        completed_epoch: int,
        global_step: int,
        model: nn.Module,
        ema: EMA,
        optimizer_state: dict[str, Any],
        rng_streams: dict[str, Tensor],
        data_fingerprint: str,
        validation_plan_hash: str,
        validation_plan_path: str | Path,
        last_completed_validation_epoch: int | None,
        wandb_run_id: str | None,
        resolved_config: dict[str, Any],
        feature_cache_fingerprints: dict[str, str] | None = None,
        fid: float | None = None,
    ) -> Path:
        name = f"epoch-{completed_epoch:04d}"
        target = self.directory / name
        if target.exists():
            raise FileExistsError(f"checkpoint already exists: {target}")
        temporary = self.directory / f".{name}.partial-{os.getpid()}"
        temporary.mkdir()
        raw_state = clean_state_dict(model)
        save_file(_contiguous(raw_state), str(temporary / "model.safetensors"))
        save_file(_contiguous(ema.state_dict()), str(temporary / "ema.safetensors"))
        training = {
            "schema_version": TRAINING_STATE_SCHEMA_VERSION,
            "epoch": completed_epoch,
            "global_step": global_step,
            "last_completed_validation_epoch": last_completed_validation_epoch,
            "wandb_run_id": wandb_run_id,
            "data_fingerprint": data_fingerprint,
            "validation_plan_hash": validation_plan_hash,
            "optimizer": optimizer_state,
            "rng_streams": rng_streams,
            "feature_cache_fingerprints": feature_cache_fingerprints,
        }
        torch.save(training, temporary / "training.pt")
        (temporary / "config.json").write_text(
            json.dumps(resolved_config, indent=2, sort_keys=True) + "\n"
        )
        plan_path = Path(validation_plan_path)
        if not plan_path.is_file():
            raise FileNotFoundError(f"validation plan is missing: {plan_path}")
        plan_state = torch.load(plan_path, map_location="cpu", weights_only=True)
        if plan_state.get("plan_hash") != validation_plan_hash:
            raise ValueError("validation plan file and hash do not match")
        shutil.copy2(plan_path, temporary / "validation-plan.pt")
        (temporary / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "completed_epoch": completed_epoch,
                    "global_step": global_step,
                    "data_fingerprint": data_fingerprint,
                    "validation_plan_hash": validation_plan_hash,
                    "feature_cache_fingerprints": feature_cache_fingerprints,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        os.replace(temporary, target)
        self._point("last", name)
        if fid is not None:
            best_path = self.directory / "best-fid.json"
            previous = json.loads(best_path.read_text()) if best_path.exists() else None
            if previous is None or fid < previous["fid"]:
                best_path.write_text(json.dumps({"fid": fid, "checkpoint": name}) + "\n")
                self._point("best-fid", name)
        self._retain()
        return target

    def _point(self, pointer: str, target: str) -> None:
        link = self.directory / pointer
        temporary = self.directory / f".{pointer}-{os.getpid()}"
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(target, target_is_directory=True)
        os.replace(temporary, link)

    def _retain(self) -> None:
        if self.keep_last_n <= 0:
            return
        checkpoints = sorted(
            path for path in self.directory.glob("epoch-[0-9][0-9][0-9][0-9]") if path.is_dir()
        )
        protected: set[str] = set()
        best = self.directory / "best-fid"
        if best.is_symlink():
            protected.add(os.readlink(best))
        protected.update(
            path.name
            for path in checkpoints
            if any(
                child.is_dir() and child.name.startswith("samples-")
                for child in path.iterdir()
            )
        )
        removable = [
            path for path in checkpoints[: -self.keep_last_n] if path.name not in protected
        ]
        for path in removable:
            shutil.rmtree(path)

    def load(
        self,
        checkpoint: str | Path,
        model: nn.Module,
        ema: EMA,
        expected_data_fingerprint: str | None = None,
        expected_validation_plan_hash: str | None = None,
        expected_feature_cache_fingerprints: dict[str, str] | None = None,
    ) -> ResumeState:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = self.directory / path
        metadata_path = path / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError(
                "checkpoint predates schema v2; old checkpoints are not supported"
            )
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("schema_version") != 2:
            raise ValueError(
                "checkpoint schema is incompatible; schema-v1 checkpoints cannot be migrated"
            )
        from .config import load_checkpoint_config

        state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
        state_schema = state.get("schema_version")
        if state_schema == 2:
            raise ValueError(
                "checkpoint training-state schema v2 is sampling-only after "
                "optimizer routing change; exact training resume requires schema v3"
            )
        if state_schema != TRAINING_STATE_SCHEMA_VERSION:
            raise ValueError(
                "checkpoint training state is incompatible with schema v3"
            )
        load_checkpoint_config(path / "config.json")
        model.load_state_dict(load_file(str(path / "model.safetensors")))
        ema.load_state_dict(load_file(str(path / "ema.safetensors")))
        if expected_data_fingerprint and state["data_fingerprint"] != expected_data_fingerprint:
            raise ValueError("checkpoint dataset/preprocessing fingerprint mismatch")
        if (
            expected_validation_plan_hash
            and state["validation_plan_hash"] != expected_validation_plan_hash
        ):
            raise ValueError("checkpoint validation-plan hash mismatch")
        saved_cache_fingerprints = state.get("feature_cache_fingerprints")
        if expected_feature_cache_fingerprints != saved_cache_fingerprints and (
            expected_feature_cache_fingerprints is not None
            or saved_cache_fingerprints is not None
        ):
            raise ValueError("checkpoint feature-cache fingerprints mismatch")
        return ResumeState(**state)

    def export_ema(self, ema: EMA, path: str | Path) -> None:
        path = Path(path)
        temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
        save_file(_contiguous(ema.state_dict()), str(temporary))
        os.replace(temporary, path)
