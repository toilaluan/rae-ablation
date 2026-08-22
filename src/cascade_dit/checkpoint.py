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

CHECKPOINT_SCHEMA_VERSION = 1
TRAINING_STATE_SCHEMA_VERSION = 1


def _contiguous(state: dict[str, object]) -> dict[str, Tensor]:
    return {
        key: value.detach().contiguous()
        for key, value in state.items()
        if isinstance(value, Tensor)
    }


def _validate_provenance(
    architecture: str,
    representation_provenance: dict[str, Any],
    cache_provenance: dict[str, str],
) -> None:
    if architecture not in {"latent", "shared_cascade", "decoupled_cascade"}:
        raise ValueError("checkpoint architecture is invalid")
    mage = representation_provenance.get("mage_vae")
    if not isinstance(mage, dict):
        raise ValueError("checkpoint requires Mage-VAE provenance")
    required = {
        "repository",
        "revision",
        "weight_sha256",
        "posterior",
        "image_latent_shape",
        "encode_preprocessing",
    }
    if not required.issubset(mage) or mage.get("posterior") != "mean":
        raise ValueError("checkpoint Mage-VAE provenance is incomplete")
    if architecture != "latent":
        semantic = representation_provenance.get("semantic_encoder")
        if not isinstance(semantic, dict) or not all(
            semantic.get(key) for key in ("repository", "revision", "fingerprint")
        ):
            raise ValueError("cascade checkpoint requires semantic encoder provenance")
    expected_caches = {"image_latents", "semantic_tokens"}
    if set(cache_provenance) != expected_caches or not all(cache_provenance.values()):
        raise ValueError("checkpoint cache provenance is incomplete")


@dataclass
class ResumeState:
    schema_version: int
    epoch: int
    global_step: int
    last_completed_validation_epoch: int | None
    wandb_run_id: str | None
    architecture: str
    data_fingerprint: str
    validation_plan_hash: str
    optimizer: dict[str, Any]
    rng_streams: dict[str, Tensor]
    representation_provenance: dict[str, Any]
    cache_provenance: dict[str, str]


def should_checkpoint(completed_epoch: int, total_epochs: int, interval: int) -> bool:
    return completed_epoch == total_epochs or (
        interval > 0 and completed_epoch % interval == 0
    )


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995) -> None:
        self.decay = decay
        self.shadow = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        self._floating_keys = tuple(
            key for key, value in self.shadow.items() if value.is_floating_point()
        )
        self._nonfloating_keys = tuple(
            key for key, value in self.shadow.items() if not value.is_floating_point()
        )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA state keys do not match model")
        groups: dict[
            tuple[torch.device, torch.dtype], tuple[list[Tensor], list[Tensor]]
        ] = {}
        for key in self._floating_keys:
            target = self.shadow[key]
            source = state[key].detach()
            if source.device != target.device or source.dtype != target.dtype:
                raise ValueError("EMA model device or dtype changed after initialization")
            targets, sources = groups.setdefault(
                (target.device, target.dtype), ([], [])
            )
            targets.append(target)
            sources.append(source)
        for targets, sources in groups.values():
            torch._foreach_lerp_(targets, sources, 1 - self.decay)
        for key in self._nonfloating_keys:
            self.shadow[key].copy_(state[key])

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
        architecture: str,
        data_fingerprint: str,
        validation_plan_hash: str,
        validation_plan_path: str | Path,
        last_completed_validation_epoch: int | None,
        wandb_run_id: str | None,
        resolved_config: dict[str, Any],
        representation_provenance: dict[str, Any],
        cache_provenance: dict[str, str],
        fid: float | None = None,
    ) -> Path:
        _validate_provenance(
            architecture, representation_provenance, cache_provenance
        )
        configured_architecture = resolved_config.get("model", {}).get("architecture")
        if configured_architecture != architecture:
            raise ValueError("checkpoint architecture and resolved config disagree")
        model_architecture = getattr(getattr(model, "config", None), "architecture", None)
        if model_architecture != architecture:
            raise ValueError("checkpoint architecture and model disagree")
        name = f"epoch-{completed_epoch:04d}"
        target = self.directory / name
        if target.exists():
            raise FileExistsError(f"checkpoint already exists: {target}")
        temporary = self.directory / f".{name}.partial-{os.getpid()}"
        temporary.mkdir()
        save_file(
            _contiguous(clean_state_dict(model)), str(temporary / "model.safetensors")
        )
        save_file(
            _contiguous(ema.state_dict()), str(temporary / "ema.safetensors")
        )
        training = {
            "schema_version": TRAINING_STATE_SCHEMA_VERSION,
            "epoch": completed_epoch,
            "global_step": global_step,
            "last_completed_validation_epoch": last_completed_validation_epoch,
            "wandb_run_id": wandb_run_id,
            "architecture": architecture,
            "data_fingerprint": data_fingerprint,
            "validation_plan_hash": validation_plan_hash,
            "optimizer": optimizer_state,
            "rng_streams": rng_streams,
            "representation_provenance": representation_provenance,
            "cache_provenance": cache_provenance,
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
        metadata = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "completed_epoch": completed_epoch,
            "global_step": global_step,
            "architecture": architecture,
            "data_fingerprint": data_fingerprint,
            "validation_plan_hash": validation_plan_hash,
            "representation_provenance": representation_provenance,
            "cache_provenance": cache_provenance,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, target)
        self._point("last", name)
        if fid is not None:
            self._update_best_fid(fid, name)
        self._retain()
        return target

    def _update_best_fid(self, fid: float, name: str) -> None:
        best_path = self.directory / "best-fid.json"
        previous = json.loads(best_path.read_text()) if best_path.exists() else None
        if previous is None or fid < previous["fid"]:
            best_path.write_text(json.dumps({"fid": fid, "checkpoint": name}) + "\n")
            self._point("best-fid", name)

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
            path
            for path in self.directory.glob("epoch-[0-9][0-9][0-9][0-9]")
            if path.is_dir()
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
        for path in (
            item
            for item in checkpoints[: -self.keep_last_n]
            if item.name not in protected
        ):
            shutil.rmtree(path)

    def load(
        self,
        checkpoint: str | Path,
        model: nn.Module,
        ema: EMA,
        *,
        expected_architecture: str | None = None,
        expected_data_fingerprint: str | None = None,
        expected_validation_plan_hash: str | None = None,
        expected_representation_provenance: dict[str, Any] | None = None,
        expected_cache_provenance: dict[str, str] | None = None,
    ) -> ResumeState:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = self.directory / path
        metadata_path = path / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError("checkpoint predates CascadeDiT; old checkpoints are unsupported")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("checkpoint schema is incompatible; migration is unsupported")
        from .config import load_checkpoint_config

        config = load_checkpoint_config(path / "config.json")
        state = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
        if state.get("schema_version") != TRAINING_STATE_SCHEMA_VERSION:
            raise ValueError("checkpoint training-state schema is incompatible")
        _validate_provenance(
            state.get("architecture", ""),
            state.get("representation_provenance", {}),
            state.get("cache_provenance", {}),
        )
        for key in (
            "architecture",
            "data_fingerprint",
            "validation_plan_hash",
            "representation_provenance",
            "cache_provenance",
        ):
            if metadata.get(key) != state.get(key):
                raise ValueError(f"checkpoint {key} provenance is inconsistent")
        if config.model.architecture != state["architecture"]:
            raise ValueError("checkpoint architecture and config are inconsistent")
        checks = (
            ("architecture", expected_architecture),
            ("data_fingerprint", expected_data_fingerprint),
            ("validation_plan_hash", expected_validation_plan_hash),
            ("representation_provenance", expected_representation_provenance),
            ("cache_provenance", expected_cache_provenance),
        )
        for key, expected in checks:
            if expected is not None and state[key] != expected:
                raise ValueError(f"checkpoint {key} mismatch")
        model.load_state_dict(load_file(str(path / "model.safetensors")))
        ema.load_state_dict(load_file(str(path / "ema.safetensors")))
        return ResumeState(**state)

    def export_ema(self, ema: EMA, path: str | Path) -> None:
        destination = Path(path)
        temporary = destination.with_suffix(
            destination.suffix + f".partial-{os.getpid()}"
        )
        save_file(_contiguous(ema.state_dict()), str(temporary))
        os.replace(temporary, destination)
