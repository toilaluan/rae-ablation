from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from .config import ExperimentConfig
from .reg import FeatureTargets

Component = Literal["reg", "repa"]

CACHE_SCHEMA_VERSION = 3
VALUES_FILE = "values.bf16"
COMPLETION_FILE = "completion.u8"
MANIFEST_FILE = "manifest.json"


def _json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _dataset_spec(config: ExperimentConfig, dataset_size: int) -> dict[str, Any]:
    return {
        "repository": config.data.repository,
        "revision": config.data.revision,
        "split": config.data.split,
        "max_samples": config.data.max_samples,
        "dataset_size": dataset_size,
        "image_size": config.model.image_size,
        "preprocessing": [
            "PIL.convert(RGB)",
            f"PIL.resize({config.model.image_size}x{config.model.image_size}, bicubic)",
            "uint8_to_float32_[0,1]",
            "horizontal_flip_disabled",
        ],
    }


def component_spec(
    config: ExperimentConfig,
    dataset_size: int,
    component: Component,
) -> dict[str, Any]:
    """Build a minimal component identity plus non-fingerprinted diagnostics."""
    if dataset_size < 1:
        raise ValueError("feature-cache dataset size must be positive")
    features = config.model.features
    encoder = features.encoder
    if component == "reg":
        reg = features.reg
        target = {
            "feature_schema": "dino-normalized-cls-and-optional-pooled-spatial-v2",
            "token_mode": reg.token_mode,
            "pool_size": reg.pool_size if reg.token_mode == "cls_spatial" else None,
            "num_tokens": reg.num_tokens,
            "dtype": "bfloat16",
        }
    elif component == "repa":
        target = {
            "feature_schema": "dino-resized-normalized-patches-v2",
            "target_grid": config.model.grid_size,
            "num_tokens": config.model.grid_size**2,
            "dtype": "bfloat16",
        }
    else:
        raise ValueError("feature-cache component must be reg or repa")
    fingerprint_inputs = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "component": component,
        "dataset": _dataset_spec(config, dataset_size),
        "encoder": {
            "model_name": encoder.model_name,
            "feature_dim": encoder.feature_dim,
            "register_tokens": encoder.register_tokens,
        },
        "target": target,
    }
    environment: dict[str, str] = {"torch": torch.__version__}
    try:
        environment["timm"] = importlib.metadata.version("timm")
    except importlib.metadata.PackageNotFoundError:
        environment["timm"] = "not-installed"
    return {
        "fingerprint_inputs": fingerprint_inputs,
        "environment": environment,
    }


def component_fingerprint(spec: dict[str, Any]) -> str:
    inputs = spec.get("fingerprint_inputs")
    if not isinstance(inputs, dict):
        raise ValueError("feature-cache spec has no fingerprint inputs")
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def component_path(
    config: ExperimentConfig,
    dataset_size: int,
    component: Component,
) -> Path:
    spec = component_spec(config, dataset_size, component)
    return (
        Path(config.feature_cache.directory)
        / component
        / component_fingerprint(spec)
    )


def _shape(spec: dict[str, Any]) -> tuple[int, int, int]:
    inputs = spec["fingerprint_inputs"]
    return (
        int(inputs["dataset"]["dataset_size"]),
        int(inputs["target"]["num_tokens"]),
        int(inputs["encoder"]["feature_dim"]),
    )


def _manifest(spec: dict[str, Any], *, complete: bool) -> dict[str, Any]:
    inputs = spec["fingerprint_inputs"]
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "component": inputs["component"],
        "fingerprint": component_fingerprint(spec),
        "complete": complete,
        "fingerprint_inputs": inputs,
        "environment": spec["environment"],
        "shape": list(_shape(spec)),
        "values_file": VALUES_FILE,
        "completion_file": COMPLETION_FILE,
    }


def _load_manifest(directory: Path) -> dict[str, Any]:
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"feature-cache manifest is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("feature-cache manifest must be a JSON object")
    if value.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(
            "feature-cache manifest uses an incompatible schema; old caches must be rebuilt"
        )
    return value


def _validate_manifest(
    directory: Path,
    expected_spec: dict[str, Any],
    *,
    require_complete: bool,
) -> dict[str, Any]:
    manifest = _load_manifest(directory)
    expected_fingerprint = component_fingerprint(expected_spec)
    if (
        manifest.get("fingerprint") != expected_fingerprint
        or manifest.get("fingerprint_inputs")
        != expected_spec["fingerprint_inputs"]
        or manifest.get("shape") != list(_shape(expected_spec))
        or manifest.get("values_file") != VALUES_FILE
        or manifest.get("completion_file") != COMPLETION_FILE
    ):
        raise ValueError(f"feature-cache manifest is incompatible: {directory}")
    if require_complete and not manifest.get("complete"):
        raise ValueError(f"feature cache is incomplete: {directory}")
    shape = _shape(expected_spec)
    values_path = directory / VALUES_FILE
    completion_path = directory / COMPLETION_FILE
    if not values_path.is_file() or values_path.stat().st_size != int(
        np.prod(shape, dtype=np.int64)
    ) * 2:
        raise ValueError(f"feature-cache values file has the wrong size: {values_path}")
    if not completion_path.is_file() or completion_path.stat().st_size != shape[0]:
        raise ValueError(
            f"feature-cache completion file has the wrong size: {completion_path}"
        )
    completion = np.memmap(
        completion_path, mode="r", dtype=np.uint8, shape=(shape[0],)
    )
    if require_complete and np.any(completion != 1):
        raise ValueError(f"feature-cache completion bitmap is incomplete: {directory}")
    return manifest


@contextmanager
def cache_build_lock(directory: str | Path) -> Iterator[None]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".build.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another process is building feature cache {directory}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class FeatureComponentCacheWriter:
    def __init__(self, directory: str | Path, spec: dict[str, Any]) -> None:
        self.directory = Path(directory)
        self.spec = spec
        self.shape = _shape(spec)
        _validate_manifest(self.directory, spec, require_complete=False)
        self.values = np.memmap(
            self.directory / VALUES_FILE,
            mode="r+",
            dtype=np.uint16,
            shape=self.shape,
        )
        self.completion = np.memmap(
            self.directory / COMPLETION_FILE,
            mode="r+",
            dtype=np.uint8,
            shape=(self.shape[0],),
        )

    @classmethod
    def initialize(cls, directory: str | Path, spec: dict[str, Any]) -> bool:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / MANIFEST_FILE
        if manifest_path.exists():
            manifest = _validate_manifest(directory, spec, require_complete=False)
            return bool(manifest["complete"])
        shape = _shape(spec)
        values = np.memmap(
            directory / VALUES_FILE, mode="w+", dtype=np.uint16, shape=shape
        )
        completion = np.memmap(
            directory / COMPLETION_FILE,
            mode="w+",
            dtype=np.uint8,
            shape=(shape[0],),
        )
        completion[:] = 0
        values.flush()
        completion.flush()
        _json_atomic(manifest_path, _manifest(spec, complete=False))
        return False

    def pending_indices(self, rank: int = 0, world_size: int = 1) -> np.ndarray:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be within world size")
        pending = np.flatnonzero(np.asarray(self.completion) == 0)
        return pending[rank::world_size]

    def write(self, indices: Tensor, values: Tensor) -> None:
        positions = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if positions.ndim != 1:
            raise ValueError("feature-cache indices must have shape [B]")
        if np.any(positions < 0) or np.any(positions >= self.shape[0]):
            raise IndexError("feature-cache index is out of range")
        expected = (len(positions), self.shape[1], self.shape[2])
        if tuple(values.shape) != expected:
            raise ValueError(f"cached component values must have shape {expected}")
        bits = (
            values.detach()
            .to(device="cpu", dtype=torch.bfloat16)
            .contiguous()
            .view(torch.uint16)
            .numpy()
        )
        self.values[positions] = bits
        self.values.flush()
        self.completion[positions] = 1
        self.completion.flush()

    def finalize(self) -> None:
        if np.any(np.asarray(self.completion) != 1):
            raise ValueError("cannot finalize an incomplete feature cache")
        _json_atomic(
            self.directory / MANIFEST_FILE,
            _manifest(self.spec, complete=True),
        )


class FeatureComponentCache:
    def __init__(
        self,
        config: ExperimentConfig,
        dataset_size: int,
        component: Component,
    ) -> None:
        self.component = component
        self.spec = component_spec(config, dataset_size, component)
        self.directory = component_path(config, dataset_size, component)
        self.manifest = _validate_manifest(
            self.directory, self.spec, require_complete=True
        )
        self.fingerprint = str(self.manifest["fingerprint"])
        self.shape = _shape(self.spec)
        self._values = np.memmap(
            self.directory / VALUES_FILE,
            mode="r",
            dtype=np.uint16,
            shape=self.shape,
        )

    def lookup(self, indices: Tensor) -> Tensor:
        positions = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if positions.ndim != 1:
            raise ValueError("feature-cache indices must have shape [B]")
        if np.any(positions < 0) or np.any(positions >= self.shape[0]):
            raise IndexError("feature-cache index is out of range")
        bits = np.asarray(self._values[positions]).copy()
        return torch.from_numpy(bits).view(torch.bfloat16)


class FeatureCache:
    """Read whichever independent feature components are active."""

    def __init__(self, config: ExperimentConfig, dataset_size: int) -> None:
        features = config.model.features
        self.reg = (
            FeatureComponentCache(config, dataset_size, "reg")
            if features.reg.enabled
            else None
        )
        self.repa = (
            FeatureComponentCache(config, dataset_size, "repa")
            if features.repa.enabled
            else None
        )

    @property
    def fingerprints(self) -> dict[str, str]:
        return {
            component: cache.fingerprint
            for component, cache in (("reg", self.reg), ("repa", self.repa))
            if cache is not None
        }

    def lookup(self, indices: Tensor) -> FeatureTargets:
        return FeatureTargets(
            self.reg.lookup(indices) if self.reg is not None else None,
            self.repa.lookup(indices) if self.repa is not None else None,
        )
