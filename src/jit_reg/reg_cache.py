from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .config import ExperimentConfig

CACHE_SCHEMA_VERSION = 1
TOKEN_FILE = "tokens.bf16"
COMPLETION_FILE = "completion.u8"
MANIFEST_FILE = "manifest.json"


def _json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def cache_spec(config: ExperimentConfig, dataset_size: int) -> dict[str, Any]:
    if dataset_size < 1:
        raise ValueError("REG cache dataset size must be positive")
    reg = config.model.reg
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": {
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
        },
        "target": {
            "model_name": reg.model_name,
            "feature_dim": reg.feature_dim,
            "register_tokens": reg.register_tokens,
            "token_mode": reg.token_mode,
            "pool_size": reg.pool_size if reg.token_mode == "cls_spatial" else None,
            "num_tokens": reg.num_tokens,
            "dtype": "bfloat16",
        },
        "environment": {
            "timm": importlib.metadata.version("timm"),
            "torch": torch.__version__,
        },
    }


def cache_fingerprint(spec: dict[str, Any]) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_path(config: ExperimentConfig, dataset_size: int) -> Path:
    spec = cache_spec(config, dataset_size)
    return Path(config.reg_cache.directory) / cache_fingerprint(spec)


def _shape(spec: dict[str, Any]) -> tuple[int, int, int]:
    dataset = spec["dataset"]
    target = spec["target"]
    return (
        int(dataset["dataset_size"]),
        int(target["num_tokens"]),
        int(target["feature_dim"]),
    )


def _manifest(spec: dict[str, Any], *, complete: bool) -> dict[str, Any]:
    return {
        "fingerprint": cache_fingerprint(spec),
        "complete": complete,
        "spec": spec,
        "shape": list(_shape(spec)),
        "token_file": TOKEN_FILE,
        "completion_file": COMPLETION_FILE,
    }


def _load_manifest(directory: Path) -> dict[str, Any]:
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"REG cache manifest is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("REG cache manifest must be a JSON object")
    return value


def _validate_manifest(
    directory: Path,
    expected_spec: dict[str, Any],
    *,
    require_complete: bool,
) -> dict[str, Any]:
    manifest = _load_manifest(directory)
    expected = _manifest(expected_spec, complete=bool(manifest.get("complete")))
    if manifest != expected:
        raise ValueError(f"REG cache manifest is incompatible: {directory}")
    if require_complete and not manifest["complete"]:
        raise ValueError(f"REG cache is incomplete: {directory}")
    shape = _shape(expected_spec)
    expected_token_bytes = int(np.prod(shape, dtype=np.int64)) * 2
    token_path = directory / TOKEN_FILE
    completion_path = directory / COMPLETION_FILE
    if not token_path.is_file() or token_path.stat().st_size != expected_token_bytes:
        raise ValueError(f"REG cache token file has the wrong size: {token_path}")
    if not completion_path.is_file() or completion_path.stat().st_size != shape[0]:
        raise ValueError(f"REG cache completion file has the wrong size: {completion_path}")
    completion = np.memmap(completion_path, mode="r", dtype=np.uint8, shape=(shape[0],))
    if require_complete and np.any(completion != 1):
        raise ValueError(f"REG cache completion bitmap is incomplete: {directory}")
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
            raise RuntimeError(f"another process is building REG cache {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class REGTokenCacheWriter:
    def __init__(self, directory: str | Path, spec: dict[str, Any]) -> None:
        self.directory = Path(directory)
        self.spec = spec
        self.shape = _shape(spec)
        _validate_manifest(self.directory, spec, require_complete=False)
        self.tokens = np.memmap(
            self.directory / TOKEN_FILE,
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
        tokens = np.memmap(
            directory / TOKEN_FILE,
            mode="w+",
            dtype=np.uint16,
            shape=shape,
        )
        completion = np.memmap(
            directory / COMPLETION_FILE,
            mode="w+",
            dtype=np.uint8,
            shape=(shape[0],),
        )
        completion[:] = 0
        tokens.flush()
        completion.flush()
        _json_atomic(manifest_path, _manifest(spec, complete=False))
        return False

    def pending_indices(self, rank: int = 0, world_size: int = 1) -> np.ndarray:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be within world size")
        pending = np.flatnonzero(np.asarray(self.completion) == 0)
        return pending[rank::world_size]

    def write(self, indices: Tensor, tokens: Tensor) -> None:
        positions = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        expected = (len(positions), self.shape[1], self.shape[2])
        if tuple(tokens.shape) != expected:
            raise ValueError(f"cached REG tokens must have shape {expected}")
        bits = (
            tokens.detach()
            .to(device="cpu", dtype=torch.bfloat16)
            .contiguous()
            .view(torch.uint16)
            .numpy()
        )
        self.tokens[positions] = bits
        self.tokens.flush()
        self.completion[positions] = 1
        self.completion.flush()

    def finalize(self) -> None:
        if np.any(np.asarray(self.completion) != 1):
            raise ValueError("cannot finalize an incomplete REG cache")
        _json_atomic(self.directory / MANIFEST_FILE, _manifest(self.spec, complete=True))


class REGTokenCache:
    def __init__(self, config: ExperimentConfig, dataset_size: int) -> None:
        self.spec = cache_spec(config, dataset_size)
        self.directory = cache_path(config, dataset_size)
        self.manifest = _validate_manifest(
            self.directory,
            self.spec,
            require_complete=True,
        )
        self.fingerprint = str(self.manifest["fingerprint"])
        self.shape = _shape(self.spec)
        self._tokens = np.memmap(
            self.directory / TOKEN_FILE,
            mode="r",
            dtype=np.uint16,
            shape=self.shape,
        )

    def lookup(self, indices: Tensor) -> Tensor:
        positions = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if positions.ndim != 1:
            raise ValueError("REG cache indices must have shape [B]")
        if np.any(positions < 0) or np.any(positions >= self.shape[0]):
            raise IndexError("REG cache index is out of range")
        bits = np.asarray(self._tokens[positions]).copy()
        return torch.from_numpy(bits).view(torch.bfloat16)
