from __future__ import annotations

import fcntl
import hashlib
import json
import os
import queue
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .config import ExperimentConfig
from .data import preprocessing_metadata
from .vision import vision_fingerprint

ENDPOINT_CACHE_SCHEMA_VERSION = 3
TARGET_FILE = "target.bf16"
SEMANTIC_FILE = "semantic.bf16"
COMPLETION_FILE = "completion.u8"
MANIFEST_FILE = "manifest.json"


def _json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def endpoint_cache_spec(
    config: ExperimentConfig,
    dataset_size: int,
    latent_statistics_sha256: str,
) -> dict[str, Any]:
    if dataset_size < 1 or not latent_statistics_sha256:
        raise ValueError("endpoint cache requires a dataset and latent statistics")
    return {
        "schema_version": ENDPOINT_CACHE_SCHEMA_VERSION,
        "dataset": {
            **preprocessing_metadata(config.data, config.vision.image_size),
            "dataset_size": dataset_size,
        },
        "vision_fingerprint": vision_fingerprint(config.vision),
        "latent_statistics_sha256": latent_statistics_sha256,
        "tensors": {
            "target": {
                "description": "normalized-raw-block2-patch-states",
                "shape": list(config.vision.token_shape),
            },
            "semantic": {
                "description": "normalized-finalnorm-block12-patch-states",
                "shape": list(config.vision.token_shape),
            },
            "dtype": "bfloat16",
        },
    }


def endpoint_cache_fingerprint(spec: dict[str, Any]) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def endpoint_cache_path(config: ExperimentConfig, spec: dict[str, Any]) -> Path:
    return Path(config.vision.cache.directory) / endpoint_cache_fingerprint(spec)


def _shape(spec: dict[str, Any], name: str) -> tuple[int, ...]:
    return (
        int(spec["dataset"]["dataset_size"]),
        *(int(value) for value in spec["tensors"][name]["shape"]),
    )


def _files(spec: dict[str, Any]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return (
        (TARGET_FILE, _shape(spec, "target")),
        (SEMANTIC_FILE, _shape(spec, "semantic")),
    )


def required_endpoint_cache_bytes(spec: dict[str, Any]) -> int:
    values = sum(int(np.prod(shape, dtype=np.int64)) * 2 for _, shape in _files(spec))
    return values + int(spec["dataset"]["dataset_size"])


def preflight_endpoint_cache_capacity(
    directory: str | Path,
    spec: dict[str, Any],
    path: Path | None = None,
) -> int:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = path or root
    expected = [
        (filename, int(np.prod(shape, dtype=np.int64)) * 2)
        for filename, shape in _files(spec)
    ]
    expected.append((COMPLETION_FILE, int(spec["dataset"]["dataset_size"])))
    needed = 0
    for filename, size in expected:
        candidate = target / filename
        allocated = min(candidate.stat().st_blocks * 512, size) if candidate.is_file() else 0
        needed += max(size - allocated, 0)
    available = shutil.disk_usage(root).free
    if needed > available:
        raise OSError(
            "insufficient disk capacity for endpoint cache: "
            f"{needed / 2**30:.1f} GiB required, {available / 2**30:.1f} GiB available"
        )
    return needed


def _manifest(spec: dict[str, Any], complete: bool) -> dict[str, Any]:
    return {
        "schema_version": ENDPOINT_CACHE_SCHEMA_VERSION,
        "fingerprint": endpoint_cache_fingerprint(spec),
        "complete": complete,
        "spec": spec,
        "shapes": {name: list(_shape(spec, name)) for name in ("target", "semantic")},
        "files": {
            "target": TARGET_FILE,
            "semantic": SEMANTIC_FILE,
            "completion": COMPLETION_FILE,
        },
    }


def _validate_manifest(
    directory: Path,
    spec: dict[str, Any],
    *,
    require_complete: bool,
) -> dict[str, Any]:
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"endpoint-cache manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != ENDPOINT_CACHE_SCHEMA_VERSION:
        raise ValueError("endpoint-cache manifest uses an incompatible schema")
    if manifest.get("fingerprint") != endpoint_cache_fingerprint(spec) or manifest.get("spec") != spec:
        raise ValueError(f"endpoint-cache manifest is incompatible: {directory}")
    for filename, shape in _files(spec):
        size = int(np.prod(shape, dtype=np.int64)) * 2
        candidate = directory / filename
        if not candidate.is_file() or candidate.stat().st_size != size:
            raise ValueError(f"endpoint-cache file has the wrong size: {candidate}")
    count = int(spec["dataset"]["dataset_size"])
    completion_path = directory / COMPLETION_FILE
    if not completion_path.is_file() or completion_path.stat().st_size != count:
        raise ValueError(f"endpoint-cache file has the wrong size: {completion_path}")
    if require_complete:
        if not manifest.get("complete"):
            raise ValueError(f"endpoint cache is incomplete: {directory}")
        completion = np.memmap(completion_path, mode="r", dtype=np.uint8, shape=(count,))
        if np.any(completion != 1):
            raise ValueError(f"endpoint-cache completion bitmap is incomplete: {directory}")
    return manifest


@contextmanager
def endpoint_cache_build_lock(directory: str | Path) -> Iterator[None]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".build.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process is building endpoint cache {root}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class EndpointCacheWriter:
    def __init__(self, directory: str | Path, spec: dict[str, Any]) -> None:
        self.directory = Path(directory)
        self.spec = spec
        _validate_manifest(self.directory, spec, require_complete=False)
        self.target_shape = _shape(spec, "target")
        self.semantic_shape = _shape(spec, "semantic")
        self.target = np.memmap(
            self.directory / TARGET_FILE, mode="r+", dtype=np.uint16, shape=self.target_shape
        )
        self.semantic = np.memmap(
            self.directory / SEMANTIC_FILE,
            mode="r+",
            dtype=np.uint16,
            shape=self.semantic_shape,
        )
        self.completion = np.memmap(
            self.directory / COMPLETION_FILE,
            mode="r+",
            dtype=np.uint8,
            shape=(self.target_shape[0],),
        )

    @classmethod
    def initialize(cls, directory: str | Path, spec: dict[str, Any]) -> bool:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        manifest_path = destination / MANIFEST_FILE
        if manifest_path.exists():
            return bool(_validate_manifest(destination, spec, require_complete=False)["complete"])
        arrays = []
        for filename, shape in _files(spec):
            arrays.append(np.memmap(destination / filename, mode="w+", dtype=np.uint16, shape=shape))
        count = int(spec["dataset"]["dataset_size"])
        completion = np.memmap(
            destination / COMPLETION_FILE, mode="w+", dtype=np.uint8, shape=(count,)
        )
        completion[:] = 0
        for value in (*arrays, completion):
            value.flush()
        _json_atomic(manifest_path, _manifest(spec, False))
        return False

    def pending_indices(self, rank: int = 0, world_size: int = 1) -> np.ndarray:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be within world size")
        return np.flatnonzero(np.asarray(self.completion) == 0)[rank::world_size]

    @staticmethod
    def _bits(value: Tensor) -> np.ndarray:
        return value.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint16).numpy()

    def write(self, indices: Tensor, target: Tensor, semantic: Tensor) -> None:
        positions = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        expected = {
            "target": (len(positions), *self.target_shape[1:]),
            "semantic": (len(positions), *self.semantic_shape[1:]),
        }
        if positions.ndim != 1 or any(
            tuple(value.shape) != expected[name]
            for name, value in (("target", target), ("semantic", semantic))
        ):
            raise ValueError(f"cached endpoint tensors must have shapes {expected}")
        if np.any(positions < 0) or np.any(positions >= self.target_shape[0]):
            raise IndexError("endpoint-cache index is out of range")
        self.target[positions] = self._bits(target)
        self.semantic[positions] = self._bits(semantic)
        for value in (self.target, self.semantic):
            value.flush()
        self.completion[positions] = 1
        self.completion.flush()

    def finalize(self) -> None:
        if np.any(np.asarray(self.completion) != 1):
            raise ValueError("cannot finalize an incomplete endpoint cache")
        _json_atomic(self.directory / MANIFEST_FILE, _manifest(self.spec, True))
        _validate_manifest(self.directory, self.spec, require_complete=True)


@dataclass
class _HostBuffer:
    indices: Tensor
    target: Tensor
    semantic: Tensor


@dataclass
class _WriteJob:
    buffer: _HostBuffer
    count: int
    event: torch.cuda.Event
    sources: tuple[Tensor, Tensor, Tensor]


class AsyncEndpointCacheWriter:
    def __init__(
        self,
        writer: EndpointCacheWriter,
        batch_size: int,
        device: torch.device | str,
    ) -> None:
        if batch_size < 1:
            raise ValueError("cache pipeline batch size must be positive")
        self.writer = writer
        self.device = torch.device(device)
        self.asynchronous = self.device.type == "cuda"
        self._error: BaseException | None = None
        self._closed = False
        self._free: queue.Queue[_HostBuffer] = queue.Queue(maxsize=2)
        self._jobs: queue.Queue[_WriteJob | None] = queue.Queue(maxsize=2)
        self._copy_stream: torch.cuda.Stream | None = None
        self._thread: threading.Thread | None = None
        if self.asynchronous:
            self._copy_stream = torch.cuda.Stream(device=self.device)
            for _ in range(2):
                self._free.put(
                    _HostBuffer(
                        torch.empty(batch_size, dtype=torch.int64, pin_memory=True),
                        torch.empty((batch_size, *writer.target_shape[1:]), dtype=torch.bfloat16, pin_memory=True),
                        torch.empty((batch_size, *writer.semantic_shape[1:]), dtype=torch.bfloat16, pin_memory=True),
                    )
                )
            self._thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._thread.start()

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("endpoint-cache writer failed") from self._error

    def acquire_buffer(self) -> _HostBuffer | None:
        self._raise_error()
        return self._free.get() if self.asynchronous else None

    def release_buffer(self, buffer: _HostBuffer | None) -> None:
        if buffer is not None:
            self._free.put(buffer)

    def submit(
        self,
        indices: Tensor,
        target: Tensor,
        semantic: Tensor,
        *,
        buffer: _HostBuffer | None = None,
    ) -> None:
        count = indices.numel()
        if not self.asynchronous:
            self.writer.write(indices, target, semantic)
            return
        assert buffer is not None and self._copy_stream is not None
        current = torch.cuda.current_stream(self.device)
        self._copy_stream.wait_stream(current)
        event = torch.cuda.Event()
        with torch.cuda.stream(self._copy_stream):
            buffer.indices[:count].copy_(indices, non_blocking=True)
            buffer.target[:count].copy_(target, non_blocking=True)
            buffer.semantic[:count].copy_(semantic, non_blocking=True)
            event.record()
        self._jobs.put(_WriteJob(buffer, count, event, (indices, target, semantic)))

    def _writer_loop(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                job.event.synchronize()
                if self._error is None:
                    self.writer.write(
                        job.buffer.indices[: job.count],
                        job.buffer.target[: job.count],
                        job.buffer.semantic[: job.count],
                    )
                self._free.put(job.buffer)
            except BaseException as error:
                if self._error is None:
                    self._error = error
                if job is not None:
                    self._free.put(job.buffer)
            finally:
                self._jobs.task_done()

    def drain(self) -> None:
        if self.asynchronous:
            self._jobs.join()
        self._raise_error()

    def close(self) -> None:
        if self._closed:
            self._raise_error()
            return
        if self.asynchronous:
            self._jobs.put(None)
            assert self._thread is not None
            self._thread.join()
        self._closed = True
        self._raise_error()


class EndpointCache:
    def __init__(
        self,
        config: ExperimentConfig,
        dataset_size: int,
        latent_statistics_sha256: str,
    ) -> None:
        self.spec = endpoint_cache_spec(config, dataset_size, latent_statistics_sha256)
        self.directory = endpoint_cache_path(config, self.spec)
        self.manifest = _validate_manifest(self.directory, self.spec, require_complete=True)
        self.fingerprint = str(self.manifest["fingerprint"])
        self.target = np.memmap(
            self.directory / TARGET_FILE, mode="r", dtype=np.uint16, shape=_shape(self.spec, "target")
        )
        self.semantic = np.memmap(
            self.directory / SEMANTIC_FILE, mode="r", dtype=np.uint16, shape=_shape(self.spec, "semantic")
        )

    @staticmethod
    def _lookup(values: np.memmap, indices: Tensor) -> Tensor:
        positions = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= values.shape[0]):
            raise IndexError("endpoint-cache index is out of range")
        return torch.from_numpy(np.asarray(values[positions]).copy()).view(torch.bfloat16)

    def lookup(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self._lookup(self.target, indices),
            self._lookup(self.semantic, indices),
        )
