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

ENDPOINT_CACHE_SCHEMA_VERSION = 1
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
            "target": "normalized-block2-patch-tokens",
            "semantic": "normalized-block12-unpooled-patch-tokens",
            "shape": list(config.vision.token_shape),
            "dtype": "bfloat16",
        },
    }


def endpoint_cache_fingerprint(spec: dict[str, Any]) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def endpoint_cache_path(config: ExperimentConfig, spec: dict[str, Any]) -> Path:
    return Path(config.vision.cache.directory) / endpoint_cache_fingerprint(spec)


def _shape(spec: dict[str, Any]) -> tuple[int, ...]:
    return (
        int(spec["dataset"]["dataset_size"]),
        *(int(value) for value in spec["tensors"]["shape"]),
    )


def required_endpoint_cache_bytes(spec: dict[str, Any]) -> int:
    shape = _shape(spec)
    values = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.uint16).itemsize
    return 2 * values + shape[0]


def preflight_endpoint_cache_capacity(
    directory: str | Path,
    spec: dict[str, Any],
    path: Path | None = None,
) -> int:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = path or root
    shape = _shape(spec)
    value_bytes = int(np.prod(shape, dtype=np.int64)) * 2
    expected = (
        (TARGET_FILE, value_bytes),
        (SEMANTIC_FILE, value_bytes),
        (COMPLETION_FILE, shape[0]),
    )
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
        "shape": list(_shape(spec)),
        "target_file": TARGET_FILE,
        "semantic_file": SEMANTIC_FILE,
        "completion_file": COMPLETION_FILE,
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
    if (
        manifest.get("fingerprint") != endpoint_cache_fingerprint(spec)
        or manifest.get("spec") != spec
        or manifest.get("shape") != list(_shape(spec))
    ):
        raise ValueError(f"endpoint-cache manifest is incompatible: {directory}")
    shape = _shape(spec)
    value_bytes = int(np.prod(shape, dtype=np.int64)) * 2
    for filename, size in (
        (TARGET_FILE, value_bytes),
        (SEMANTIC_FILE, value_bytes),
        (COMPLETION_FILE, shape[0]),
    ):
        candidate = directory / filename
        if not candidate.is_file() or candidate.stat().st_size != size:
            raise ValueError(f"endpoint-cache file has the wrong size: {candidate}")
    if require_complete:
        if not manifest.get("complete"):
            raise ValueError(f"endpoint cache is incomplete: {directory}")
        completion = np.memmap(
            directory / COMPLETION_FILE, mode="r", dtype=np.uint8, shape=(shape[0],)
        )
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
        self.shape = _shape(spec)
        _validate_manifest(self.directory, spec, require_complete=False)
        self.target = np.memmap(
            self.directory / TARGET_FILE, mode="r+", dtype=np.uint16, shape=self.shape
        )
        self.semantic = np.memmap(
            self.directory / SEMANTIC_FILE,
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
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        manifest_path = destination / MANIFEST_FILE
        if manifest_path.exists():
            return bool(
                _validate_manifest(destination, spec, require_complete=False)["complete"]
            )
        shape = _shape(spec)
        target = np.memmap(
            destination / TARGET_FILE, mode="w+", dtype=np.uint16, shape=shape
        )
        semantic = np.memmap(
            destination / SEMANTIC_FILE, mode="w+", dtype=np.uint16, shape=shape
        )
        completion = np.memmap(
            destination / COMPLETION_FILE,
            mode="w+",
            dtype=np.uint8,
            shape=(shape[0],),
        )
        completion[:] = 0
        target.flush()
        semantic.flush()
        completion.flush()
        _json_atomic(manifest_path, _manifest(spec, False))
        return False

    def pending_indices(self, rank: int = 0, world_size: int = 1) -> np.ndarray:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be within world size")
        return np.flatnonzero(np.asarray(self.completion) == 0)[rank::world_size]

    def write(self, indices: Tensor, target: Tensor, semantic: Tensor) -> None:
        positions = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        expected = (len(positions), *self.shape[1:])
        if positions.ndim != 1 or tuple(target.shape) != expected or semantic.shape != target.shape:
            raise ValueError(f"cached endpoint tensors must have shape {expected}")
        if np.any(positions < 0) or np.any(positions >= self.shape[0]):
            raise IndexError("endpoint-cache index is out of range")
        target_bits = (
            target.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
        )
        semantic_bits = (
            semantic.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
        )
        self.target[positions] = target_bits
        self.semantic[positions] = semantic_bits
        self.target.flush()
        self.semantic.flush()
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
                        torch.empty(
                            (batch_size, *writer.shape[1:]),
                            dtype=torch.bfloat16,
                            pin_memory=True,
                        ),
                        torch.empty(
                            (batch_size, *writer.shape[1:]),
                            dtype=torch.bfloat16,
                            pin_memory=True,
                        ),
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
        self.shape = _shape(self.spec)
        self.target = np.memmap(
            self.directory / TARGET_FILE, mode="r", dtype=np.uint16, shape=self.shape
        )
        self.semantic = np.memmap(
            self.directory / SEMANTIC_FILE, mode="r", dtype=np.uint16, shape=self.shape
        )

    @staticmethod
    def _lookup(values: np.memmap, indices: Tensor) -> Tensor:
        positions = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= values.shape[0]):
            raise IndexError("endpoint-cache index is out of range")
        bits = np.asarray(values[positions]).copy()
        return torch.from_numpy(bits).view(torch.bfloat16)

    def lookup(self, indices: Tensor) -> tuple[Tensor, Tensor]:
        return self._lookup(self.target, indices), self._lookup(self.semantic, indices)
