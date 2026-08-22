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

CACHE_SCHEMA_VERSION = 1
VALUES_FILE = "values.bf16"
COMPLETION_FILE = "completion.u8"
MANIFEST_FILE = "manifest.json"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def cache_fingerprint(spec: dict[str, Any]) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_shape(spec: dict[str, Any]) -> tuple[int, ...]:
    return (
        int(spec["dataset"]["dataset_size"]),
        *(int(value) for value in spec["tensor"]["shape"]),
    )


def required_cache_bytes(spec: dict[str, Any]) -> int:
    shape = cache_shape(spec)
    return int(np.prod(shape, dtype=np.int64)) * 2 + shape[0]


def cache_path(root: str | Path, spec: dict[str, Any]) -> Path:
    return Path(root) / cache_fingerprint(spec)


def preflight_cache_capacity(
    root: str | Path,
    spec: dict[str, Any],
    path: str | Path | None = None,
) -> int:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    target = Path(path) if path is not None else root_path
    shape = cache_shape(spec)
    value_bytes = int(np.prod(shape, dtype=np.int64)) * 2
    expected = ((VALUES_FILE, value_bytes), (COMPLETION_FILE, shape[0]))
    needed = 0
    for filename, size in expected:
        candidate = target / filename
        allocated = min(candidate.stat().st_blocks * 512, size) if candidate.is_file() else 0
        needed += max(size - allocated, 0)
    available = shutil.disk_usage(root_path).free
    if needed > available:
        raise OSError(
            f"insufficient cache capacity: {needed / 2**30:.1f} GiB required, "
            f"{available / 2**30:.1f} GiB available"
        )
    return needed


def manifest_for(spec: dict[str, Any], complete: bool) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": cache_fingerprint(spec),
        "complete": complete,
        "spec": spec,
        "shape": list(cache_shape(spec)),
        "values_file": VALUES_FILE,
        "completion_file": COMPLETION_FILE,
    }


def validate_manifest(
    directory: str | Path,
    spec: dict[str, Any],
    *,
    require_complete: bool,
) -> dict[str, Any]:
    path = Path(directory)
    manifest_path = path / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cache manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"cache manifest is corrupt: {manifest_path}") from error
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("cache manifest uses an incompatible schema")
    if (
        manifest.get("fingerprint") != cache_fingerprint(spec)
        or manifest.get("spec") != spec
        or manifest.get("shape") != list(cache_shape(spec))
    ):
        raise ValueError(f"cache manifest is incompatible: {path}")
    shape = cache_shape(spec)
    expected = (
        (VALUES_FILE, int(np.prod(shape, dtype=np.int64)) * 2),
        (COMPLETION_FILE, shape[0]),
    )
    for filename, size in expected:
        candidate = path / filename
        if not candidate.is_file() or candidate.stat().st_size != size:
            raise ValueError(f"cache file has the wrong size: {candidate}")
    completion = np.memmap(
        path / COMPLETION_FILE, mode="r", dtype=np.uint8, shape=(shape[0],)
    )
    if np.any((completion != 0) & (completion != 1)):
        raise ValueError("cache completion bitmap is corrupt")
    if require_complete and (
        not manifest.get("complete") or np.any(completion != 1)
    ):
        raise ValueError(f"cache is incomplete: {path}")
    return manifest


@contextmanager
def cache_build_lock(directory: str | Path) -> Iterator[None]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".build.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process is building cache {root}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class RepresentationCacheWriter:
    def __init__(self, directory: str | Path, spec: dict[str, Any]) -> None:
        self.directory = Path(directory)
        self.spec = spec
        self.shape = cache_shape(spec)
        validate_manifest(self.directory, spec, require_complete=False)
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
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        manifest_path = destination / MANIFEST_FILE
        if manifest_path.exists():
            manifest = validate_manifest(destination, spec, require_complete=False)
            complete = bool(manifest["complete"])
            if complete:
                validate_manifest(destination, spec, require_complete=True)
            return complete
        shape = cache_shape(spec)
        values = np.memmap(
            destination / VALUES_FILE, mode="w+", dtype=np.uint16, shape=shape
        )
        completion = np.memmap(
            destination / COMPLETION_FILE,
            mode="w+",
            dtype=np.uint8,
            shape=(shape[0],),
        )
        completion[:] = 0
        values.flush()
        completion.flush()
        atomic_json(manifest_path, manifest_for(spec, False))
        return False

    def pending_indices(self, rank: int = 0, world_size: int = 1) -> np.ndarray:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be within world size")
        return np.flatnonzero(np.asarray(self.completion) == 0)[rank::world_size]

    def write(self, indices: Tensor, values: Tensor) -> None:
        positions = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        expected = (len(positions), *self.shape[1:])
        if positions.ndim != 1 or tuple(values.shape) != expected:
            raise ValueError(f"cached tensors must have shape {expected}")
        if np.any(positions < 0) or np.any(positions >= self.shape[0]):
            raise IndexError("cache index is out of range")
        if positions.size != np.unique(positions).size:
            raise ValueError("cache write indices must be unique")
        bits = (
            values.detach()
            .cpu()
            .to(torch.bfloat16)
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
            raise ValueError("cannot finalize an incomplete cache")
        atomic_json(self.directory / MANIFEST_FILE, manifest_for(self.spec, True))
        validate_manifest(self.directory, self.spec, require_complete=True)


@dataclass
class HostBuffer:
    indices: Tensor
    values: Tensor


@dataclass
class _WriteJob:
    buffer: HostBuffer
    count: int
    event: torch.cuda.Event
    sources: tuple[Tensor, Tensor]


class AsyncRepresentationCacheWriter:
    def __init__(
        self,
        writer: RepresentationCacheWriter,
        batch_size: int,
        device: torch.device | str,
        *,
        enabled: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("cache pipeline batch size must be positive")
        self.writer = writer
        self.device = torch.device(device)
        self.asynchronous = enabled and self.device.type == "cuda"
        self._error: BaseException | None = None
        self._closed = False
        self._free: queue.Queue[HostBuffer] = queue.Queue(maxsize=2)
        self._jobs: queue.Queue[_WriteJob | None] = queue.Queue(maxsize=2)
        self._copy_stream: torch.cuda.Stream | None = None
        self._thread: threading.Thread | None = None
        if self.asynchronous:
            self._copy_stream = torch.cuda.Stream(device=self.device)
            for _ in range(2):
                self._free.put(
                    HostBuffer(
                        torch.empty(batch_size, dtype=torch.int64, pin_memory=True),
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
            raise RuntimeError("cache writer failed") from self._error

    def acquire_buffer(self) -> HostBuffer | None:
        self._raise_error()
        return self._free.get() if self.asynchronous else None

    def release_buffer(self, buffer: HostBuffer | None) -> None:
        if buffer is not None:
            self._free.put(buffer)

    def submit(
        self,
        indices: Tensor,
        values: Tensor,
        *,
        buffer: HostBuffer | None = None,
    ) -> None:
        count = indices.numel()
        if not self.asynchronous:
            self.writer.write(indices, values)
            return
        if buffer is None or self._copy_stream is None:
            raise ValueError("an acquired pinned buffer is required for async writes")
        current = torch.cuda.current_stream(self.device)
        self._copy_stream.wait_stream(current)
        event = torch.cuda.Event()
        with torch.cuda.stream(self._copy_stream):
            buffer.indices[:count].copy_(indices, non_blocking=True)
            buffer.values[:count].copy_(values, non_blocking=True)
            event.record()
        self._jobs.put(_WriteJob(buffer, count, event, (indices, values)))

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
                        job.buffer.values[: job.count],
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
            if self._thread is not None:
                self._thread.join()
        self._closed = True
        self._raise_error()


class RepresentationCache:
    def __init__(self, directory: str | Path, spec: dict[str, Any]) -> None:
        self.directory = Path(directory)
        self.spec = spec
        self.manifest = validate_manifest(
            self.directory, self.spec, require_complete=True
        )
        self.fingerprint = str(self.manifest["fingerprint"])
        self.shape = cache_shape(spec)
        self.values = np.memmap(
            self.directory / VALUES_FILE,
            mode="r",
            dtype=np.uint16,
            shape=self.shape,
        )

    def lookup(self, indices: Tensor) -> Tensor:
        positions = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if (
            positions.ndim != 1
            or np.any(positions < 0)
            or np.any(positions >= self.values.shape[0])
        ):
            raise IndexError("cache index is out of range")
        bits = np.asarray(self.values[positions]).copy()
        return torch.from_numpy(bits).view(torch.bfloat16)
