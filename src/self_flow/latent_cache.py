from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import queue
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .config import ExperimentConfig
from .vae import LATENT_SCHEMA

LATENT_CACHE_SCHEMA_VERSION = 1
VALUES_FILE = "values.bf16"
COMPLETION_FILE = "completion.u8"
MANIFEST_FILE = "manifest.json"


def _json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def latent_cache_spec(config: ExperimentConfig, dataset_size: int) -> dict[str, Any]:
    if dataset_size < 1:
        raise ValueError("latent-cache dataset size must be positive")
    inputs = {
        "schema_version": LATENT_CACHE_SCHEMA_VERSION,
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
                "horizontal_flip_removed",
            ],
        },
        "vae": {
            "class": "diffusers.AutoencoderKLFlux2",
            "model_name": config.vae.model_name,
            "revision": config.vae.revision,
            "subfolder": config.vae.subfolder,
            "encoding_schema": LATENT_SCHEMA,
            "posterior": "mode",
        },
        "target": {
            "shape": list(config.model.input_shape),
            "dtype": "bfloat16",
        },
    }
    environment = {"torch": torch.__version__}
    try:
        environment["diffusers"] = importlib.metadata.version("diffusers")
    except importlib.metadata.PackageNotFoundError:
        environment["diffusers"] = "not-installed"
    return {"fingerprint_inputs": inputs, "environment": environment}


def latent_cache_fingerprint(spec: dict[str, Any]) -> str:
    inputs = spec.get("fingerprint_inputs")
    if not isinstance(inputs, dict):
        raise ValueError("latent-cache spec has no fingerprint inputs")
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def latent_cache_path(config: ExperimentConfig, dataset_size: int) -> Path:
    spec = latent_cache_spec(config, dataset_size)
    return Path(config.vae.cache.directory) / latent_cache_fingerprint(spec)


def _shape(spec: dict[str, Any]) -> tuple[int, ...]:
    inputs = spec["fingerprint_inputs"]
    return (
        int(inputs["dataset"]["dataset_size"]),
        *(int(value) for value in inputs["target"]["shape"]),
    )


def required_latent_cache_bytes(spec: dict[str, Any]) -> int:
    shape = _shape(spec)
    return int(np.prod(shape, dtype=np.int64)) * 2 + shape[0]


def preflight_latent_cache_capacity(
    directory: str | Path,
    spec: dict[str, Any],
    path: Path | None = None,
) -> int:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = path or root
    shape = _shape(spec)
    needed = 0
    for filename, expected in (
        (VALUES_FILE, int(np.prod(shape, dtype=np.int64)) * 2),
        (COMPLETION_FILE, shape[0]),
    ):
        file = target / filename
        allocated = min(file.stat().st_blocks * 512, expected) if file.is_file() else 0
        needed += max(expected - allocated, 0)
    available = shutil.disk_usage(root).free
    if needed > available:
        gib = 2**30
        raise OSError(
            "insufficient disk capacity for latent cache: "
            f"{needed / gib:.1f} GiB required, {available / gib:.1f} GiB available"
        )
    return needed


def _manifest(spec: dict[str, Any], *, complete: bool) -> dict[str, Any]:
    return {
        "schema_version": LATENT_CACHE_SCHEMA_VERSION,
        "fingerprint": latent_cache_fingerprint(spec),
        "complete": complete,
        "fingerprint_inputs": spec["fingerprint_inputs"],
        "environment": spec["environment"],
        "shape": list(_shape(spec)),
        "values_file": VALUES_FILE,
        "completion_file": COMPLETION_FILE,
    }


def _validate_manifest(
    directory: Path,
    spec: dict[str, Any],
    *,
    require_complete: bool,
) -> dict[str, Any]:
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"latent-cache manifest is missing: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != LATENT_CACHE_SCHEMA_VERSION:
        raise ValueError("latent-cache manifest is incompatible")
    if (
        manifest.get("fingerprint") != latent_cache_fingerprint(spec)
        or manifest.get("fingerprint_inputs") != spec["fingerprint_inputs"]
        or manifest.get("shape") != list(_shape(spec))
    ):
        raise ValueError(f"latent-cache manifest is incompatible: {directory}")
    if require_complete and not manifest.get("complete"):
        raise ValueError(f"latent cache is incomplete: {directory}")
    shape = _shape(spec)
    values = directory / VALUES_FILE
    completion = directory / COMPLETION_FILE
    if not values.is_file() or values.stat().st_size != int(np.prod(shape, dtype=np.int64)) * 2:
        raise ValueError(f"latent-cache values file has the wrong size: {values}")
    if not completion.is_file() or completion.stat().st_size != shape[0]:
        raise ValueError(f"latent-cache completion file has the wrong size: {completion}")
    if require_complete:
        bitmap = np.memmap(completion, mode="r", dtype=np.uint8, shape=(shape[0],))
        if np.any(bitmap != 1):
            raise ValueError(f"latent-cache completion bitmap is incomplete: {directory}")
    return manifest


@contextmanager
def latent_cache_build_lock(directory: str | Path) -> Iterator[None]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".build.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process is building latent cache {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class LatentCacheWriter:
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
        manifest = directory / MANIFEST_FILE
        if manifest.exists():
            return bool(_validate_manifest(directory, spec, require_complete=False)["complete"])
        shape = _shape(spec)
        values = np.memmap(directory / VALUES_FILE, mode="w+", dtype=np.uint16, shape=shape)
        completion = np.memmap(
            directory / COMPLETION_FILE,
            mode="w+",
            dtype=np.uint8,
            shape=(shape[0],),
        )
        completion[:] = 0
        values.flush()
        completion.flush()
        _json_atomic(manifest, _manifest(spec, complete=False))
        return False

    def pending_indices(self, rank: int = 0, world_size: int = 1) -> np.ndarray:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be within world size")
        return np.flatnonzero(np.asarray(self.completion) == 0)[rank::world_size]

    def write(self, indices: Tensor, values: Tensor) -> None:
        positions = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        expected = (len(positions), *self.shape[1:])
        if positions.ndim != 1 or tuple(values.shape) != expected:
            raise ValueError(f"cached latents must have shape {expected}")
        if np.any(positions < 0) or np.any(positions >= self.shape[0]):
            raise IndexError("latent-cache index is out of range")
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
            raise ValueError("cannot finalize an incomplete latent cache")
        _json_atomic(self.directory / MANIFEST_FILE, _manifest(self.spec, complete=True))


@dataclass(frozen=True)
class LatentCachePipelineMetrics:
    examples: int
    batches: int
    buffer_backpressure_seconds: float
    copy_seconds: float
    write_flush_seconds: float
    total_seconds: float

    @property
    def images_per_second(self) -> float:
        return self.examples / max(self.total_seconds, 1e-9)


@dataclass
class _HostBuffer:
    indices: Tensor
    values: Tensor


@dataclass
class _WriteJob:
    buffer: _HostBuffer
    count: int
    sources: tuple[Tensor, Tensor]
    copy_started: torch.cuda.Event
    copy_finished: torch.cuda.Event


class AsyncLatentCacheWriter:
    def __init__(
        self,
        writer: LatentCacheWriter,
        batch_size: int,
        device: torch.device | str,
    ) -> None:
        if batch_size < 1:
            raise ValueError("cache pipeline batch size must be positive")
        self.writer = writer
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.asynchronous = self.device.type == "cuda"
        self._started = time.perf_counter()
        self._examples = 0
        self._batches = 0
        self._backpressure = 0.0
        self._copy = 0.0
        self._write = 0.0
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
                    )
                )
            self._thread = threading.Thread(
                target=self._writer_loop,
                name="latent-cache-writer",
                daemon=False,
            )
            self._thread.start()

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("latent-cache disk writer failed") from self._error

    def acquire_buffer(self) -> _HostBuffer | None:
        self._raise_error()
        if not self.asynchronous:
            return None
        started = time.perf_counter()
        while True:
            self._raise_error()
            try:
                result = self._free.get(timeout=0.05)
                self._backpressure += time.perf_counter() - started
                return result
            except queue.Empty:
                continue

    def release_buffer(self, buffer: _HostBuffer | None) -> None:
        if buffer is not None:
            self._free.put(buffer)

    def submit(
        self,
        indices: Tensor,
        values: Tensor,
        *,
        buffer: _HostBuffer | None = None,
    ) -> None:
        self._raise_error()
        count = indices.numel()
        expected = (count, *self.writer.shape[1:])
        if indices.ndim != 1 or not 0 < count <= self.batch_size:
            self.release_buffer(buffer)
            raise ValueError("cache pipeline indices must be a non-empty batch")
        if tuple(values.shape) != expected:
            self.release_buffer(buffer)
            raise ValueError(f"cache pipeline latents must have shape {expected}")
        if not self.asynchronous:
            started = time.perf_counter()
            self.writer.write(indices, values)
            self._write += time.perf_counter() - started
        else:
            buffer = buffer or self.acquire_buffer()
            assert buffer is not None and self._copy_stream is not None
            current = torch.cuda.current_stream(self.device)
            self._copy_stream.wait_stream(current)
            copy_started = torch.cuda.Event(enable_timing=True)
            copy_finished = torch.cuda.Event(enable_timing=True)
            copy_indices = indices.to(dtype=torch.int64)
            with torch.cuda.stream(self._copy_stream):
                copy_started.record()
                buffer.indices[:count].copy_(copy_indices, non_blocking=copy_indices.is_cuda)
                buffer.values[:count].copy_(values, non_blocking=True)
                copy_finished.record()
            self._jobs.put(
                _WriteJob(
                    buffer,
                    count,
                    (copy_indices, values),
                    copy_started,
                    copy_finished,
                )
            )
        self._examples += count
        self._batches += 1

    def _writer_loop(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                job.copy_finished.synchronize()
                self._copy += job.copy_started.elapsed_time(job.copy_finished) / 1_000
                if self._error is None:
                    started = time.perf_counter()
                    self.writer.write(
                        job.buffer.indices[: job.count],
                        job.buffer.values[: job.count],
                    )
                    self._write += time.perf_counter() - started
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

    def metrics(self) -> LatentCachePipelineMetrics:
        return LatentCachePipelineMetrics(
            self._examples,
            self._batches,
            self._backpressure,
            self._copy,
            self._write,
            time.perf_counter() - self._started,
        )


class LatentCache:
    def __init__(self, config: ExperimentConfig, dataset_size: int) -> None:
        self.spec = latent_cache_spec(config, dataset_size)
        self.directory = latent_cache_path(config, dataset_size)
        self.manifest = _validate_manifest(self.directory, self.spec, require_complete=True)
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
            raise ValueError("latent-cache indices must have shape [B]")
        if np.any(positions < 0) or np.any(positions >= self.shape[0]):
            raise IndexError("latent-cache index is out of range")
        bits = np.asarray(self._values[positions])
        return torch.from_numpy(bits).view(torch.bfloat16)
