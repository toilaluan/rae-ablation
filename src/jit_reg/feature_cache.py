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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from .config import ExperimentConfig
from .reg import FeatureTargets

Component = Literal["reg", "repa"]

CACHE_SCHEMA_VERSION = 4
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
            "feature_schema": "dino-unit-std-pooled-spatial-v3",
            "pool_size": reg.pool_size,
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


def required_cache_bytes(spec: dict[str, Any]) -> int:
    """Return allocated value and completion bytes for one component cache."""
    shape = _shape(spec)
    return int(np.prod(shape, dtype=np.int64)) * 2 + shape[0]


def preflight_cache_capacity(
    directory: str | Path,
    specs: Mapping[Component, dict[str, Any]],
    paths: Mapping[Component, Path] | None = None,
) -> int:
    """Reject cache builds that cannot fit their not-yet-allocated payloads."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    needed = 0
    for component, spec in specs.items():
        shape = _shape(spec)
        path = paths[component] if paths is not None else root / component
        expected_values = int(np.prod(shape, dtype=np.int64)) * 2
        expected_completion = shape[0]
        for filename, expected in (
            (VALUES_FILE, expected_values),
            (COMPLETION_FILE, expected_completion),
        ):
            file = path / filename
            allocated = (
                min(file.stat().st_blocks * 512, expected)
                if file.is_file()
                else 0
            )
            needed += max(expected - allocated, 0)
    available = shutil.disk_usage(root).free
    if needed > available:
        gib = 2**30
        raise OSError(
            "insufficient disk capacity for feature cache: "
            f"{needed / gib:.1f} GiB required, {available / gib:.1f} GiB available"
        )
    return needed


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


@dataclass(frozen=True)
class CachePipelineMetrics:
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
    values: dict[Component, Tensor]


@dataclass
class _WriteJob:
    buffer: _HostBuffer
    count: int
    sources: tuple[Tensor, ...]
    copy_started: torch.cuda.Event
    copy_finished: torch.cuda.Event


class AsyncFeatureCacheWriter:
    """Two-buffer CUDA-to-host pipeline with one durable disk-writer thread."""

    def __init__(
        self,
        writers: Mapping[Component, FeatureComponentCacheWriter],
        batch_size: int,
        device: torch.device | str,
    ) -> None:
        if not writers:
            raise ValueError("asynchronous cache writer requires a component writer")
        if batch_size < 1:
            raise ValueError("cache pipeline batch size must be positive")
        self.writers = dict(writers)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.asynchronous = self.device.type == "cuda"
        self._started = time.perf_counter()
        self._examples = 0
        self._batches = 0
        self._backpressure_seconds = 0.0
        self._copy_seconds = 0.0
        self._write_seconds = 0.0
        self._metrics_lock = threading.Lock()
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
                        indices=torch.empty(
                            batch_size, dtype=torch.int64, pin_memory=True
                        ),
                        values={
                            component: torch.empty(
                                (batch_size, writer.shape[1], writer.shape[2]),
                                dtype=torch.bfloat16,
                                pin_memory=True,
                            )
                            for component, writer in self.writers.items()
                        },
                    )
                )
            self._thread = threading.Thread(
                target=self._writer_loop,
                name="feature-cache-writer",
                daemon=False,
            )
            self._thread.start()

    def __enter__(self) -> AsyncFeatureCacheWriter:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        del exception_type, exception, traceback
        self.close()

    def _raise_worker_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("feature-cache disk writer failed") from self._error

    def acquire_buffer(self) -> _HostBuffer | None:
        """Reserve a host slot before GPU extraction to bound live target tensors."""
        if self._closed:
            raise RuntimeError("feature-cache pipeline is closed")
        self._raise_worker_error()
        if not self.asynchronous:
            return None
        started = time.perf_counter()
        while True:
            self._raise_worker_error()
            try:
                buffer = self._free.get(timeout=0.05)
                break
            except queue.Empty:
                continue
        with self._metrics_lock:
            self._backpressure_seconds += time.perf_counter() - started
        return buffer

    def release_buffer(self, buffer: _HostBuffer | None) -> None:
        """Return a reservation when extraction fails before submission."""
        if buffer is not None:
            self._free.put(buffer)

    def submit(
        self,
        indices: Tensor,
        targets: FeatureTargets,
        *,
        buffer: _HostBuffer | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("feature-cache pipeline is closed")
        self._raise_worker_error()
        try:
            count = indices.numel()
            if indices.ndim != 1 or count < 1 or count > self.batch_size:
                raise ValueError("cache pipeline indices must be a non-empty batch")
            active: dict[Component, Tensor] = {}
            for component in self.writers:
                value = getattr(targets, component)
                if value is None:
                    raise ValueError(
                        f"active {component.upper()} cache has no targets"
                    )
                expected = (
                    count,
                    self.writers[component].shape[1],
                    self.writers[component].shape[2],
                )
                if tuple(value.shape) != expected:
                    raise ValueError(
                        f"{component.upper()} cache targets must have shape {expected}"
                    )
                active[component] = value
        except BaseException:
            self.release_buffer(buffer)
            raise

        if not self.asynchronous:
            started = time.perf_counter()
            for component, value in active.items():
                self.writers[component].write(indices, value)
            elapsed = time.perf_counter() - started
            with self._metrics_lock:
                self._write_seconds += elapsed
                self._examples += count
                self._batches += 1
            return

        buffer = buffer if buffer is not None else self.acquire_buffer()
        assert buffer is not None
        assert self._copy_stream is not None
        current_stream = torch.cuda.current_stream(self.device)
        self._copy_stream.wait_stream(current_stream)
        copy_started = torch.cuda.Event(enable_timing=True)
        copy_finished = torch.cuda.Event(enable_timing=True)
        copy_indices = indices.to(dtype=torch.int64)
        with torch.cuda.stream(self._copy_stream):
            copy_started.record()
            buffer.indices[:count].copy_(
                copy_indices, non_blocking=copy_indices.is_cuda
            )
            for component, value in active.items():
                buffer.values[component][:count].copy_(
                    value, non_blocking=True
                )
            copy_finished.record()
        self._jobs.put(
            _WriteJob(
                buffer=buffer,
                count=count,
                sources=(copy_indices, *active.values()),
                copy_started=copy_started,
                copy_finished=copy_finished,
            )
        )
        with self._metrics_lock:
            self._examples += count
            self._batches += 1

    def _writer_loop(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                job.copy_finished.synchronize()
                with self._metrics_lock:
                    self._copy_seconds += (
                        job.copy_started.elapsed_time(job.copy_finished) / 1_000
                    )
                if self._error is None:
                    started = time.perf_counter()
                    for component, writer in self.writers.items():
                        writer.write(
                            job.buffer.indices[: job.count],
                            job.buffer.values[component][: job.count],
                        )
                    with self._metrics_lock:
                        self._write_seconds += time.perf_counter() - started
                # Keeping the job alive until here retains all CUDA source tensors.
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
        self._raise_worker_error()

    def close(self) -> None:
        if self._closed:
            self._raise_worker_error()
            return
        if self.asynchronous:
            self._jobs.put(None)
            assert self._thread is not None
            self._thread.join()
        self._closed = True
        self._raise_worker_error()

    def metrics(self) -> CachePipelineMetrics:
        with self._metrics_lock:
            return CachePipelineMetrics(
                examples=self._examples,
                batches=self._batches,
                buffer_backpressure_seconds=self._backpressure_seconds,
                copy_seconds=self._copy_seconds,
                write_flush_seconds=self._write_seconds,
                total_seconds=time.perf_counter() - self._started,
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
        # NumPy advanced indexing returns a detached array, so no second copy is needed.
        bits = np.asarray(self._values[positions])
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
