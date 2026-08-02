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

TOKEN_CACHE_SCHEMA_VERSION = 13
VALUES_FILE = "values.bf16"
COMPLETION_FILE = "completion.u8"
MEAN_FILE = "mean.f32"
STD_FILE = "std.f32"
MANIFEST_FILE = "manifest.json"
STATISTICS_ALGORITHM = "per-channel-population-fp64-sum-and-square-v1"


def _json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def token_cache_spec(config: ExperimentConfig, dataset_size: int) -> dict[str, Any]:
    if dataset_size < 1:
        raise ValueError("token-cache dataset size must be positive")
    inputs = {
        "schema_version": TOKEN_CACHE_SCHEMA_VERSION,
        "dataset": {
            "repository": config.data.repository,
            "revision": config.data.revision,
            "split": config.data.split,
            "max_samples": config.data.max_samples,
            "dataset_size": dataset_size,
            "image_size": 256,
            "preprocessing": [
                "PIL.convert(RGB)",
                "PIL.resize(256x256, bicubic) when needed",
                "uint8_to_float32_[0,1]",
                "no-random-augmentation",
            ],
        },
        "tensor": {"shape": list(config.model.input_shape), "dtype": "bfloat16"},
        "rae": {
            "loader": "transformers.AutoModel.from_pretrained",
            "base_repository": config.rae.base_repository,
            "base_revision": config.rae.base_revision,
            "compressor_repository": config.rae.compressor_repository,
            "compressor_revision": config.rae.compressor_revision,
            "trust_remote_code": config.rae.trust_remote_code,
            "remote_code_schema": config.rae.remote_code_schema,
            "load_order": "base-then-compressor-with-rae-injection",
        },
        "statistics": {
            "algorithm": STATISTICS_ALGORITHM,
            "axes": ["image", "height", "width"],
            "accumulator_dtype": "float64",
            "output_dtype": "float32",
            "standard_deviation": "population",
        },
    }
    environment = {"torch": torch.__version__}
    try:
        environment["transformers"] = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        environment["transformers"] = "not-installed"
    return {"fingerprint_inputs": inputs, "environment": environment}


def token_cache_fingerprint(spec: dict[str, Any]) -> str:
    inputs = spec.get("fingerprint_inputs")
    if not isinstance(inputs, dict):
        raise ValueError("token-cache spec has no fingerprint inputs")
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def token_cache_path(config: ExperimentConfig, dataset_size: int) -> Path:
    spec = token_cache_spec(config, dataset_size)
    return Path(config.rae.cache.directory) / token_cache_fingerprint(spec)


def _shape(spec: dict[str, Any]) -> tuple[int, ...]:
    inputs = spec["fingerprint_inputs"]
    return (
        int(inputs["dataset"]["dataset_size"]),
        *(int(value) for value in inputs["tensor"]["shape"]),
    )


def required_token_cache_bytes(spec: dict[str, Any]) -> int:
    shape = _shape(spec)
    statistics_bytes = 2 * shape[1] * np.dtype(np.float32).itemsize
    return int(np.prod(shape, dtype=np.int64)) * 2 + shape[0] + statistics_bytes


def preflight_token_cache_capacity(
    directory: str | Path,
    spec: dict[str, Any],
    path: Path | None = None,
) -> int:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = path or root
    shape = _shape(spec)
    expected_files = (
        (VALUES_FILE, int(np.prod(shape, dtype=np.int64)) * 2),
        (COMPLETION_FILE, shape[0]),
        (MEAN_FILE, shape[1] * 4),
        (STD_FILE, shape[1] * 4),
    )
    needed = 0
    for filename, expected in expected_files:
        file = target / filename
        allocated = min(file.stat().st_blocks * 512, expected) if file.is_file() else 0
        needed += max(expected - allocated, 0)
    available = shutil.disk_usage(root).free
    if needed > available:
        gib = 2**30
        raise OSError(
            "insufficient disk capacity for token cache: "
            f"{needed / gib:.1f} GiB required, {available / gib:.1f} GiB available"
        )
    return needed


def _manifest(spec: dict[str, Any], *, complete: bool) -> dict[str, Any]:
    return {
        "schema_version": TOKEN_CACHE_SCHEMA_VERSION,
        "fingerprint": token_cache_fingerprint(spec),
        "complete": complete,
        "fingerprint_inputs": spec["fingerprint_inputs"],
        "environment": spec["environment"],
        "shape": list(_shape(spec)),
        "values_file": VALUES_FILE,
        "completion_file": COMPLETION_FILE,
        "mean_file": MEAN_FILE,
        "std_file": STD_FILE,
    }


def _validate_manifest(
    directory: Path, spec: dict[str, Any], *, require_complete: bool
) -> dict[str, Any]:
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"token-cache manifest is missing: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != TOKEN_CACHE_SCHEMA_VERSION:
        raise ValueError("token-cache manifest is incompatible with schema v13")
    if (
        manifest.get("fingerprint") != token_cache_fingerprint(spec)
        or manifest.get("fingerprint_inputs") != spec["fingerprint_inputs"]
        or manifest.get("shape") != list(_shape(spec))
    ):
        raise ValueError(f"token-cache manifest is incompatible: {directory}")
    shape = _shape(spec)
    expected = ((VALUES_FILE, int(np.prod(shape)) * 2), (COMPLETION_FILE, shape[0]))
    for filename, size in expected:
        file = directory / filename
        if not file.is_file() or file.stat().st_size != size:
            raise ValueError(f"token-cache file has the wrong size: {file}")
    if require_complete:
        if not manifest.get("complete"):
            raise ValueError(f"token cache is incomplete: {directory}")
        bitmap = np.memmap(directory / COMPLETION_FILE, mode="r", dtype=np.uint8, shape=(shape[0],))
        if np.any(bitmap != 1):
            raise ValueError(f"token-cache completion bitmap is incomplete: {directory}")
        for filename in (MEAN_FILE, STD_FILE):
            file = directory / filename
            if not file.is_file() or file.stat().st_size != shape[1] * 4:
                raise ValueError(f"token-cache statistics are invalid: {file}")
        mean = np.fromfile(directory / MEAN_FILE, dtype=np.float32)
        std = np.fromfile(directory / STD_FILE, dtype=np.float32)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("token-cache statistics contain invalid values")
    return manifest


@contextmanager
def token_cache_build_lock(directory: str | Path) -> Iterator[None]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".build.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process is building token cache {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def normalize_tokens(tokens: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    if tokens.ndim != 4 or mean.ndim != 1 or std.ndim != 1:
        raise ValueError("normalization expects [B,C,H,W] tokens and [C] statistics")
    if tokens.shape[1] != mean.numel() or mean.shape != std.shape:
        raise ValueError("token channels and statistics do not match")
    return (tokens.float() - mean[None, :, None, None]) / std[None, :, None, None]


def denormalize_tokens(tokens: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    if tokens.ndim != 4 or mean.ndim != 1 or std.ndim != 1:
        raise ValueError("denormalization expects [B,C,H,W] tokens and [C] statistics")
    if tokens.shape[1] != mean.numel() or mean.shape != std.shape:
        raise ValueError("token channels and statistics do not match")
    return tokens.float() * std[None, :, None, None] + mean[None, :, None, None]


def _write_f32_atomic(path: Path, value: Tensor) -> None:
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    value.detach().cpu().to(torch.float32).contiguous().numpy().tofile(temporary)
    os.replace(temporary, path)


class TokenCacheWriter:
    def __init__(self, directory: str | Path, spec: dict[str, Any]) -> None:
        self.directory = Path(directory)
        self.spec = spec
        self.shape = _shape(spec)
        _validate_manifest(self.directory, spec, require_complete=False)
        self.values = np.memmap(self.directory / VALUES_FILE, mode="r+", dtype=np.uint16, shape=self.shape)
        self.completion = np.memmap(self.directory / COMPLETION_FILE, mode="r+", dtype=np.uint8, shape=(self.shape[0],))

    @classmethod
    def initialize(cls, directory: str | Path, spec: dict[str, Any]) -> bool:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / MANIFEST_FILE
        if manifest.exists():
            return bool(_validate_manifest(directory, spec, require_complete=False)["complete"])
        shape = _shape(spec)
        values = np.memmap(directory / VALUES_FILE, mode="w+", dtype=np.uint16, shape=shape)
        completion = np.memmap(directory / COMPLETION_FILE, mode="w+", dtype=np.uint8, shape=(shape[0],))
        completion[:] = 0
        values.flush()
        completion.flush()
        _json_atomic(manifest, _manifest(spec, complete=False))
        return False

    @property
    def values_complete(self) -> bool:
        return not np.any(np.asarray(self.completion) != 1)

    def pending_indices(self, rank: int = 0, world_size: int = 1) -> np.ndarray:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be within world size")
        return np.flatnonzero(np.asarray(self.completion) == 0)[rank::world_size]

    def write(self, indices: Tensor, values: Tensor) -> None:
        positions = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        expected = (len(positions), *self.shape[1:])
        if positions.ndim != 1 or tuple(values.shape) != expected:
            raise ValueError(f"cached tokens must have shape {expected}")
        if np.any(positions < 0) or np.any(positions >= self.shape[0]):
            raise IndexError("token-cache index is out of range")
        bits = values.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().view(torch.uint16).numpy()
        self.values[positions] = bits
        self.values.flush()
        self.completion[positions] = 1
        self.completion.flush()

    def compute_statistics(
        self,
        rank: int = 0,
        world_size: int = 1,
        *,
        chunk_size: int = 256,
        device: torch.device | str = "cpu",
    ) -> tuple[Tensor, Tensor]:
        if not self.values_complete:
            raise ValueError("cannot compute statistics before all token values are complete")
        if not 0 <= rank < world_size or chunk_size < 1:
            raise ValueError("invalid statistics partition or chunk size")
        channels = self.shape[1]
        target = torch.device(device)
        total = torch.zeros(channels, dtype=torch.float64, device=target)
        square = torch.zeros_like(total)
        count = torch.zeros((), dtype=torch.float64, device=target)
        local_indices = np.arange(rank, self.shape[0], world_size, dtype=np.int64)
        for start in range(0, len(local_indices), chunk_size):
            indices = local_indices[start : start + chunk_size]
            bits = np.asarray(self.values[indices]).copy()
            values = torch.from_numpy(bits).view(torch.bfloat16).to(target, dtype=torch.float64)
            total += values.sum(dim=(0, 2, 3))
            square += values.square().sum(dim=(0, 2, 3))
            count += values.shape[0] * values.shape[2] * values.shape[3]
        if world_size > 1:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("distributed statistics require an initialized process group")
            for value in (total, square, count):
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        mean = total / count
        variance = (square / count - mean.square()).clamp_min(0)
        std = variance.sqrt()
        if torch.any(std <= 0) or not torch.all(torch.isfinite(std)):
            raise ValueError("token statistics contain a non-positive or non-finite standard deviation")
        return mean.float(), std.float()

    def save_statistics(self, mean: Tensor, std: Tensor) -> None:
        if mean.shape != (self.shape[1],) or std.shape != mean.shape:
            raise ValueError("token statistics must have shape [channels]")
        if not torch.all(torch.isfinite(mean)) or not torch.all(torch.isfinite(std)) or torch.any(std <= 0):
            raise ValueError("token statistics are invalid")
        _write_f32_atomic(self.directory / MEAN_FILE, mean)
        _write_f32_atomic(self.directory / STD_FILE, std)

    def finalize(self, mean: Tensor | None = None, std: Tensor | None = None) -> None:
        if not self.values_complete:
            raise ValueError("cannot finalize an incomplete token cache")
        if (mean is None) != (std is None):
            raise ValueError("mean and std must be supplied together")
        if mean is not None and std is not None:
            self.save_statistics(mean, std)
        for filename in (MEAN_FILE, STD_FILE):
            if not (self.directory / filename).is_file():
                raise ValueError("cannot finalize token cache without statistics")
        _json_atomic(self.directory / MANIFEST_FILE, _manifest(self.spec, complete=True))
        _validate_manifest(self.directory, self.spec, require_complete=True)


@dataclass(frozen=True)
class TokenCachePipelineMetrics:
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


class AsyncTokenCacheWriter:
    """Overlap CUDA-to-host copies and disk flushes with compressor execution."""

    def __init__(self, writer: TokenCacheWriter, batch_size: int, device: torch.device | str) -> None:
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
                target=self._writer_loop, name="token-cache-writer", daemon=False
            )
            self._thread.start()

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("token-cache disk writer failed") from self._error

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
            raise ValueError(f"cache pipeline tokens must have shape {expected}")
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
                        job.buffer.indices[: job.count], job.buffer.values[: job.count]
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

    def metrics(self) -> TokenCachePipelineMetrics:
        return TokenCachePipelineMetrics(
            self._examples,
            self._batches,
            self._backpressure,
            self._copy,
            self._write,
            time.perf_counter() - self._started,
        )


class TokenCache:
    def __init__(self, config: ExperimentConfig, dataset_size: int) -> None:
        self.spec = token_cache_spec(config, dataset_size)
        self.directory = token_cache_path(config, dataset_size)
        self.manifest = _validate_manifest(self.directory, self.spec, require_complete=True)
        self.fingerprint = self.manifest["fingerprint"]
        self.shape = _shape(self.spec)
        self.values = np.memmap(self.directory / VALUES_FILE, mode="r", dtype=np.uint16, shape=self.shape)
        self.mean = torch.from_numpy(np.fromfile(self.directory / MEAN_FILE, dtype=np.float32).copy())
        self.std = torch.from_numpy(np.fromfile(self.directory / STD_FILE, dtype=np.float32).copy())

    def raw_lookup(self, indices: Tensor) -> Tensor:
        positions = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= self.shape[0]):
            raise IndexError("token-cache index is out of range")
        # NumPy advanced indexing already returns an owning copy. Copying it again
        # doubles host-memory traffic for every training batch.
        bits = self.values[positions]
        return torch.from_numpy(bits).view(torch.bfloat16)

    def lookup(self, indices: Tensor) -> Tensor:
        raw = self.raw_lookup(indices)
        normalized = normalize_tokens(raw, self.mean, self.std)
        return normalized.to(torch.bfloat16)

    def denormalize(self, tokens: Tensor) -> Tensor:
        return denormalize_tokens(tokens, self.mean.to(tokens.device), self.std.to(tokens.device))


@dataclass(frozen=True)
class TokenStatistics:
    mean: Tensor
    std: Tensor

    def denormalize(self, tokens: Tensor) -> Tensor:
        return denormalize_tokens(tokens, self.mean.to(tokens.device), self.std.to(tokens.device))


def load_token_statistics(directory: str | Path, fingerprint: str) -> TokenStatistics:
    path = Path(directory) / fingerprint
    manifest_path = path / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"token-cache manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != TOKEN_CACHE_SCHEMA_VERSION
        or manifest.get("fingerprint") != fingerprint
        or not manifest.get("complete")
    ):
        raise ValueError("sampling requires a complete schema-v13 token cache")
    shape = manifest.get("shape")
    if not isinstance(shape, list) or len(shape) != 4:
        raise ValueError("token-cache manifest has an invalid shape")
    channels = int(shape[1])
    mean = torch.from_numpy(np.fromfile(path / MEAN_FILE, dtype=np.float32).copy())
    std = torch.from_numpy(np.fromfile(path / STD_FILE, dtype=np.float32).copy())
    if mean.shape != (channels,) or std.shape != mean.shape or torch.any(std <= 0):
        raise ValueError("token-cache statistics are invalid")
    return TokenStatistics(mean, std)
