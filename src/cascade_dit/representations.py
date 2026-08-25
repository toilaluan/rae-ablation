from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .config import IMAGE_SIZE, PATCH_SIZE, TOKEN_COUNT, TOKEN_DIM

REPRESENTATION_SCHEMA_VERSION = 2
REPRESENTATION_NAMES = ("dino12", "dino8", "dino4", "rgb")


def required_representation_bytes(dataset_size: int) -> int:
    if dataset_size < 1:
        raise ValueError("dataset size must be positive")
    return 3 * dataset_size * (TOKEN_COUNT * TOKEN_DIM * 2 + 1)


def preflight_representation_capacity(root: str | Path, dataset_size: int) -> int:
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    per_layer_values = dataset_size * TOKEN_COUNT * TOKEN_DIM * 2
    needed = 0
    for name in ("dino12", "dino8", "dino4"):
        for path, expected in (
            (directory / f"{name}.bf16", per_layer_values),
            (directory / f"{name}.completion.u8", dataset_size),
        ):
            allocated = min(path.stat().st_blocks * 512, expected) if path.is_file() else 0
            needed += max(expected - allocated, 0)
    available = shutil.disk_usage(directory).free
    if needed > available:
        raise OSError(
            f"insufficient representation-cache capacity: {needed / 2**30:.1f} GiB "
            f"required, {available / 2**30:.1f} GiB available"
        )
    return needed


def patchify_rgb(images: Tensor) -> Tensor:
    if images.ndim != 4 or images.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("RGB images must have shape [B,3,128,128]")
    return F.unfold(images, kernel_size=PATCH_SIZE, stride=PATCH_SIZE).transpose(1, 2)


def unpatchify_rgb(tokens: Tensor) -> Tensor:
    if tokens.ndim != 3 or tokens.shape[1:] != (TOKEN_COUNT, TOKEN_DIM):
        raise ValueError("RGB patch tokens must have shape [B,64,768]")
    return F.fold(
        tokens.transpose(1, 2),
        output_size=(IMAGE_SIZE, IMAGE_SIZE),
        kernel_size=PATCH_SIZE,
        stride=PATCH_SIZE,
    )


@dataclass(frozen=True)
class ChannelStatistics:
    mean: Tensor
    std: Tensor
    count: int

    def normalize(self, values: Tensor) -> Tensor:
        return (values - self.mean.to(values)) / self.std.to(values)

    def denormalize(self, values: Tensor) -> Tensor:
        return values * self.std.to(values) + self.mean.to(values)


class ChannelStatisticsAccumulator:
    """Streaming, mergeable float64 channel statistics for [..., C] tokens."""

    def __init__(self, channels: int = TOKEN_DIM) -> None:
        self.channels = channels
        self.count = 0
        self.mean = torch.zeros(channels, dtype=torch.float64)
        self.m2 = torch.zeros(channels, dtype=torch.float64)

    def update(self, values: Tensor) -> None:
        if values.ndim < 2 or values.shape[-1] != self.channels:
            raise ValueError(f"statistics values must end in {self.channels} channels")
        flat = values.detach().reshape(-1, self.channels).to("cpu", torch.float64)
        if flat.numel() == 0:
            return
        batch_count = flat.shape[0]
        batch_mean = flat.mean(0)
        batch_m2 = ((flat - batch_mean).square()).sum(0)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * (batch_count / total)
        self.m2 += batch_m2 + delta.square() * (self.count * batch_count / total)
        self.count = total

    def merge(self, other: ChannelStatisticsAccumulator) -> None:
        if other.channels != self.channels:
            raise ValueError("statistics channel counts differ")
        if other.count == 0:
            return
        if self.count == 0:
            self.count, self.mean, self.m2 = other.count, other.mean.clone(), other.m2.clone()
            return
        delta = other.mean - self.mean
        total = self.count + other.count
        self.m2 += other.m2 + delta.square() * (self.count * other.count / total)
        self.mean += delta * (other.count / total)
        self.count = total

    def finalize(self, epsilon: float = 1e-6) -> ChannelStatistics:
        if self.count < 2:
            raise ValueError("at least two tokens are required for statistics")
        variance = self.m2 / self.count
        return ChannelStatistics(
            self.mean.float(), variance.clamp_min(epsilon**2).sqrt().float(), self.count
        )


def deterministic_subset_indices(dataset_size: int, subset_size: int, seed: int) -> Tensor:
    if dataset_size < 1 or not 0 < subset_size <= dataset_size:
        raise ValueError("subset size must be within the dataset")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(dataset_size, generator=generator)[:subset_size]


def fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class RepresentationManifest:
    dataset_size: int
    dataset_fingerprint: str
    encoder_fingerprint: str
    preparation_fingerprint: str
    statistics: dict[str, dict[str, Any]]
    cache_files: dict[str, str]
    completion_files: dict[str, str]
    alignments: dict[str, dict[str, Any]]
    cumulative_projectors: dict[str, str]
    tensor_file: str = "representations.safetensors"
    complete: bool = False
    schema_version: int = REPRESENTATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + f".partial-{os.getpid()}")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)

    @classmethod
    def load(cls, path: str | Path, *, require_complete: bool = True) -> RepresentationManifest:
        target = Path(path)
        try:
            values = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"representation manifest is missing or corrupt: {target}") from error
        if values.get("schema_version") != REPRESENTATION_SCHEMA_VERSION:
            raise ValueError("representation manifest schema is incompatible")
        manifest = cls(**values)
        if require_complete and not manifest.complete:
            raise ValueError("representation manifest is incomplete")
        root = target.parent
        if not all(
            (
                manifest.dataset_fingerprint,
                manifest.encoder_fingerprint,
                manifest.preparation_fingerprint,
            )
        ):
            raise ValueError("representation manifest fingerprints are incomplete")
        for name in REPRESENTATION_NAMES:
            try:
                statistics = manifest.statistics[name]
                mean, std = statistics["mean"], statistics["std"]
                count = int(statistics["count"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"representation statistics are corrupt: {name}") from error
            if len(mean) != TOKEN_DIM or len(std) != TOKEN_DIM or count < 2:
                raise ValueError(f"representation statistics have the wrong shape: {name}")
            if any(float(value) <= 0 for value in std):
                raise ValueError(f"representation standard deviations are invalid: {name}")
        expected_bytes = manifest.dataset_size * TOKEN_COUNT * TOKEN_DIM * 2
        for name in ("dino12", "dino8", "dino4"):
            cache = root / manifest.cache_files[name]
            completion = root / manifest.completion_files[name]
            if not cache.is_file() or cache.stat().st_size != expected_bytes:
                raise ValueError(f"representation cache is missing or has the wrong size: {cache}")
            if not completion.is_file() or completion.stat().st_size != manifest.dataset_size:
                raise ValueError(f"representation completion bitmap is invalid: {completion}")
            bitmap = np.memmap(completion, mode="r", dtype=np.uint8, shape=(manifest.dataset_size,))
            if np.any((bitmap != 0) & (bitmap != 1)) or (require_complete and np.any(bitmap != 1)):
                raise ValueError(f"representation cache is incomplete or corrupt: {name}")
        tensor_path = root / manifest.tensor_file
        if not tensor_path.is_file():
            raise ValueError(f"representation tensor bundle is missing: {tensor_path}")
        from safetensors.torch import load_file

        tensors = load_file(tensor_path)
        if set(manifest.cumulative_projectors) != set(REPRESENTATION_NAMES):
            raise ValueError("cumulative projectors do not cover all stages")
        try:
            matrix_keys = tuple(values["matrix"] for values in manifest.alignments.values())
            if any(
                values.get("parameterization")
                != "full-rank orthogonal initialization alignment"
                or not math.isfinite(float(values["heldout_mse"]))
                or not math.isfinite(float(values["orthogonality_error"]))
                for values in manifest.alignments.values()
            ):
                raise ValueError("invalid alignment diagnostics")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("representation alignment diagnostics are corrupt") from error
        keys = {
            *matrix_keys,
            *manifest.cumulative_projectors.values(),
            *(f"statistics.{name}.mean" for name in REPRESENTATION_NAMES),
            *(f"statistics.{name}.std" for name in REPRESENTATION_NAMES),
        }
        if not keys.issubset(tensors):
            raise ValueError("representation tensor bundle is incomplete")
        for key in (*manifest.cumulative_projectors.values(), *matrix_keys):
            matrix = tensors[key]
            if matrix.shape != (TOKEN_DIM, TOKEN_DIM) or not torch.isfinite(matrix).all():
                raise ValueError(f"alignment tensor has the wrong shape: {key}")
            identity = torch.eye(TOKEN_DIM, dtype=matrix.dtype, device=matrix.device)
            if torch.linalg.matrix_norm(matrix.T @ matrix - identity) > 1e-3:
                raise ValueError(f"alignment tensor is not orthogonal: {key}")
        for name in REPRESENTATION_NAMES:
            for statistic in ("mean", "std"):
                key = f"statistics.{name}.{statistic}"
                recorded = torch.tensor(manifest.statistics[name][statistic])
                if tensors[key].shape != (TOKEN_DIM,) or not torch.equal(tensors[key], recorded):
                    raise ValueError(f"manifest and tensor statistics disagree: {key}")
        return manifest


class LayerMemmap:
    """A resumable raw BF16 cache with a per-image completion bitmap."""

    def __init__(self, values_path: str | Path, completion_path: str | Path, size: int) -> None:
        self.values_path = Path(values_path)
        self.completion_path = Path(completion_path)
        self.size = size
        self.values = np.memmap(
            self.values_path, mode="r+", dtype=np.uint16, shape=(size, TOKEN_COUNT, TOKEN_DIM)
        )
        self.completion = np.memmap(
            self.completion_path, mode="r+", dtype=np.uint8, shape=(size,)
        )

    @classmethod
    def initialize(cls, root: str | Path, name: str, size: int) -> LayerMemmap:
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=True)
        values_path = directory / f"{name}.bf16"
        completion_path = directory / f"{name}.completion.u8"
        expected = size * TOKEN_COUNT * TOKEN_DIM * 2
        if values_path.exists() != completion_path.exists():
            raise ValueError(f"incomplete cache file pair for {name}")
        if not values_path.exists():
            values = np.memmap(
                values_path, mode="w+", dtype=np.uint16, shape=(size, TOKEN_COUNT, TOKEN_DIM)
            )
            completion = np.memmap(completion_path, mode="w+", dtype=np.uint8, shape=(size,))
            completion[:] = 0
            values.flush()
            completion.flush()
            del values, completion
        elif values_path.stat().st_size != expected or completion_path.stat().st_size != size:
            raise ValueError(f"existing cache files have incompatible shape for {name}")
        return cls(values_path, completion_path, size)

    def pending_indices(self, rank: int = 0, world_size: int = 1) -> np.ndarray:
        if not 0 <= rank < world_size:
            raise ValueError("rank must be within world size")
        return np.flatnonzero(np.asarray(self.completion) == 0)[rank::world_size]

    def write(self, indices: Tensor, values: Tensor) -> None:
        positions = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        expected = (positions.size, TOKEN_COUNT, TOKEN_DIM)
        if tuple(values.shape) != expected or positions.size != np.unique(positions).size:
            raise ValueError(f"cache values must have shape {expected} with unique indices")
        if np.any(positions < 0) or np.any(positions >= self.size):
            raise IndexError("cache index is out of range")
        bits = values.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint16).numpy()
        self.values[positions] = bits
        self.values.flush()
        self.completion[positions] = 1
        self.completion.flush()

    def lookup(self, indices: Tensor) -> Tensor:
        positions = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if np.any(self.completion[positions] != 1):
            raise ValueError("requested representation has not been cached")
        return torch.from_numpy(np.asarray(self.values[positions]).copy()).view(torch.bfloat16)

    def finalize(self) -> None:
        if np.any(np.asarray(self.completion) != 1):
            raise ValueError("cannot finalize an incomplete representation cache")
