from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

SAMPLING_MANIFEST_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class SamplingParameters:
    steps: int = 40
    solver: str = "euler"
    timestep_shift: float = 1.0
    guidance_scale: float = 1.0
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    batch_size: int = 32
    seed: int = 0

    def validate(self) -> None:
        if min(self.steps, self.batch_size) < 1:
            raise ValueError("sampling steps and batch size must be positive")
        if self.solver not in {"euler", "heun"}:
            raise ValueError("sampling solver must be euler or heun")
        if self.timestep_shift <= 0:
            raise ValueError("sampling timestep shift must be positive")
        if not 0 <= self.guidance_start < self.guidance_end <= 1:
            raise ValueError("sampling CFG interval must satisfy 0 <= start < end <= 1")


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def noise_for_seeds(
    seeds: Tensor,
    shape: tuple[int, ...],
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    values = [
        torch.randn(shape, generator=torch.Generator().manual_seed(int(seed)), dtype=dtype)
        for seed in seeds.cpu()
    ]
    return torch.stack(values).to(device)


def sampling_plan(num_samples: int, num_classes: int, seed: int) -> tuple[Tensor, Tensor]:
    if num_samples < 1 or num_samples % num_classes:
        raise ValueError("sample count must be positive and divisible by num_classes")
    labels = torch.arange(num_samples, dtype=torch.int64) % num_classes
    permutation = torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed))
    labels = labels[permutation]
    seeds = torch.arange(num_samples, dtype=torch.int64) + seed * 1_000_003
    return labels, seeds


def sampling_plan_hash(labels: Tensor, seeds: Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(labels.numpy().tobytes())
    digest.update(seeds.numpy().tobytes())
    return digest.hexdigest()


def sampling_output_name(num_samples: int, parameters: SamplingParameters) -> str:
    parameters.validate()
    return (
        f"samples-{parameters.solver}-n{num_samples}-steps{parameters.steps}"
        f"-cfg{parameters.guidance_scale:g}-shift{parameters.timestep_shift:g}"
    )


def _write_json(path: Path, values: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _distributed() -> tuple[int, int, bool]:
    active = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not active:
        return 0, 1, False
    return torch.distributed.get_rank(), torch.distributed.get_world_size(), True


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _paths(output: Path) -> dict[str, Path]:
    return {
        "images": output / ".images.npy",
        "labels": output / ".labels.npy",
        "completion": output / ".completion.npy",
        "archive": output / "samples.npz",
        "manifest": output / "samples.json",
        "preview": output / "preview.png",
    }


def _load_memmaps(paths: dict[str, Path], mode: str = "r+") -> tuple[np.memmap, ...]:
    arrays = tuple(
        np.load(paths[name], mmap_mode=mode, allow_pickle=False)
        for name in ("images", "labels", "completion")
    )
    if not all(isinstance(array, np.memmap) for array in arrays):
        raise RuntimeError("temporary sampling arrays are not memory mapped")
    return arrays  # type: ignore[return-value]


def _validate_plan(
    labels: Tensor,
    seeds: Tensor,
    num_samples: int,
    num_classes: int,
) -> tuple[Tensor, Tensor]:
    labels = labels.detach().to(device="cpu", dtype=torch.int64).contiguous()
    seeds = seeds.detach().to(device="cpu", dtype=torch.int64).contiguous()
    if labels.shape != (num_samples,) or seeds.shape != (num_samples,):
        raise ValueError("sampling labels and seeds must have shape [num_samples]")
    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError("sampling labels are outside the configured class range")
    return labels, seeds


def _initialize_or_resume(
    output: Path,
    paths: dict[str, Path],
    base: dict[str, Any],
    labels: Tensor,
) -> tuple[bool, int]:
    output.mkdir(parents=True, exist_ok=True)
    sample_count = int(base["num_samples"])
    image_size = int(base["image_size"])
    temporary_names = ("images", "labels", "completion")
    if paths["manifest"].is_file():
        manifest = json.loads(paths["manifest"].read_text())
        if manifest.get("schema_version") != SAMPLING_MANIFEST_SCHEMA_VERSION:
            raise ValueError("sampling manifest schema is incompatible")
        if {key: manifest.get(key) for key in base} != base:
            raise ValueError("existing sample manifest does not match this request")
        if manifest.get("complete"):
            if not manifest.get("archive_retained", True):
                raise ValueError("completed sample archive was pruned after evaluation")
            if not paths["archive"].is_file() or not paths["preview"].is_file():
                raise ValueError("completed sample manifest is missing final artifacts")
            return True, sample_count
        if not all(paths[name].is_file() for name in temporary_names):
            raise ValueError("interrupted sampling run is missing temporary storage")
        images, saved_labels, completion = _load_memmaps(paths)
        if images.shape != (sample_count, image_size, image_size, 3) or images.dtype != np.uint8:
            raise ValueError("temporary sample images do not match their manifest")
        if saved_labels.shape != (sample_count,) or saved_labels.dtype != np.int64:
            raise ValueError("temporary sample labels do not match their manifest")
        if completion.shape != (sample_count,) or completion.dtype != np.uint8:
            raise ValueError("temporary completion map does not match its manifest")
        if not np.array_equal(saved_labels, labels.numpy()):
            raise ValueError("saved labels differ from the deterministic sampling plan")
        completed = int(np.count_nonzero(completion))
        del images, saved_labels, completion
        return False, completed
    if any(paths[name].exists() for name in (*temporary_names, "archive", "preview")):
        raise ValueError("refusing to overwrite untracked sample artifacts")
    images = np.lib.format.open_memmap(
        paths["images"],
        mode="w+",
        dtype=np.uint8,
        shape=(sample_count, image_size, image_size, 3),
    )
    saved_labels = np.lib.format.open_memmap(
        paths["labels"], mode="w+", dtype=np.int64, shape=(sample_count,)
    )
    completion = np.lib.format.open_memmap(
        paths["completion"], mode="w+", dtype=np.uint8, shape=(sample_count,)
    )
    saved_labels[:] = labels.numpy()
    completion[:] = 0
    images.flush()
    saved_labels.flush()
    completion.flush()
    del images, saved_labels, completion
    _write_json(
        paths["manifest"],
        base | {"complete": False, "completed_samples": 0, "archive_retained": False},
    )
    return False, 0


def _write_zip_member(archive: zipfile.ZipFile, source: Path, name: str) -> None:
    with source.open("rb") as source_handle, archive.open(
        name, mode="w", force_zip64=True
    ) as target_handle:
        shutil.copyfileobj(source_handle, target_handle, length=8 * 1024 * 1024)


def _package(images: Path, labels: Path, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + f".partial-{os.getpid()}")
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:
            _write_zip_member(archive, images, "arr_0.npy")
            _write_zip_member(archive, labels, "arr_1.npy")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _save_preview(images: np.ndarray, labels: np.ndarray, target: Path) -> None:
    count = min(16, len(images))
    if count < 1:
        raise ValueError("cannot save a preview without samples")
    side = images.shape[1]
    canvas = Image.new("RGB", (4 * side, 4 * (side + 18)), "white")
    draw = ImageDraw.Draw(canvas)
    for index in range(count):
        row, column = divmod(index, 4)
        x, y = column * side, row * (side + 18)
        canvas.paste(Image.fromarray(images[index]), (x, y))
        draw.text((x + 2, y + side + 1), str(int(labels[index])), fill="black")
    canvas.save(target)


GenerateBatch = Callable[[Tensor, Tensor], Tensor]


def sample_to_archive(
    *,
    output_directory: str | Path,
    num_samples: int,
    num_classes: int,
    image_size: int,
    checkpoint_hash: str,
    parameters: SamplingParameters,
    generate_batch: GenerateBatch,
    labels: Tensor | None = None,
    seeds: Tensor | None = None,
    plan_hash: str | None = None,
) -> Path:
    """Generate one deterministic, resumable archive across all initialized ranks."""
    parameters.validate()
    if num_samples < 1 or num_classes < 1 or image_size < 1:
        raise ValueError("sample, class, and image counts must be positive")
    if (labels is None) != (seeds is None):
        raise ValueError("explicit sampling labels and seeds must be supplied together")
    if labels is None or seeds is None:
        labels, seeds = sampling_plan(num_samples, num_classes, parameters.seed)
    labels, seeds = _validate_plan(labels, seeds, num_samples, num_classes)
    resolved_plan_hash = plan_hash or sampling_plan_hash(labels, seeds)
    if not resolved_plan_hash:
        raise ValueError("sampling plan hash cannot be empty")
    output = Path(output_directory)
    paths = _paths(output)
    base = {
        "schema_version": SAMPLING_MANIFEST_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_hash,
        "sampling_plan_hash": resolved_plan_hash,
        "num_samples": num_samples,
        "num_classes": num_classes,
        "image_size": image_size,
        "parameters": asdict(parameters),
    }
    rank, world_size, distributed = _distributed()
    setup: list[Any] = [False, 0, None]
    if rank == 0:
        try:
            setup[0], setup[1] = _initialize_or_resume(output, paths, base, labels)
        except Exception as error:
            setup[2] = f"{type(error).__name__}: {error}"
    if distributed:
        torch.distributed.broadcast_object_list(setup, src=0)
    if setup[2] is not None:
        raise RuntimeError(str(setup[2]))
    if setup[0]:
        return paths["archive"]

    _barrier()
    images, saved_labels, completion = _load_memmaps(paths)
    positions = torch.arange(rank, num_samples, world_size, dtype=torch.int64)
    pending = positions[
        torch.from_numpy(np.asarray(completion[positions.numpy()] == 0))
    ]
    started = time.perf_counter()
    done = 0
    for beginning in range(0, pending.numel(), parameters.batch_size):
        if rank == 0:
            done += min(parameters.batch_size, pending.numel() - beginning)
            elapsed = time.perf_counter() - started
            remaining = (elapsed / done) * (pending.numel() - done) if done else 0.0
            print(
                f"[{rank}/{world_size}] "
                f"sampling {done}/{pending.numel()} "
                f"({done / pending.numel():.1%}) "
                f"elapsed {elapsed:.1f}s "
                f"remaining {remaining:.1f}s",
                flush=True,
            )
        batch_positions = pending[beginning : beginning + parameters.batch_size]
        generated = generate_batch(labels[batch_positions], seeds[batch_positions])
        expected = batch_positions.numel()
        if generated.shape != (expected, 3, image_size, image_size):
            raise ValueError("sample generator returned an invalid image shape")
        pixels = generated.detach().float().clamp(0, 1).mul(255).round().byte()
        numpy_positions = batch_positions.numpy()
        images[numpy_positions] = pixels.permute(0, 2, 3, 1).cpu().numpy()
        images.flush()
        completion[numpy_positions] = 1
        completion.flush()
        if rank == 0:
            _write_json(
                paths["manifest"],
                base
                | {
                    "complete": False,
                    "completed_samples": int(np.count_nonzero(completion)),
                    "archive_retained": False,
                },
            )
    images.flush()
    completion.flush()
    del images, saved_labels, completion
    _barrier()
    finalization: list[str | None] = [None]
    if rank == 0:
        try:
            images, saved_labels, completion = _load_memmaps(paths)
            completed = int(np.count_nonzero(completion))
            if completed != num_samples or not np.all(completion == 1):
                raise RuntimeError(
                    f"sampling finished with {num_samples - completed} missing images"
                )
            _package(paths["images"], paths["labels"], paths["archive"])
            _save_preview(images[:16], saved_labels[:16], paths["preview"])
            _write_json(
                paths["manifest"],
                base
                | {
                    "complete": True,
                    "completed_samples": num_samples,
                    "archive_retained": True,
                    "wall_clock_seconds": time.perf_counter() - started,
                },
            )
            del images, saved_labels, completion
        except Exception as error:
            finalization[0] = f"{type(error).__name__}: {error}"
    if distributed:
        torch.distributed.broadcast_object_list(finalization, src=0)
    if finalization[0] is not None:
        raise RuntimeError(finalization[0])
    return paths["archive"]


def prune_sample_archive(output_directory: str | Path) -> None:
    """Remove large generated arrays after a successful in-training evaluation."""
    output = Path(output_directory)
    paths = _paths(output)
    if not (output / "evaluation.json").is_file():
        raise ValueError("sample archives can be pruned only after evaluation is saved")
    manifest = json.loads(paths["manifest"].read_text())
    if manifest.get("schema_version") != SAMPLING_MANIFEST_SCHEMA_VERSION:
        raise ValueError("sampling manifest schema is incompatible")
    if not manifest.get("complete"):
        raise ValueError("cannot prune an incomplete sampling run")
    for name in ("archive", "images", "labels", "completion"):
        paths[name].unlink(missing_ok=True)
    _write_json(paths["manifest"], manifest | {"archive_retained": False})
