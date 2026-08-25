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

SAMPLING_MANIFEST_SCHEMA_VERSION = 2


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

    @property
    def sampling_nfe(self) -> int:
        return self.steps if self.solver == "euler" else 2 * self.steps - 1

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


def _write_zip_member(archive: zipfile.ZipFile, source: Path, name: str) -> None:
    with source.open("rb") as source_handle, archive.open(
        name, mode="w", force_zip64=True
    ) as target_handle:
        shutil.copyfileobj(source_handle, target_handle, length=8 * 1024 * 1024)


def _package(images: Path, labels: Path, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + f".partial-{os.getpid()}")
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        _write_zip_member(archive, images, "arr_0.npy")
        _write_zip_member(archive, labels, "arr_1.npy")
    os.replace(temporary, target)


def _save_preview(images: np.ndarray, labels: np.ndarray, target: Path) -> None:
    count = min(16, len(images))
    side = images.shape[1]
    canvas = Image.new("RGB", (4 * side, 4 * (side + 18)), "white")
    draw = ImageDraw.Draw(canvas)
    for index in range(count):
        row, column = divmod(index, 4)
        x, y = column * side, row * (side + 18)
        canvas.paste(Image.fromarray(images[index]), (x, y))
        draw.text((x + 2, y + side + 1), str(int(labels[index])), fill="black")
    canvas.save(target)


save_preview = _save_preview


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
) -> Path:
    parameters.validate()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    labels, seeds = sampling_plan(num_samples, num_classes, parameters.seed)
    plan_hash = sampling_plan_hash(labels, seeds)
    images_path = output / ".images.npy"
    labels_path = output / ".labels.npy"
    completion_path = output / ".completion.npy"
    archive_path = output / "samples.npz"
    manifest_path = output / "samples.json"
    base = {
        "schema_version": SAMPLING_MANIFEST_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_hash,
        "sampling_plan_hash": plan_hash,
        "num_samples": num_samples,
        "num_classes": num_classes,
        "image_size": image_size,
        "parameters": asdict(parameters),
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if {key: manifest.get(key) for key in base} != base:
            raise ValueError("existing sample manifest does not match this request")
        if manifest.get("complete"):
            if not archive_path.is_file():
                raise ValueError("completed sample manifest is missing its archive")
            return archive_path
    elif any(path.exists() for path in (images_path, labels_path, completion_path, archive_path)):
        raise ValueError("refusing to overwrite untracked sample artifacts")
    else:
        images = np.lib.format.open_memmap(
            images_path, mode="w+", dtype=np.uint8, shape=(num_samples, image_size, image_size, 3)
        )
        saved_labels = np.lib.format.open_memmap(
            labels_path, mode="w+", dtype=np.int64, shape=(num_samples,)
        )
        completion = np.lib.format.open_memmap(
            completion_path, mode="w+", dtype=np.uint8, shape=(num_samples,)
        )
        saved_labels[:] = labels.numpy()
        completion[:] = 0
        images.flush()
        saved_labels.flush()
        completion.flush()
        _write_json(manifest_path, base | {"complete": False, "completed_samples": 0})
    images = np.load(images_path, mmap_mode="r+")
    saved_labels = np.load(labels_path, mmap_mode="r+")
    completion = np.load(completion_path, mmap_mode="r+")
    if not np.array_equal(saved_labels, labels.numpy()):
        raise ValueError("saved labels differ from the deterministic sampling plan")
    started = time.perf_counter()
    for beginning in range(0, num_samples, parameters.batch_size):
        end = min(beginning + parameters.batch_size, num_samples)
        if np.all(completion[beginning:end] == 1):
            continue
        generated = generate_batch(labels[beginning:end], seeds[beginning:end])
        if generated.shape != (end - beginning, 3, image_size, image_size):
            raise ValueError("sample generator returned an invalid image shape")
        pixels = generated.detach().float().clamp(0, 1).mul(255).round().byte()
        images[beginning:end] = pixels.permute(0, 2, 3, 1).cpu().numpy()
        images.flush()
        completion[beginning:end] = 1
        completion.flush()
        _write_json(
            manifest_path,
            base | {"complete": False, "completed_samples": int(np.count_nonzero(completion))},
        )
    _package(images_path, labels_path, archive_path)
    _save_preview(images[:16], saved_labels[:16], output / "preview.png")
    elapsed = time.perf_counter() - started
    _write_json(
        manifest_path,
        base
        | {
            "complete": True,
            "completed_samples": num_samples,
            "wall_clock_seconds": elapsed,
        },
    )
    return archive_path
