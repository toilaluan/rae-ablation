from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

from .validation import ValidationPlan


@dataclass(frozen=True)
class SamplingParameters:
    batch_size: int = 128
    sampling_steps: int = 50
    solver: str = "euler"
    guidance_scale: float = 3.0
    timestep_shift: float = 1.0
    cfg_start: float = 0.0
    cfg_end: float = 1.0
    log_interval: int = 10

    @property
    def sampling_nfe(self) -> int:
        if self.solver == "euler":
            return self.sampling_steps
        if self.solver == "heun":
            return 2 * self.sampling_steps - 1
        raise ValueError("sampling solver must be euler or heun")

    def validate(self) -> None:
        if self.batch_size < 1 or self.sampling_steps < 1 or self.log_interval < 1:
            raise ValueError("sampling batch size, steps, and log interval must be positive")
        if self.solver not in {"euler", "heun"}:
            raise ValueError("sampling solver must be euler or heun")
        if not math.isfinite(self.timestep_shift) or self.timestep_shift <= 0:
            raise ValueError("timestep shift must be finite and positive")
        if not math.isfinite(self.guidance_scale):
            raise ValueError("guidance scale must be finite")
        if not 0 <= self.cfg_start <= self.cfg_end <= 1:
            raise ValueError("CFG interval must satisfy 0 <= start <= end <= 1")


GenerateBatch = Callable[[Tensor, Tensor, Tensor], Tensor]
SAMPLING_MANIFEST_SCHEMA_VERSION = 17
FEATURE_SCHEMA_VERSION = 10


def _float_slug(value: float) -> str:
    return repr(float(value)).replace("-", "m").replace("+", "").replace(".", "p")


def sampling_output_name(sample_count: int, parameters: SamplingParameters) -> str:
    parameters.validate()
    if sample_count < 1:
        raise ValueError("sample count must be positive")
    return (
        f"samples-{parameters.solver}-n{sample_count}-steps{parameters.sampling_steps}"
        f"-cfg{_float_slug(parameters.guidance_scale)}"
        f"-shift{_float_slug(parameters.timestep_shift)}"
        f"-window{_float_slug(parameters.cfg_start)}-{_float_slug(parameters.cfg_end)}"
    )


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
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_sampling_plan(
    plan: ValidationPlan, sample_count: int, num_classes: int
) -> tuple[Tensor, Tensor, Tensor]:
    if sample_count < 1 or sample_count % num_classes:
        raise ValueError("sample count must be positive and divisible by model.num_classes")
    if min(plan.labels.numel(), plan.sample_ids.numel(), plan.noise_seeds.numel()) < sample_count:
        raise ValueError(f"sampling plan does not contain {sample_count} samples")
    labels = plan.labels[:sample_count].to(torch.int64)
    sample_ids = plan.sample_ids[:sample_count].to(torch.int64)
    noise_seeds = plan.noise_seeds[:sample_count].to(torch.int64)
    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError("sampling plan contains a label outside model.num_classes")
    counts = torch.bincount(labels, minlength=num_classes)
    expected = sample_count // num_classes
    if counts.numel() != num_classes or not torch.all(counts == expected):
        raise ValueError(f"sampling plan must contain exactly {expected} samples per class")
    if sample_ids.unique().numel() != sample_count or torch.any(sample_ids < 0):
        raise ValueError("sampling plan sample IDs must be unique and non-negative")
    return labels, sample_ids, noise_seeds


def _write_json_atomic(path: Path, values: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".partial-{os.getpid()}")
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _distributed() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _peak_memory_gib() -> float:
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def _paths(output_dir: Path) -> dict[str, Path]:
    return {
        "archive": output_dir / "samples.npz",
        "manifest": output_dir / "samples.json",
        "preview": output_dir / "preview.png",
        "images": output_dir / ".samples-arr_0.npy",
        "labels": output_dir / ".samples-arr_1.npy",
        "completion": output_dir / ".samples-completion.npy",
    }


def _manifest_base(
    *,
    checkpoint_hash: str,
    plan: ValidationPlan,
    labels: Tensor,
    sample_ids: Tensor,
    sample_count: int,
    num_classes: int,
    image_size: int,
    patch_size: int,
    grid_size: int,
    model_variant: str,
    feature_schema: dict[str, Any],
    parameters: SamplingParameters,
) -> dict[str, Any]:
    counts = torch.bincount(labels, minlength=num_classes)
    ids_digest = hashlib.sha256(sample_ids.numpy().tobytes()).hexdigest()
    return {
        "schema_version": SAMPLING_MANIFEST_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_hash,
        "validation_plan_hash": plan.plan_hash,
        "sample_ids_sha256": ids_digest,
        "image_size": image_size,
        "patch_size": patch_size,
        "jit_grid_size": grid_size,
        "model_variant": model_variant,
        "feature_schema": feature_schema,
        "sample_count": sample_count,
        "num_classes": num_classes,
        "class_counts": counts.tolist(),
        "image_shape": [image_size, image_size, 3],
        "image_dtype": "uint8",
        "sampling": {
            "solver": parameters.solver,
            "sampling_steps": parameters.sampling_steps,
            "sampling_nfe": parameters.sampling_nfe,
            "guidance_scale": parameters.guidance_scale,
            "timestep_shift": parameters.timestep_shift,
            "cfg_interval": [parameters.cfg_start, parameters.cfg_end],
        },
    }


def _load_memmaps(paths: dict[str, Path], mode: str = "r+") -> tuple[np.memmap, ...]:
    images = np.load(paths["images"], mmap_mode=mode, allow_pickle=False)
    labels = np.load(paths["labels"], mmap_mode=mode, allow_pickle=False)
    completion = np.load(paths["completion"], mmap_mode=mode, allow_pickle=False)
    if not all(isinstance(value, np.memmap) for value in (images, labels, completion)):
        raise RuntimeError("temporary sampling arrays are not memory mapped")
    return images, labels, completion


def _initialize_or_resume(
    output_dir: Path,
    paths: dict[str, Path],
    base: dict[str, Any],
    labels: Tensor,
) -> tuple[bool, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if paths["manifest"].exists():
        existing = json.loads(paths["manifest"].read_text())
        if existing.get("schema_version") != SAMPLING_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "sampling manifest uses an incompatible schema; old manifests cannot be resumed"
            )
        comparable = {key: existing.get(key) for key in base}
        if comparable != base:
            raise ValueError("existing sampling manifest does not match this request")
        if existing.get("complete"):
            if not paths["archive"].exists() or not paths["preview"].exists():
                raise ValueError("completed sampling manifest is missing final artifacts")
            for key in ("images", "labels", "completion"):
                paths[key].unlink(missing_ok=True)
            return True, int(base["sample_count"])
        if not all(paths[key].exists() for key in ("images", "labels", "completion")):
            raise ValueError("interrupted sampling run is missing temporary storage")
        images_mm, labels_mm, completion_mm = _load_memmaps(paths)
        image_size = int(base["image_size"])
        expected_images = (base["sample_count"], image_size, image_size, 3)
        if images_mm.shape != expected_images or images_mm.dtype != np.uint8:
            raise ValueError("temporary image array does not match the sampling manifest")
        if labels_mm.shape != (base["sample_count"],) or labels_mm.dtype != np.int64:
            raise ValueError("temporary label array does not match the sampling manifest")
        if completion_mm.shape != (base["sample_count"],) or completion_mm.dtype != np.uint8:
            raise ValueError("temporary completion bitmap does not match the sampling manifest")
        if not np.array_equal(labels_mm, labels.numpy()):
            raise ValueError("temporary labels do not match the sampling plan")
        completed = int(np.count_nonzero(completion_mm))
        if paths["archive"].exists():
            if completed != int(base["sample_count"]):
                raise ValueError("final sample archive exists before sampling completed")
            with zipfile.ZipFile(paths["archive"]) as archive:
                if set(archive.namelist()) != {"arr_0.npy", "arr_1.npy"}:
                    raise ValueError("final sample archive has unexpected members")
                if archive.getinfo("arr_0.npy").file_size != paths["images"].stat().st_size:
                    raise ValueError("final sample archive image payload has the wrong size")
                if archive.getinfo("arr_1.npy").file_size != paths["labels"].stat().st_size:
                    raise ValueError("final sample archive label payload has the wrong size")
            save_preview(images_mm[:16], labels_mm[:16], paths["preview"])
            _write_json_atomic(
                paths["manifest"],
                {**base, "complete": True, "completed_samples": completed},
            )
            del images_mm, labels_mm, completion_mm
            for key in ("images", "labels", "completion"):
                paths[key].unlink()
            return True, completed
        return False, completed

    if any(paths[key].exists() for key in ("archive", "images", "labels", "completion")):
        raise ValueError("refusing to sample into untracked existing artifacts")
    count = int(base["sample_count"])
    images_mm = np.lib.format.open_memmap(
        paths["images"],
        mode="w+",
        dtype=np.uint8,
        shape=(count, int(base["image_size"]), int(base["image_size"]), 3),
    )
    labels_mm = np.lib.format.open_memmap(
        paths["labels"], mode="w+", dtype=np.int64, shape=(count,)
    )
    completion_mm = np.lib.format.open_memmap(
        paths["completion"], mode="w+", dtype=np.uint8, shape=(count,)
    )
    labels_mm[:] = labels.numpy()
    completion_mm[:] = 0
    images_mm.flush()
    labels_mm.flush()
    completion_mm.flush()
    _write_json_atomic(paths["manifest"], {**base, "complete": False, "completed_samples": 0})
    return False, 0


def _as_uint8_nhwc(images: Tensor, expected: int, image_size: int) -> np.ndarray:
    if images.ndim != 4 or images.shape != (expected, 3, image_size, image_size):
        raise ValueError(
            f"generator must return {expected}x3x{image_size}x{image_size} images, "
            f"received {tuple(images.shape)}"
        )
    if not images.is_floating_point():
        raise ValueError("generator output must be floating point in [0, 1]")
    pixels = images.detach().float().clamp(0, 1).mul(255).round().byte().cpu()
    return pixels.permute(0, 2, 3, 1).contiguous().numpy()


def _write_zip_member(archive: zipfile.ZipFile, source: Path, name: str) -> None:
    with (
        source.open("rb") as input_handle,
        archive.open(name, mode="w", force_zip64=True) as output_handle,
    ):
        shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)


def package_npz(images_path: Path, labels_path: Path, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + f".partial-{os.getpid()}")
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:
            _write_zip_member(archive, images_path, "arr_0.npy")
            _write_zip_member(archive, labels_path, "arr_1.npy")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def save_preview(images: np.ndarray, labels: np.ndarray, target: Path) -> None:
    count = min(16, len(images))
    if count < 1:
        raise ValueError("cannot create a preview without samples")
    height, width = images.shape[1:3]
    caption_height = 20
    canvas = Image.new("RGB", (4 * width, 4 * (height + caption_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index in range(count):
        row, column = divmod(index, 4)
        x, y = column * width, row * (height + caption_height)
        canvas.paste(Image.fromarray(np.asarray(images[index]), mode="RGB"), (x, y))
        draw.text((x + 3, y + height + 3), f"class {int(labels[index])}", fill="black")
    temporary = target.with_suffix(target.suffix + f".partial-{os.getpid()}")
    canvas.save(temporary, format="PNG")
    os.replace(temporary, target)


def sample_to_archive(
    *,
    output_dir: str | Path,
    plan: ValidationPlan,
    sample_count: int,
    num_classes: int,
    image_size: int,
    patch_size: int,
    grid_size: int,
    model_variant: str,
    feature_schema: dict[str, Any],
    checkpoint_hash: str,
    parameters: SamplingParameters,
    generate_batch: GenerateBatch,
) -> Path:
    """Generate a resumable, class-balanced ADM-compatible sample archive."""
    parameters.validate()
    if image_size < 1 or patch_size < 1 or grid_size < 1:
        raise ValueError("image size, patch size, and model grid must be positive")
    if not model_variant:
        raise ValueError("model variant cannot be empty")
    if feature_schema.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"sampling requires a schema-v{FEATURE_SCHEMA_VERSION} feature description"
        )
    labels, sample_ids, noise_seeds = validate_sampling_plan(plan, sample_count, num_classes)
    output_dir = Path(output_dir)
    paths = _paths(output_dir)
    base = _manifest_base(
        checkpoint_hash=checkpoint_hash,
        plan=plan,
        labels=labels,
        sample_ids=sample_ids,
        sample_count=sample_count,
        num_classes=num_classes,
        image_size=image_size,
        patch_size=patch_size,
        grid_size=grid_size,
        model_variant=model_variant,
        feature_schema=feature_schema,
        parameters=parameters,
    )
    rank, world_size = _distributed()
    state: list[Any] = [False, 0, None]
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not distributed:
        state[0], state[1] = _initialize_or_resume(output_dir, paths, base, labels)
    elif rank == 0:
        try:
            state[0], state[1] = _initialize_or_resume(output_dir, paths, base, labels)
        except Exception as error:  # broadcast rank-zero setup failures before any barrier
            state[2] = f"{type(error).__name__}: {error}"
    if distributed:
        torch.distributed.broadcast_object_list(state, src=0)
    if state[2] is not None:
        raise RuntimeError(state[2])
    if rank == 0:
        status = "complete" if state[0] else "resuming" if state[1] else "new"
        print(
            f"sampling configuration: samples={sample_count} classes={num_classes} "
            f"batch_size={parameters.batch_size} solver={parameters.solver} "
            f"steps={parameters.sampling_steps} nfe={parameters.sampling_nfe} "
            f"guidance={parameters.guidance_scale:g} "
            f"timestep_shift={parameters.timestep_shift:g} "
            f"cfg_interval=[{parameters.cfg_start:g}, {parameters.cfg_end:g}]"
        )
        print(f"sampling state: {status}, completed={state[1]}/{sample_count}")
    if state[0]:
        if rank == 0:
            print(f"preview: {paths['preview']}")
            print(f"sample archive: {paths['archive']}")
        return paths["archive"]

    _barrier()
    images_mm, _labels_mm, completion_mm = _load_memmaps(paths)
    local_positions = torch.arange(rank, sample_count, world_size, dtype=torch.int64)
    pending = local_positions[
        torch.from_numpy(np.asarray(completion_mm[local_positions.numpy()] == 0))
    ]
    started = time.perf_counter()
    completed_at_start = int(state[1])
    for batch_index, begin in enumerate(range(0, pending.numel(), parameters.batch_size), start=1):
        positions = pending[begin : begin + parameters.batch_size]
        decoded = generate_batch(positions, labels[positions], noise_seeds[positions])
        images_mm[positions.numpy()] = _as_uint8_nhwc(decoded, positions.numel(), image_size)
        images_mm.flush()
        completion_mm[positions.numpy()] = 1
        completion_mm.flush()
        if rank == 0 and batch_index % parameters.log_interval == 0:
            complete = int(np.count_nonzero(completion_mm))
            elapsed = max(time.perf_counter() - started, 1e-9)
            rate = (complete - completed_at_start) / elapsed
            eta = (sample_count - complete) / rate if rate > 0 else float("inf")
            _write_json_atomic(
                paths["manifest"],
                {**base, "complete": False, "completed_samples": complete},
            )
            print(
                f"sampling progress: {complete}/{sample_count} images, {rate:.2f} images/s, "
                f"ETA={eta:.0f}s, peak_memory={_peak_memory_gib():.2f} GiB"
            )

    images_mm.flush()
    completion_mm.flush()
    del images_mm, _labels_mm, completion_mm
    _barrier()
    if rank == 0:
        images_mm, labels_mm, completion_mm = _load_memmaps(paths)
        complete = int(np.count_nonzero(completion_mm))
        if complete != sample_count or not np.all(completion_mm == 1):
            raise RuntimeError(f"sampling finished with {sample_count - complete} missing images")
        package_npz(paths["images"], paths["labels"], paths["archive"])
        save_preview(images_mm[:16], labels_mm[:16], paths["preview"])
        _write_json_atomic(
            paths["manifest"],
            {**base, "complete": True, "completed_samples": sample_count},
        )
        del images_mm, labels_mm, completion_mm
        for key in ("images", "labels", "completion"):
            paths[key].unlink()
        print(f"preview: {paths['preview']}")
        print(f"sample archive: {paths['archive']}")
    _barrier()
    return paths["archive"]
