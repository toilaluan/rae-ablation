from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .codecs import CompactRAECodec, load_codecs
from .config import CacheConfig
from .constants import IMAGENET_REPO, IMAGENET_REVISION, LATENT_SPECS
from .latents import ChannelMoments


@dataclass
class CacheManifest:
    version: int
    dataset: str
    dataset_revision: str
    split: str
    preprocessing: list[str]
    dtype: str
    shard_size: int
    sample_count: int
    codecs: dict[str, dict[str, Any]]
    completed_shards: list[int] = field(default_factory=list)

    def immutable(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("completed_shards")
        return value

    @property
    def fingerprint(self) -> str:
        blob = json.dumps(self.immutable(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_manifest(path: str | Path) -> CacheManifest:
    return CacheManifest(**json.loads(Path(path).read_text()))


def initialize_manifest(root: Path, expected: CacheManifest) -> CacheManifest:
    path = root / "manifest.json"
    print(path)
    root.mkdir(parents=True, exist_ok=True)
    for partial in root.glob("*.partial"):
        partial.unlink()
    if path.exists():
        existing = read_manifest(path)
        if existing.immutable() != expected.immutable():
            raise ValueError("cache manifest does not match requested dataset/codecs/layout")
        return existing
    atomic_json(path, asdict(expected))
    return expected


def _codec_metadata(name: str) -> dict[str, Any]:
    from .constants import COMPACT_REPO, COMPACT_REVISION, RAE_REPO, RAE_REVISION

    channels, height, width = LATENT_SPECS[name]
    result: dict[str, Any] = {"shape": [channels, height, width]}
    if name == "rae":
        result.update(repository=RAE_REPO, revision=RAE_REVISION)
    else:
        result.update(
            repository=COMPACT_REPO,
            revision=COMPACT_REVISION,
            base_repository=RAE_REPO,
            base_revision=RAE_REVISION,
        )
    return result


def expected_manifest(config: CacheConfig, sample_count: int) -> CacheManifest:
    names = ["rae", "rae_compact"] if config.codec == "both" else [config.codec]
    return CacheManifest(
        version=1,
        dataset=IMAGENET_REPO,
        dataset_revision=IMAGENET_REVISION,
        split=config.split,
        preprocessing=["PIL.convert(RGB)", "tensor", "scale_uint8_to_[0,1]"],
        dtype="float16",
        shard_size=config.shard_size,
        sample_count=sample_count,
        codecs={name: _codec_metadata(name) for name in names},
    )


def cache_bytes(sample_count: int, codecs: list[str]) -> int:
    return sum(sample_count * math.prod(LATENT_SPECS[name]) * 2 for name in codecs)


def disk_preflight(root: Path, required: int) -> None:
    available = shutil.disk_usage(root.parent if root.parent.exists() else Path.cwd()).free
    if available < required:
        raise OSError(
            f"insufficient cache disk: need {required / 2**30:.1f} GiB, "
            f"have {available / 2**30:.1f} GiB"
        )


class _ImageDataset(Dataset[tuple[Tensor, int, int]]):
    def __init__(self, dataset: Any, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Tensor, int, int]:
        index = self.indices[item]
        row = self.dataset[index]
        image = np.asarray(row["image"].convert("RGB"), dtype=np.uint8)
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255)
        return tensor, int(row["label"]), index


@dataclass
class PendingShard:
    shard_id: int
    tensors: dict[str, list[Tensor]]
    events: list[torch.cuda.Event]


class AsyncShardWriter:
    def __init__(self, root: Path, max_pending: int = 3) -> None:
        self.root = root
        self.queue: queue.Queue[PendingShard | None] = queue.Queue(max_pending)
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="latent-shard-writer", daemon=True)
        self.thread.start()

    def submit(self, shard: PendingShard) -> None:
        if self.error is not None:
            raise RuntimeError("cache writer failed") from self.error
        self.queue.put(shard)

    def _run(self) -> None:
        from safetensors.torch import save_file

        try:
            while (pending := self.queue.get()) is not None:
                for event in pending.events:
                    event.synchronize()
                tensors = {
                    name: torch.cat(parts).contiguous() for name, parts in pending.tensors.items()
                }
                target = self.root / f"shard-{pending.shard_id:06d}.safetensors"
                partial = target.with_suffix(".partial")
                save_file(tensors, str(partial), metadata={"shard_id": str(pending.shard_id)})
                os.replace(partial, target)
                self.queue.task_done()
        except BaseException as error:  # propagated to the producer on close
            self.error = error

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("cache writer failed") from self.error


def _world() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _cpu_copy(
    tensor: Tensor, stream: torch.cuda.Stream | None
) -> tuple[Tensor, torch.cuda.Event | None]:
    if tensor.device.type != "cuda":
        return tensor.detach().to(device="cpu", dtype=torch.float16), None
    cpu = torch.empty(tensor.shape, dtype=torch.float16, device="cpu", pin_memory=True)
    assert stream is not None
    stream.wait_stream(torch.cuda.current_stream(tensor.device))
    with torch.cuda.stream(stream):
        cpu.copy_(tensor.detach(), non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)
    return cpu, event


@torch.inference_mode()
def build_cache(config: CacheConfig, device: str | torch.device | None = None) -> CacheManifest:
    """Create resumable, rank-sharded fp16 caches with overlapped CUDA D2H and writes."""
    from datasets import load_dataset

    rank, world_size = _world()
    if device is None:
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset = load_dataset(
        IMAGENET_REPO,
        split=config.split,
        revision=IMAGENET_REVISION,
    )
    sample_count = min(len(dataset), config.max_samples or len(dataset))
    root = Path(config.directory)
    expected = expected_manifest(config, sample_count)
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        initialize_manifest(root, expected)
        completed = verify_cache(root, strict=False)
        cached_samples = sum(
            min(config.shard_size, sample_count - shard_id * config.shard_size)
            for shard_id in completed
        )
        remaining_samples = sample_count - cached_samples
        disk_preflight(root, int(cache_bytes(remaining_samples, list(expected.codecs)) * 1.01))
    if world_size > 1:
        torch.distributed.barrier()
    manifest = read_manifest(root / "manifest.json")

    loaded = load_codecs(config.codec, str(device))
    if isinstance(loaded, tuple):
        rae, compact = loaded
    elif isinstance(loaded, CompactRAECodec):
        compact, rae = loaded, loaded.base
    else:
        rae, compact = loaded, None

    writer = AsyncShardWriter(root, config.output_buffers)
    d2h_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    shard_count = math.ceil(sample_count / config.shard_size)
    started = time.perf_counter()
    processed = 0
    for shard_id in range(rank, shard_count, world_size):
        target = root / f"shard-{shard_id:06d}.safetensors"
        if target.exists():
            continue
        begin = shard_id * config.shard_size
        end = min(begin + config.shard_size, sample_count)
        loader = DataLoader(
            _ImageDataset(dataset, list(range(begin, end))),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=config.workers > 0,
            prefetch_factor=config.prefetch_factor if config.workers > 0 else None,
        )
        parts: dict[str, list[Tensor]] = {name: [] for name in expected.codecs}
        parts.update(labels=[], indices=[])
        events: list[torch.cuda.Event] = []
        for images, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                rae_latent = rae.encode(images)
                compact_latent = compact.compress(rae_latent) if compact is not None else None
            if "rae" in parts:
                copied, event = _cpu_copy(rae_latent, d2h_stream)
                parts["rae"].append(copied)
                if event is not None:
                    events.append(event)
            if "rae_compact" in parts:
                assert compact_latent is not None
                copied, event = _cpu_copy(compact_latent, d2h_stream)
                parts["rae_compact"].append(copied)
                if event is not None:
                    events.append(event)
            parts["labels"].append(labels.to(torch.int64))
            parts["indices"].append(indices.to(torch.int64))
            processed += images.shape[0]
        writer.submit(PendingShard(shard_id, parts, events))
        elapsed = max(time.perf_counter() - started, 1e-9)
        print(
            f"rank={rank} shard={shard_id + 1}/{shard_count} "
            f"images/s={processed / elapsed:.1f} writer_queue={writer.queue.qsize()}/{config.output_buffers}",
            flush=True,
        )
    writer.close()
    if world_size > 1:
        torch.distributed.barrier()

    completed = verify_cache(root, strict=config.verify)
    manifest.completed_shards = completed
    if rank == 0:
        atomic_json(root / "manifest.json", asdict(manifest))
        compute_cache_statistics(root, manifest)
    if world_size > 1:
        torch.distributed.barrier()
    return manifest


def verify_cache(root: str | Path, strict: bool = True) -> list[int]:
    from safetensors import safe_open

    root = Path(root)
    manifest = read_manifest(root / "manifest.json")
    shard_count = math.ceil(manifest.sample_count / manifest.shard_size)
    completed: list[int] = []
    seen: set[int] = set()
    for shard_id in range(shard_count):
        path = root / f"shard-{shard_id:06d}.safetensors"
        if not path.exists():
            if strict:
                raise FileNotFoundError(f"missing cache shard {shard_id}")
            continue
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            expected_keys = {*manifest.codecs, "labels", "indices"}
            if keys != expected_keys:
                raise ValueError(f"shard {shard_id} keys are {keys}, expected {expected_keys}")
            indices = handle.get_tensor("indices")
            labels = handle.get_tensor("labels")
            expected_n = min(
                manifest.shard_size, manifest.sample_count - shard_id * manifest.shard_size
            )
            if indices.shape != labels.shape or indices.numel() != expected_n:
                raise ValueError(f"shard {shard_id} has invalid sample count")
            if set(indices.tolist()) & seen:
                raise ValueError("cache contains duplicate source indices")
            seen.update(indices.tolist())
            for name, metadata in manifest.codecs.items():
                tensor = handle.get_tensor(name)
                if tensor.dtype != torch.float16 or list(tensor.shape[1:]) != metadata["shape"]:
                    raise ValueError(f"shard {shard_id} has invalid {name} tensor")
        completed.append(shard_id)
    if strict and seen != set(range(manifest.sample_count)):
        raise ValueError("cache source indices are incomplete")
    return completed


def compute_cache_statistics(root: str | Path, manifest: CacheManifest | None = None) -> None:
    from safetensors.torch import load_file

    root = Path(root)
    manifest = manifest or read_manifest(root / "manifest.json")
    moments = {
        name: ChannelMoments(metadata["shape"][0]) for name, metadata in manifest.codecs.items()
    }
    for shard_id in manifest.completed_shards:
        tensors = load_file(str(root / f"shard-{shard_id:06d}.safetensors"), device="cpu")
        for name, accumulator in moments.items():
            accumulator.update(tensors[name])
    for name, accumulator in moments.items():
        accumulator.finalize().save(root / f"stats-{name}.safetensors")


class ShardedLatentDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Map-style cache reader with one-shard process-local memoization."""

    def __init__(self, root: str | Path, codec: str) -> None:
        self.root = Path(root)
        self.manifest = read_manifest(self.root / "manifest.json")
        if codec not in self.manifest.codecs:
            raise ValueError(f"cache does not contain {codec}")
        self.codec = codec
        self._loaded_id = -1
        self._loaded: dict[str, Tensor] = {}

    def __len__(self) -> int:
        return self.manifest.sample_count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        from safetensors.torch import load_file

        shard_id, offset = divmod(index, self.manifest.shard_size)
        if shard_id != self._loaded_id:
            self._loaded = load_file(str(self.root / f"shard-{shard_id:06d}.safetensors"))
            self._loaded_id = shard_id
        return (
            self._loaded[self.codec][offset],
            self._loaded["labels"][offset],
            self._loaded["indices"][offset],
        )
