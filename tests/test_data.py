from __future__ import annotations

import numpy as np
import torch

from cascade_dit.config import ExperimentConfig
from cascade_dit.data import (
    JointBatchPreprocessor,
    JointDataset,
    joint_collator,
    prepare_pixels,
    set_joint_loader_epoch,
)
from cascade_dit.representations import RepresentationManifest


def _manifest(size: int = 2) -> RepresentationManifest:
    channel = torch.arange(768, dtype=torch.float32)
    return RepresentationManifest(
        dataset_size=size,
        dataset_fingerprint="dataset",
        encoder_fingerprint="encoder",
        preparation_fingerprint="preparation",
        stages=[
            {"name": "dino12", "source": {"kind": "dino", "layer": 12}},
            {"name": "rgb", "source": {"kind": "rgb", "layer": None}},
        ],
        statistics={
            "dino12": {
                "mean": (channel / 100).tolist(),
                "std": (channel / 200 + 1).tolist(),
                "count": 10,
            },
            "rgb": {
                "mean": (channel / 80 + 1).tolist(),
                "std": (channel / 160 + 2).tolist(),
                "count": 10,
            },
        },
        cache_files={"dino12": "dino12.bf16"},
        completion_files={"dino12": "dino12.completion.u8"},
        complete=True,
    )


def test_batch_preprocessing_normalizes_only_semantics_and_preserves_raw_rgb() -> None:
    config = ExperimentConfig()
    manifest = _manifest()
    generator = torch.Generator().manual_seed(17)
    semantic = torch.randn(64, 768, generator=generator).bfloat16()
    pixels = torch.randint(
        0, 256, (3, 128, 128), dtype=torch.uint8, generator=generator
    )
    raw = joint_collator([(semantic, pixels, 4)])
    prepared = JointBatchPreprocessor(config, manifest, "cpu")(raw)

    semantic_stats = manifest.statistics["dino12"]
    expected_semantic = (
        semantic.float() - torch.tensor(semantic_stats["mean"])
    ) / torch.tensor(semantic_stats["std"])
    assert torch.allclose(prepared.semantic[0], expected_semantic)
    assert torch.equal(prepared.pixels, prepare_pixels(pixels[None], "cpu"))
    assert prepared.pixels.min() >= 0 and prepared.pixels.max() <= 1
    assert prepared.pixels.shape == (1, 3, 128, 128)
    assert not hasattr(prepared, "pixel_ratio")


class _TinyDataset:
    def __len__(self) -> int:
        return 2

    def get_item(self, index: int):
        pixels = torch.full((3, 128, 128), index, dtype=torch.uint8)
        return pixels, index, index


def _write_cache(tmp_path) -> None:
    values = np.memmap(
        tmp_path / "dino12.bf16",
        mode="w+",
        dtype=np.uint16,
        shape=(2, 64, 768),
    )
    source = torch.arange(2, dtype=torch.bfloat16)[:, None, None].expand(2, 64, 768)
    values[:] = source.contiguous().view(torch.uint16).numpy()
    values.flush()


def test_joint_dataset_always_loads_semantic_rgb_and_label(tmp_path) -> None:
    _write_cache(tmp_path)
    dataset = JointDataset(
        _TinyDataset(),  # type: ignore[arg-type]
        ExperimentConfig(),
        _manifest(),
        tmp_path,
    )
    semantic, pixels, label = dataset[1]
    assert semantic.shape == (64, 768) and torch.all(semantic == 1)
    assert pixels.shape == (3, 128, 128) and pixels.dtype == torch.uint8
    assert label == 1
    assert not hasattr(dataset, "set_phase")


def test_loader_epoch_changes_only_distributed_sampler(tmp_path) -> None:
    _write_cache(tmp_path)
    dataset = JointDataset(
        _TinyDataset(),  # type: ignore[arg-type]
        ExperimentConfig(),
        _manifest(),
        tmp_path,
    )
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=1, rank=0, shuffle=True
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)
    set_joint_loader_epoch(loader, 7)
    assert sampler.epoch == 7
    assert not any(name.startswith("_phase") for name in vars(dataset))
