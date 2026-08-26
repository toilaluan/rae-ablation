from __future__ import annotations

import numpy as np
import pytest
import torch

from cascade_dit.config import ExperimentConfig
from cascade_dit.data import (
    EndpointBatchPreprocessor,
    EndpointDataset,
    endpoint_collator,
    prepare_pixels,
    sample_pixel_ratio,
    set_endpoint_loader_epoch,
)
from cascade_dit.representations import RepresentationManifest, patchify_rgb


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


def test_semantic_selection_is_deterministic_ninety_ten_and_bounded() -> None:
    config = ExperimentConfig()
    ratios = [sample_pixel_ratio(config, 3, index) for index in range(20_000)]
    mixed = [ratio for ratio in ratios if ratio > 0]
    assert len(mixed) / len(ratios) == pytest.approx(0.1, abs=0.01)
    assert all(0 < ratio < 1 for ratio in mixed)
    assert sample_pixel_ratio(config, 3, 17) == ratios[17]
    assert sample_pixel_ratio(config, 3, 17, phase="pixel") == 1


def test_endpoint_preprocessing_normalizes_then_mixes() -> None:
    config = ExperimentConfig()
    manifest = _manifest()
    generator = torch.Generator().manual_seed(17)
    semantic = torch.randn(64, 768, generator=generator).bfloat16()
    pixels = torch.randint(
        0, 256, (3, 128, 128), dtype=torch.uint8, generator=generator
    )
    raw = endpoint_collator(
        [(semantic, pixels, torch.tensor(0.25), 4)]
    )
    prepared = EndpointBatchPreprocessor(
        config, manifest, "cpu", load_semantic=True
    )(raw)

    semantic_stats = manifest.statistics["dino12"]
    pixel_stats = manifest.statistics["rgb"]
    expected_semantic = (
        semantic.float() - torch.tensor(semantic_stats["mean"])
    ) / torch.tensor(semantic_stats["std"])
    pixel_tokens = patchify_rgb(prepare_pixels(pixels[None], "cpu"))[0]
    expected_pixel = (
        pixel_tokens - torch.tensor(pixel_stats["mean"])
    ) / torch.tensor(pixel_stats["std"])
    assert torch.allclose(prepared.semantic[0], expected_semantic)
    assert torch.allclose(prepared.pixel[0], expected_pixel)
    assert torch.allclose(
        prepared.clean[0], torch.lerp(expected_semantic, expected_pixel, 0.25)
    )


class _TinyDataset:
    def __init__(self) -> None:
        self.loads: list[bool] = []

    def __len__(self) -> int:
        return 2

    def get_item(self, index: int):
        self.loads.append(True)
        pixels = torch.zeros(3, 128, 128, dtype=torch.uint8)
        return pixels, index, index


def _write_cache(tmp_path) -> None:
    values = np.memmap(
        tmp_path / "dino12.bf16",
        mode="w+",
        dtype=np.uint16,
        shape=(2, 64, 768),
    )
    values[:] = torch.zeros(2, 64, 768, dtype=torch.bfloat16).view(torch.uint16).numpy()
    values.flush()


def test_pure_semantic_items_still_load_rgb_for_auxiliary_target(tmp_path) -> None:
    config = ExperimentConfig()
    config.training.semantic.mixed_probability = 0
    base = _TinyDataset()
    _write_cache(tmp_path)
    dataset = EndpointDataset(
        base,  # type: ignore[arg-type]
        "semantic",
        0,
        config,
        _manifest(),
        tmp_path,
        load_semantic=True,
    )
    semantic, pixels, ratio, _label = dataset[0]
    assert semantic is not None and pixels.shape == (3, 128, 128)
    assert ratio.item() == 0
    assert base.loads == [True]


def test_pixel_phase_skips_dino_and_reuses_shared_epoch_state() -> None:
    config = ExperimentConfig()
    config.training.mode = "pixel_only"
    base = _TinyDataset()
    dataset = EndpointDataset(
        base,  # type: ignore[arg-type]
        "pixel",
        0,
        config,
        _manifest(),
        ".",
        load_semantic=False,
    )
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=1, rank=0, shuffle=True
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)
    assert dataset.semantic_cache is None
    assert dataset._phase_epoch.is_shared()
    semantic, _pixels, ratio, _label = dataset[0]
    assert semantic is None and ratio.item() == 1
    set_endpoint_loader_epoch(loader, "pixel", 7)
    assert dataset.phase == "pixel" and dataset.epoch == 7
    assert sampler.epoch == 7
    with pytest.raises(ValueError, match="semantic cache"):
        set_endpoint_loader_epoch(loader, "semantic", 0)
