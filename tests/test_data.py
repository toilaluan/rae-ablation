from __future__ import annotations

from collections import Counter

import pytest
import torch

from cascade_dit.config import (
    ExperimentConfig,
    RepresentationSourceConfig,
    RepresentationStageConfig,
)
from cascade_dit.data import (
    ADJACENT_MIX,
    ALL_LEVELS_MIX,
    PURE_CURRENT,
    PURE_NEXT,
    MixedRepresentationDataset,
    RepresentationBatchPreprocessor,
    prepare_pixels,
    representation_collator,
    sample_representation_weights,
    set_stage_loader_epoch,
)
from cascade_dit.representations import RepresentationManifest, patchify_rgb


def test_default_mixture_probabilities_and_determinism() -> None:
    config = ExperimentConfig()
    choices = [sample_representation_weights(config, 0, 3, index) for index in range(20_000)]
    counts = Counter(choice.mode for choice in choices)
    assert counts[PURE_CURRENT] / len(choices) == pytest.approx(0.85, abs=0.015)
    assert counts[ADJACENT_MIX] / len(choices) == pytest.approx(0.10, abs=0.01)
    assert counts[PURE_NEXT] / len(choices) == pytest.approx(0.025, abs=0.006)
    assert counts[ALL_LEVELS_MIX] / len(choices) == pytest.approx(0.025, abs=0.006)
    repeated = sample_representation_weights(config, 0, 3, 17)
    assert repeated.mode == choices[17].mode
    assert torch.equal(repeated.weights, choices[17].weights)


def test_adjacent_and_uniform_simplex_weights_are_well_formed() -> None:
    config = ExperimentConfig()
    adjacent = all_levels = None
    for index in range(10_000):
        choice = sample_representation_weights(config, 1, 0, index)
        if choice.mode == ADJACENT_MIX and adjacent is None:
            adjacent = choice.weights
        if choice.mode == ALL_LEVELS_MIX and all_levels is None:
            all_levels = choice.weights
        if adjacent is not None and all_levels is not None:
            break
    assert adjacent is not None and all_levels is not None
    assert torch.count_nonzero(adjacent) == 2
    assert torch.count_nonzero(adjacent[[0, 3]]) == 0
    assert torch.all(all_levels > 0)
    assert torch.isclose(adjacent.sum(), torch.tensor(1.0))
    assert torch.isclose(all_levels.sum(), torch.tensor(1.0))


def test_dynamic_level_count_and_final_stage_are_supported() -> None:
    config = ExperimentConfig()
    config.representations.stages = [
        RepresentationStageConfig("deep", 1, RepresentationSourceConfig("dino", 12)),
        RepresentationStageConfig("shallow", 1, RepresentationSourceConfig("dino", 3)),
    ]
    config.pixel_training.epochs = 1
    config.validate()
    all_levels = None
    for index in range(20_000):
        choice = sample_representation_weights(config, 0, 1, index)
        if choice.mode == ALL_LEVELS_MIX:
            all_levels = choice
            break
    assert all_levels is not None and all_levels.weights.shape == (3,)
    final = sample_representation_weights(config, 2, 1, 0)
    baseline = sample_representation_weights(config, 0, 1, 0, pixel_only=True)
    assert final.mode == baseline.mode == PURE_CURRENT
    assert torch.equal(final.weights, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(baseline.weights, final.weights)


def test_stage_loader_epoch_updates_shared_dataset_and_sampler_state() -> None:
    config = ExperimentConfig()
    levels = config.num_representation_levels

    class TinyDataset:
        def __len__(self) -> int:
            return 1

        def get_item(self, index: int, *, load_pixels: bool):
            pixels = torch.zeros(3, 128, 128, dtype=torch.uint8) if load_pixels else None
            return pixels, 0, index

    manifest = RepresentationManifest(
        dataset_size=1,
        dataset_fingerprint="dataset",
        encoder_fingerprint="encoder",
        preparation_fingerprint="preparation",
        stages=[],
        statistics={},
        cache_files={},
        completion_files={},
        complete=True,
    )
    dataset = MixedRepresentationDataset(
        TinyDataset(),  # type: ignore[arg-type]
        levels - 1,
        0,
        config,
        manifest,
        ".",
        pixel_only=True,
    )
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=1, rank=0, shuffle=True
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler)

    assert dataset._stage_epoch.is_shared()
    set_stage_loader_epoch(loader, levels - 1, 7)

    assert dataset.stage_index == levels - 1
    assert dataset.epoch == 7
    assert sampler.epoch == 7
    with pytest.raises(ValueError, match="non-negative"):
        set_stage_loader_epoch(loader, levels - 1, -1)


def test_sparse_raw_batches_are_preprocessed_like_the_cpu_reference() -> None:
    config = ExperimentConfig()
    config.representations.stages = [
        RepresentationStageConfig("deep", 1, RepresentationSourceConfig("dino", 12)),
        RepresentationStageConfig("shallow", 1, RepresentationSourceConfig("dino", 6)),
    ]
    channel = torch.arange(768, dtype=torch.float32)
    statistics = {
        stage.name: {
            "mean": (channel / 100 + level).tolist(),
            "std": (channel / 200 + level + 1).tolist(),
            "count": 10,
        }
        for level, stage in enumerate(config.stages)
    }
    manifest = RepresentationManifest(
        dataset_size=3,
        dataset_fingerprint="dataset",
        encoder_fingerprint="encoder",
        preparation_fingerprint="preparation",
        stages=[],
        statistics=statistics,
        cache_files={},
        completion_files={},
        complete=True,
    )
    generator = torch.Generator().manual_seed(17)
    deep_0 = torch.randn(64, 768, generator=generator).to(torch.bfloat16)
    deep_2 = torch.randn(64, 768, generator=generator).to(torch.bfloat16)
    shallow_1 = torch.randn(64, 768, generator=generator).to(torch.bfloat16)
    shallow_2 = torch.randn(64, 768, generator=generator).to(torch.bfloat16)
    rgb_0 = torch.randint(0, 256, (3, 128, 128), dtype=torch.uint8, generator=generator)
    samples = [
        ((deep_0, None, rgb_0), torch.tensor([0.75, 0.0, 0.25]), 4, 10, ADJACENT_MIX),
        ((None, shallow_1, None), torch.tensor([0.0, 1.0, 0.0]), 5, 11, PURE_CURRENT),
        ((deep_2, shallow_2, None), torch.tensor([0.2, 0.8, 0.0]), 6, 12, ADJACENT_MIX),
    ]

    raw_batch = representation_collator(samples)
    assert raw_batch.levels[0] is not None
    assert raw_batch.levels[1] is not None
    assert raw_batch.levels[2] is not None
    assert torch.equal(raw_batch.levels[0].batch_indices, torch.tensor([0, 2]))
    assert torch.equal(raw_batch.levels[1].batch_indices, torch.tensor([1, 2]))
    assert torch.equal(raw_batch.levels[2].batch_indices, torch.tensor([0]))
    assert raw_batch.levels[0].values.dtype == torch.bfloat16
    assert raw_batch.levels[2].values.dtype == torch.uint8

    prepared = RepresentationBatchPreprocessor(config, manifest, "cpu")(raw_batch)
    expected = torch.zeros_like(prepared.tokens)
    for batch_index, (levels, weights, _label, _index, _mode) in enumerate(samples):
        for level_index, raw in enumerate(levels):
            if raw is None:
                continue
            values = (
                patchify_rgb(prepare_pixels(raw[None], "cpu"))[0]
                if config.stages[level_index].source.kind == "rgb"
                else raw.float()
            )
            mean = torch.tensor(statistics[config.stages[level_index].name]["mean"])
            std = torch.tensor(statistics[config.stages[level_index].name]["std"])
            expected[batch_index].add_(
                (values - mean) / std,
                alpha=float(weights[level_index]),
            )

    assert torch.allclose(prepared.tokens, expected, atol=1e-6, rtol=1e-6)
    assert torch.equal(prepared.representation_weights, raw_batch.representation_weights)
    assert torch.equal(prepared.labels, torch.tensor([4, 5, 6]))
    assert torch.equal(prepared.indices, torch.tensor([10, 11, 12]))
    assert torch.equal(prepared.modes, torch.tensor([ADJACENT_MIX, PURE_CURRENT, ADJACENT_MIX]))


def test_pixel_only_preprocessing_needs_only_rgb_statistics() -> None:
    config = ExperimentConfig()
    config.pixel_training.use_representation_guiding = False
    statistics = {
        "rgb": {
            "mean": torch.zeros(768).tolist(),
            "std": torch.ones(768).tolist(),
            "count": 10,
        }
    }
    manifest = RepresentationManifest(
        dataset_size=1,
        dataset_fingerprint="dataset",
        encoder_fingerprint="ignored",
        preparation_fingerprint="ignored",
        stages=[{"name": "rgb", "source": {"kind": "rgb", "layer": None}}],
        statistics=statistics,
        cache_files={},
        completion_files={},
        complete=True,
    )
    pixels = torch.randint(0, 256, (3, 128, 128), dtype=torch.uint8)
    weights = torch.tensor([0.0, 0.0, 1.0])
    batch = representation_collator(
        [((None, None, pixels), weights, 3, 7, PURE_CURRENT)]
    )

    prepared = RepresentationBatchPreprocessor(
        config,
        manifest,
        "cpu",
        pixel_only=True,
        representation_levels=3,
    )(batch)

    assert torch.equal(prepared.representation_weights, weights[None])
    assert torch.equal(prepared.tokens, patchify_rgb(prepare_pixels(pixels[None], "cpu")))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_raw_batch_preprocessing_runs_on_cuda() -> None:
    config = ExperimentConfig()
    statistics = {
        stage.name: {
            "mean": torch.zeros(768).tolist(),
            "std": torch.ones(768).tolist(),
            "count": 10,
        }
        for stage in config.stages
    }
    manifest = RepresentationManifest(
        dataset_size=1,
        dataset_fingerprint="dataset",
        encoder_fingerprint="encoder",
        preparation_fingerprint="preparation",
        stages=[],
        statistics=statistics,
        cache_files={},
        completion_files={},
        complete=True,
    )
    values = torch.randn(64, 768).to(torch.bfloat16)
    weights = torch.tensor([1.0, 0.0, 0.0, 0.0])
    raw_batch = representation_collator([((values, None, None, None), weights, 3, 7, PURE_CURRENT)])

    prepared = RepresentationBatchPreprocessor(config, manifest, "cuda")(raw_batch)

    assert prepared.tokens.device.type == "cuda"
    assert prepared.labels.device.type == "cuda"
    assert torch.equal(prepared.tokens.cpu()[0], values.float())
