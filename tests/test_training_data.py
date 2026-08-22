from __future__ import annotations

import pytest
import torch

from cascade_dit.config import IMAGE_SIZE
from cascade_dit.training import (
    RepresentationCacheCollator,
    TrainBatch,
    _PreparedBatchIterator,
    extract_representations,
    training_requires_pixels,
)


class FakeCache:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.zeros(indices.numel(), *self.shape, dtype=torch.bfloat16)


def test_collator_supports_all_cache_combinations() -> None:
    pixels = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8)
    samples = [(pixels, 3, 7), (pixels, 4, 8)]
    collator = RepresentationCacheCollator(
        FakeCache((128, 16, 16)), FakeCache((768, 8, 8))
    )
    batch = collator(samples)
    assert batch.pixels is not None
    assert batch.image_latents.shape == (2, 128, 16, 16)
    assert batch.semantic_tokens.shape == (2, 768, 8, 8)
    cached_only = collator([(None, 3, 7), (None, 4, 8)])
    assert cached_only.pixels is None


def test_pixel_loading_policy_for_baseline_and_cascades() -> None:
    image_cache = FakeCache((128, 16, 16))
    semantic_cache = FakeCache((768, 8, 8))
    assert training_requires_pixels("latent", None, None)
    assert not training_requires_pixels("latent", image_cache, None)  # type: ignore[arg-type]
    assert training_requires_pixels("shared_cascade", image_cache, None)  # type: ignore[arg-type]
    assert not training_requires_pixels(  # type: ignore[arg-type]
        "decoupled_cascade", image_cache, semantic_cache
    )
    assert training_requires_pixels(  # type: ignore[arg-type]
        "decoupled_cascade", image_cache, semantic_cache, "pixels"
    )


def test_direct_pixel_target_is_normalized_without_mage() -> None:
    pixels = torch.tensor([0, 255], dtype=torch.uint8).reshape(2, 1, 1, 1)
    pixels = pixels.expand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    semantic = torch.zeros(2, 768, 8, 8)
    batch = TrainBatch(
        pixels,
        None,
        semantic,
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
    )
    target, restored_semantic = extract_representations(
        batch,
        "shared_cascade",
        None,
        None,
        "pixels",
    )
    assert target.dtype == torch.float32
    assert target[0].unique().item() == -1
    assert target[1].unique().item() == 1
    assert restored_semantic is semantic


def test_prepared_batch_iterator_preserves_cached_tensors() -> None:
    source = iter(
        [
            TrainBatch(
                None,
                torch.zeros(2, 128, 16, 16, dtype=torch.bfloat16),
                torch.zeros(2, 768, 8, 8, dtype=torch.bfloat16),
                torch.tensor([3, 4]),
                torch.tensor([8, 9]),
            )
        ]
    )
    iterator = _PreparedBatchIterator(source, torch.device("cpu"))
    batch = next(iterator)
    assert batch.image_latents.dtype == torch.bfloat16
    assert batch.semantic_tokens.dtype == torch.bfloat16
    with pytest.raises(StopIteration):
        next(iterator)
