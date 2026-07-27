from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from jit_reg.data import ImageNetDataset
from jit_reg.reg import FeatureTargets
from jit_reg.training import (
    TrainBatch,
    _CUDABatchPrefetcher,
    prepare_train_batch,
)


def test_imagenet_workers_emit_chw_uint8_and_python_integers() -> None:
    pixels = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(4, 5, 3)
    dataset = ImageNetDataset(
        [{"image": Image.fromarray(pixels), "label": np.int64(7)}],
        image_size=4,
    )
    image, label, index = dataset[0]
    assert image.shape == (3, 4, 4)
    assert image.dtype == torch.uint8
    assert isinstance(label, int) and label == 7
    assert isinstance(index, int) and index == 0


@pytest.mark.parametrize(
    ("reg", "repa"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_cpu_batch_preparation_matches_previous_pixel_and_cache_values(
    reg: bool,
    repa: bool,
) -> None:
    pixels = torch.arange(2 * 3 * 4 * 4, dtype=torch.uint8).reshape(
        2, 3, 4, 4
    )
    cached = (
        FeatureTargets(
            torch.randn(2, 4, 8, dtype=torch.bfloat16) if reg else None,
            torch.randn(2, 4, 8, dtype=torch.bfloat16) if repa else None,
        )
        if reg or repa
        else None
    )
    source = TrainBatch(
        pixels,
        torch.tensor([1, 2]),
        torch.tensor([3, 4]),
        torch.randn(2, 8, 2, 2, dtype=torch.bfloat16),
        cached,
    )
    prepared = prepare_train_batch(source, "cpu")
    assert prepared.pixels.dtype == torch.float32
    assert torch.equal(prepared.pixels, pixels.float().div(255))
    assert torch.equal(prepared.labels, source.labels)
    assert prepared.indices is source.indices
    assert prepared.cached_latents is not None
    assert prepared.cached_latents.dtype == torch.float32
    if cached is None:
        assert prepared.cached_targets is None
    else:
        assert prepared.cached_targets is not None
        for actual, expected in zip(
            prepared.cached_targets, cached, strict=True
        ):
            if expected is None:
                assert actual is None
            else:
                assert actual is not None
                assert actual.dtype == torch.float32
                assert torch.equal(actual, expected.float())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_prefetch_matches_synchronous_preparation() -> None:
    source = TrainBatch(
        torch.randint(0, 256, (2, 3, 4, 4), dtype=torch.uint8).pin_memory(),
        torch.tensor([1, 2]).pin_memory(),
        torch.tensor([3, 4]).pin_memory(),
        torch.randn(2, 8, 2, 2, dtype=torch.bfloat16).pin_memory(),
        FeatureTargets(
            torch.randn(2, 4, 8, dtype=torch.bfloat16).pin_memory(),
            torch.randn(2, 4, 8, dtype=torch.bfloat16).pin_memory(),
        ),
    )
    expected = prepare_train_batch(source, "cpu")
    iterator = _CUDABatchPrefetcher(iter([source]), torch.device("cuda"))
    actual = next(iterator)
    torch.cuda.current_stream().synchronize()
    assert torch.allclose(
        actual.pixels.cpu(), expected.pixels, atol=1e-7, rtol=0
    )
    assert torch.equal(actual.labels.cpu(), expected.labels)
    assert actual.cached_latents is not None
    assert expected.cached_latents is not None
    assert torch.equal(
        actual.cached_latents.cpu(), expected.cached_latents
    )
    assert actual.cached_targets is not None
    assert expected.cached_targets is not None
    assert torch.equal(
        actual.cached_targets.reg.cpu(), expected.cached_targets.reg
    )
    assert torch.equal(
        actual.cached_targets.repa.cpu(), expected.cached_targets.repa
    )
    iterator.close()
