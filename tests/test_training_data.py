from __future__ import annotations

import pytest
import torch

from jit_reg.config import IMAGE_SIZE
from jit_reg.training import TrainBatch, _collate, _PreparedBatchIterator


def test_train_collator_preserves_uint8_pixels_and_metadata() -> None:
    first = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8)
    second = torch.ones(3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8)
    batch = _collate([(first, 11, 2), (second, 13, 5)])
    assert batch.pixels.dtype == torch.uint8
    assert batch.pixels.shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.equal(batch.labels, torch.tensor([11, 13]))
    assert torch.equal(batch.indices, torch.tensor([2, 5]))


def test_prepared_batch_iterator_moves_only_online_batch() -> None:
    raw = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8)
    source = iter([TrainBatch(raw, torch.tensor([3, 4]), torch.tensor([8, 9]))])
    iterator = _PreparedBatchIterator(source, torch.device("cpu"))
    batch = next(iterator)
    assert batch.pixels.dtype == torch.uint8
    assert batch.labels.dtype == torch.long
    assert batch.indices.dtype == torch.long
    with pytest.raises(StopIteration):
        next(iterator)
