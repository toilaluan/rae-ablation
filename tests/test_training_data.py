from __future__ import annotations

import numpy as np
import pytest
import torch

from jit_reg.token_cache import TokenCache, normalize_tokens
from jit_reg.training import TrainBatch, _PreparedBatchIterator, _TrainBatchCollator


def test_train_collator_loads_raw_tokens_without_cpu_normalization() -> None:
    raw = torch.tensor([[[[3.0]]], [[[7.0]]]], dtype=torch.bfloat16)

    class RawOnlyCache:
        def raw_lookup(self, indices: torch.Tensor) -> torch.Tensor:
            assert torch.equal(indices, torch.tensor([2, 5]))
            return raw

        def lookup(self, indices: torch.Tensor) -> torch.Tensor:
            del indices
            raise AssertionError("training collator must not normalize tokens on CPU")

    batch = _TrainBatchCollator(RawOnlyCache())(
        [(None, 11, 2), (None, 13, 5)]  # type: ignore[arg-type]
    )
    assert batch.tokens is raw
    assert torch.equal(batch.labels, torch.tensor([11, 13]))
    assert torch.equal(batch.indices, torch.tensor([2, 5]))


def test_prepared_batch_iterator_normalizes_raw_tokens_on_target_device() -> None:
    raw = torch.tensor(
        [
            [[[1.0]], [[6.0]]],
            [[[5.0]], [[2.0]]],
        ],
        dtype=torch.bfloat16,
    )
    source = iter([TrainBatch(raw, torch.tensor([3, 4]), torch.tensor([8, 9]))])
    mean = torch.tensor([1.0, 2.0])
    std = torch.tensor([2.0, 4.0])

    iterator = _PreparedBatchIterator(source, torch.device("cpu"), mean, std)
    batch = next(iterator)

    assert batch.tokens.dtype == torch.float32
    assert torch.equal(batch.tokens, normalize_tokens(raw, mean, std))
    assert batch.labels.dtype == torch.long
    assert batch.indices.dtype == torch.long
    with pytest.raises(StopIteration):
        next(iterator)


def test_raw_lookup_does_not_copy_advanced_index_result_twice() -> None:
    selected = np.arange(8, dtype=np.uint16).reshape(2, 1, 2, 2)

    class Values:
        def __getitem__(self, positions: np.ndarray) -> np.ndarray:
            assert np.array_equal(positions, np.array([0, 2]))
            return selected

    cache = object.__new__(TokenCache)
    cache.shape = (3, 1, 2, 2)
    cache.values = Values()  # type: ignore[assignment]

    tokens = cache.raw_lookup(torch.tensor([0, 2]))

    assert tokens.dtype == torch.bfloat16
    assert tokens.untyped_storage().data_ptr() == selected.ctypes.data


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_prepared_batch_uses_prefetch_stream() -> None:
    raw = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2).to(torch.bfloat16)
    raw = raw.pin_memory()
    source = iter([TrainBatch(raw, torch.tensor([1, 2]).pin_memory(), torch.tensor([3, 4]).pin_memory())])
    mean = torch.tensor([1.0])
    std = torch.tensor([2.0])

    iterator = _PreparedBatchIterator(source, torch.device("cuda"), mean, std)
    batch = next(iterator)
    torch.cuda.current_stream().synchronize()

    assert iterator._stream is not None
    assert batch.tokens.is_cuda
    assert torch.equal(batch.tokens.cpu(), normalize_tokens(raw, mean, std))
    iterator.close()
