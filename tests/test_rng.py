from __future__ import annotations

import torch

from jit_reg.rng import RNGStreams, indexed_standard_normal


def test_noise_stream_round_trips_for_exact_resume() -> None:
    streams = RNGStreams.create(7, "cpu")
    torch.randn((2, 3), generator=streams.noise)
    state = streams.state_dict()
    expected = torch.randn((2, 3), generator=streams.noise)

    restored = RNGStreams.create(99, "cpu")
    restored.load_state_dict(state)
    assert torch.equal(torch.randn((2, 3), generator=restored.noise), expected)


def test_indexed_noise_is_invariant_to_batching_and_order() -> None:
    indices = torch.tensor([9, 2, 17])
    together = indexed_standard_normal(indices, (2, 2), 123, "cpu")
    reordered = indexed_standard_normal(indices.flip(0), (2, 2), 123, "cpu")
    split = torch.cat(
        (
            indexed_standard_normal(indices[:1], (2, 2), 123, "cpu"),
            indexed_standard_normal(indices[1:], (2, 2), 123, "cpu"),
        )
    )
    assert torch.equal(together, split)
    assert torch.equal(together, reordered.flip(0))
