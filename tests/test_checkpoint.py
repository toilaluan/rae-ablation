from __future__ import annotations

import torch
from torch import nn

from jit_reg.checkpoint import EMA


def test_ema_foreach_updates_floating_state_and_copies_integer_buffers() -> None:
    model = nn.BatchNorm1d(3)
    ema = EMA(model, decay=0.75)
    original = {key: value.clone() for key, value in ema.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(4)
        model.bias.add_(4)
        model.running_mean.add_(4)
        model.running_var.add_(4)
        model.num_batches_tracked.add_(1)
    ema.update(model)
    current = model.state_dict()
    for key, value in ema.state_dict().items():
        if value.is_floating_point():
            assert torch.allclose(value, original[key].lerp(current[key], 0.25))
        else:
            assert torch.equal(value, current[key])
