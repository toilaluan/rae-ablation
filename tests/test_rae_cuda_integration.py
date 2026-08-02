from __future__ import annotations

import os

import pytest
import torch

from jit_reg.config import ExperimentConfig
from jit_reg.rae import load_rae_codec
from jit_reg.token_cache import denormalize_tokens, normalize_tokens


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_RAE_CUDA_INTEGRATION") != "1" or not torch.cuda.is_available(),
    reason="set RUN_RAE_CUDA_INTEGRATION=1 on a CUDA host to load the pinned RAE pair",
)
def test_pinned_rae_encode_normalize_denormalize_decode_smoke() -> None:
    codec = load_rae_codec(ExperimentConfig().rae, "cuda")
    pixels = torch.rand(1, 3, 256, 256, device="cuda")
    tokens = codec.encode(pixels)
    assert tokens.shape == (1, 1024, 8, 8)
    mean = tokens.float().mean(dim=(0, 2, 3))
    std = tokens.float().std(dim=(0, 2, 3), correction=0).clamp_min(1e-6)
    restored = denormalize_tokens(normalize_tokens(tokens, mean, std), mean, std)
    assert torch.allclose(restored, tokens.float(), atol=1e-3, rtol=1e-3)
    decoded = codec.decode(restored)
    assert decoded.shape == (1, 3, 256, 256)
    assert torch.all(torch.isfinite(decoded))
