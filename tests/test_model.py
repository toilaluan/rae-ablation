from __future__ import annotations

import torch

from self_flow.config import ModelConfig
from self_flow.model import SiT
from self_flow.model.sit import FusedSwiGLU


def tiny_model(*, layers: list[int] | None = None) -> SiT:
    config = ModelConfig(
        image_size=32,
        patch_size=1,
        hidden_size=32,
        depth=3,
        num_heads=4,
        mlp_ratio=4,
        num_classes=10,
        variant="SiT-tiny/1",
    )
    return SiT(config, align_layers=layers, projector_hidden_size=48)


def test_sit_shapes_and_adaln_zero_initialization() -> None:
    model = tiny_model(layers=[1, 2])
    x = torch.randn(2, 128, 2, 2)
    output = model(
        x,
        torch.rand(2, 4),
        torch.tensor([1, 2]),
        capture_layers=(1, 2),
        project_features=True,
    )
    assert output.x0 is not None and output.x0.shape == x.shape
    assert all(feature.shape == (2, 4, 32) for feature in output.features)
    assert torch.count_nonzero(output.x0) == 0
    assert isinstance(model.blocks[0].mlp, FusedSwiGLU)
    assert model.blocks[0].mlp.intermediate_size == 85
    assert model.blocks[0].attention.rope_cos.shape == (4, 8)
    assert not model.blocks[0].attention_norm.elementwise_affine
    assert not hasattr(model, "position_embedding")


def test_homogeneous_and_token_timesteps_match_when_broadcast() -> None:
    model = tiny_model().eval()
    x = torch.randn(2, 128, 2, 2)
    labels = torch.tensor([1, 2])
    timestep = torch.tensor([0.2, 0.7])
    scalar = model(x, timestep, labels).x0
    tokenwise = model(x, timestep[:, None].expand(-1, 4), labels).x0
    assert torch.equal(scalar, tokenwise)


def test_truncated_forward_skips_later_blocks_and_keeps_gradients() -> None:
    model = tiny_model(layers=[1, 2])
    calls = [0, 0, 0]
    handles = [
        block.register_forward_hook(
            lambda _m, _a, _o, index=index: calls.__setitem__(index, calls[index] + 1)
        )
        for index, block in enumerate(model.blocks)
    ]
    output = model(
        torch.randn(2, 128, 2, 2),
        torch.rand(2),
        torch.tensor([1, 2]),
        capture_layers=(1, 2),
        stop_at_layer=2,
    )
    sum(feature.sum() for feature in output.features).backward()
    for handle in handles:
        handle.remove()
    assert output.x0 is None
    assert calls == [1, 1, 0]
    assert model.patch_embed.weight.grad is not None
