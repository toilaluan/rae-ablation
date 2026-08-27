from __future__ import annotations

import pytest
import torch
from torch import nn

from cascade_dit.config import (
    DDTAttentionConfig,
    DDTPathConfig,
    ModelConfig,
    PixelDecoderConfig,
    SemanticDecoderConfig,
)
from cascade_dit.model import DDT, DDTOutput, build_model
from cascade_dit.model.ddt import DiPPatchDecoder, PixelTransformerBlock


class FastDetailer(nn.Module):
    def forward(self, coarse: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return coarse + context[:, :3, None, None] * 0.01


def _model(*, base_depth: int = 2) -> DDT:
    config = ModelConfig(
        base_prediction_depth=base_depth,
        encoder=DDTPathConfig(width=16, depth=3, heads=1),
        semantic_decoder=SemanticDecoderConfig(width=32, depth=2, heads=2),
        pixel_decoder=PixelDecoderConfig(
            width=32,
            depth=3,
            heads=2,
            encoder_features=[1, 2, 3],
            decoder_features=[1, 2],
        ),
        attention=DDTAttentionConfig(qk_norm=True, rope=False, sdpa=True),
    )
    config.conditioning.num_classes = 10
    config.conditioning.timestep_tokens = 2
    config.conditioning.class_tokens = 3
    model = build_model(config)
    model.patch_detailer = FastDetailer()
    return model


def _arguments() -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(1, 64, 768),
        torch.tensor([0.2]),
        torch.tensor([1]),
    )


def _has_gradient(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in module.parameters()
    )


def test_joint_topology_shapes_and_zero_initialized_semantic_heads() -> None:
    model = _model()
    assert len(model.encoder_blocks) == 3
    assert len(model.semantic_decoder_blocks) == 2
    assert len(model.pixel_blocks) == 3
    assert all(isinstance(block, PixelTransformerBlock) for block in model.pixel_blocks)
    assert all(not hasattr(block, "adaln") for block in model.pixel_blocks)
    semantic = model(*_arguments(), decode_pixels=False)
    assert isinstance(semantic, DDTOutput)
    assert semantic.full_x0.shape == semantic.base_x0.shape == (1, 64, 768)
    assert semantic.rgb is None
    assert torch.count_nonzero(semantic.full_x0) == 0
    assert torch.count_nonzero(semantic.base_x0) == 0
    assert torch.count_nonzero(model.full_head.projection.weight) == 0
    assert torch.count_nonzero(model.base_head.projection.weight) == 0

    joint = model(*_arguments(), decode_pixels=True)
    assert joint.rgb is not None and joint.rgb.shape == (1, 3, 128, 128)
    assert joint.pixels is joint.rgb


def test_decode_pixels_false_skips_every_pixel_module(monkeypatch) -> None:
    model = _model()

    def fail(*_args, **_kwargs):
        raise AssertionError("pixel path executed")

    monkeypatch.setattr(model, "_decode_pixels", fail)
    assert model(*_arguments(), decode_pixels=False).rgb is None


def test_isolated_base_loss_stops_after_selected_encoder_block() -> None:
    model = _model(base_depth=2)
    with torch.no_grad():
        nn.init.normal_(model.base_head.projection.weight, std=0.02)
    output = model(*_arguments(), decode_pixels=False)
    output.base_x0.square().mean().backward()
    assert _has_gradient(model.source_projection)
    assert _has_gradient(model.encoder_blocks[0])
    assert _has_gradient(model.encoder_blocks[1])
    assert not _has_gradient(model.encoder_blocks[2])
    assert _has_gradient(model.base_head)
    assert not _has_gradient(model.semantic_decoder_blocks)
    assert not _has_gradient(model.full_head)
    assert not _has_gradient(model.pixel_blocks)


def test_pixel_loss_reaches_flow_trunk_without_detachment() -> None:
    model = _model()
    output = model(*_arguments(), decode_pixels=True)
    assert output.rgb is not None
    output.rgb.square().mean().backward()
    assert _has_gradient(model.source_projection)
    assert _has_gradient(model.encoder_blocks)
    assert _has_gradient(model.pixel_blocks)
    assert _has_gradient(model.coarse_patch_projection)


def test_dip_patch_decoder_is_patch_isolated() -> None:
    detailer = DiPPatchDecoder(context_width=4).eval()
    patches = torch.randn(2, 3, 16, 16)
    context = torch.randn(2, 4)
    together = detailer(patches, context)
    first = detailer(patches[:1], context[:1])
    second = detailer(patches[1:], context[1:])
    assert together.shape == (2, 3, 16, 16)
    assert torch.allclose(together, torch.cat((first, second)), atol=1e-6)


def test_feature_selections_are_validated() -> None:
    config = _model().config
    config.pixel_decoder.encoder_features = [1, 1]
    with pytest.raises(ValueError, match="unique"):
        build_model(config)
