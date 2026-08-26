from __future__ import annotations

import pytest
import torch
from torch import nn

from cascade_dit.config import (
    DDTAttentionConfig,
    DDTDecoderConfig,
    DDTPathConfig,
    ModelConfig,
)
from cascade_dit.model import DDT, DDTOutput, build_model
from cascade_dit.model.ddt import DDTDecoderBlock, DDTEncoderBlock


def _model() -> DDT:
    config = ModelConfig(
        encoder=DDTPathConfig(width=16, depth=1, heads=1),
        decoder=DDTDecoderConfig(
            width=32, depth=3, semantic_depth=2, heads=2
        ),
        attention=DDTAttentionConfig(qk_norm=True, rope=False, sdpa=True),
    )
    config.conditioning.num_classes = 10
    config.conditioning.timestep_tokens = 2
    config.conditioning.class_tokens = 3
    return build_model(config)


def _arguments() -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(2, 64, 768),
        torch.tensor([0.2, 0.8]),
        torch.tensor([1, 2]),
        torch.tensor([0.0, 0.75]),
    )


def _activate_conditioning_and_heads(model: DDT) -> None:
    with torch.no_grad():
        for block in (*model.shared_decoder_blocks, *model.pixel_decoder_blocks):
            nn.init.normal_(block.adaln[-1].weight, std=0.02)
        for head in (model.semantic_head, model.pixel_head):
            nn.init.normal_(head.adaln[-1].weight, std=0.02)
            nn.init.normal_(head.projection.weight, std=0.02)


def _has_gradient(module: nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in module.parameters()
    )


def test_split_topology_shapes_and_zero_initialized_heads() -> None:
    model = _model()
    assert len(model.shared_decoder_blocks) == 2
    assert len(model.pixel_decoder_blocks) == 1
    assert isinstance(model.encoder_blocks[0], DDTEncoderBlock)
    assert isinstance(model.shared_decoder_blocks[0], DDTDecoderBlock)
    for mode in ("semantic", "mixed", "pixel"):
        output = model(*_arguments(), prediction_mode=mode)
        assert isinstance(output, DDTOutput)
        assert output.prediction.shape == (2, 64, 768)
        assert torch.count_nonzero(output.prediction) == 0
    assert torch.count_nonzero(model.semantic_head.projection.weight) == 0
    assert torch.count_nonzero(model.pixel_head.projection.weight) == 0


def test_mixed_prediction_is_exact_ratio_interpolation() -> None:
    model = _model()
    with torch.no_grad():
        model.semantic_head.projection.bias.fill_(2)
        model.pixel_head.projection.bias.fill_(6)
    output = model(*_arguments(), prediction_mode="mixed")
    assert output.semantic_x0 is not None and output.pixel_x0 is not None
    expected = torch.lerp(
        output.semantic_x0,
        output.pixel_x0,
        _arguments()[-1][:, None, None],
    )
    assert torch.equal(output.prediction, expected)
    assert torch.all(output.prediction[0] == 2)
    assert torch.all(output.prediction[1] == 5)


def test_forward_modes_skip_inactive_heads(monkeypatch) -> None:
    model = _model()

    def fail(*_args, **_kwargs):
        raise AssertionError("inactive branch executed")

    monkeypatch.setattr(model.pixel_head, "forward", fail)
    semantic = model(*_arguments(), prediction_mode="semantic")
    assert semantic.pixel_x0 is None

    model = _model()
    monkeypatch.setattr(model.semantic_head, "forward", fail)
    pixel = model(*_arguments(), prediction_mode="pixel")
    assert pixel.semantic_x0 is None


def test_mixed_pixel_branch_is_detached_from_encoder_and_shared_decoder() -> None:
    model = _model()
    _activate_conditioning_and_heads(model)
    output = model(*_arguments(), prediction_mode="mixed")
    assert output.pixel_x0 is not None
    output.pixel_x0.float().square().mean().backward()
    assert not _has_gradient(model.source_projection)
    assert not _has_gradient(model.target_projection)
    assert not _has_gradient(model.encoder_blocks)
    assert not _has_gradient(model.shared_decoder_blocks)
    assert _has_gradient(model.pixel_decoder_blocks)
    assert _has_gradient(model.pixel_head)


def test_pixel_mode_trains_full_trunk_and_never_updates_semantic_head() -> None:
    model = _model()
    _activate_conditioning_and_heads(model)
    output = model(*_arguments(), prediction_mode="pixel")
    output.prediction.float().square().mean().backward()
    assert _has_gradient(model.source_projection)
    assert _has_gradient(model.target_projection)
    assert _has_gradient(model.encoder_blocks)
    assert _has_gradient(model.shared_decoder_blocks)
    assert _has_gradient(model.pixel_decoder_blocks)
    assert _has_gradient(model.pixel_head)
    assert not _has_gradient(model.semantic_head)


def test_pixel_ratio_shape_and_bounds_are_validated() -> None:
    model = _model()
    values, timestep, labels, ratio = _arguments()
    with pytest.raises(ValueError, match="shape"):
        model(values, timestep, labels, ratio[:, None], prediction_mode="mixed")
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        model(values, timestep, labels, ratio + 1, prediction_mode="mixed")
