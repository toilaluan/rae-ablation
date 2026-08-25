from __future__ import annotations

import copy

import pytest
import torch

from cascade_dit.config import DDTAttentionConfig, DDTPathConfig, ModelConfig
from cascade_dit.curriculum import configure_trainability
from cascade_dit.model import DDT, DDTDecoderBlock, DDTEncoderBlock
from cascade_dit.optim import parameter_count


def _model(levels: int = 4) -> DDT:
    config = ModelConfig(
        encoder=DDTPathConfig(width=32, depth=2, heads=4),
        decoder=DDTPathConfig(width=64, depth=1, heads=4),
        attention=DDTAttentionConfig(qk_norm=True, rope=True, sdpa=True),
    )
    config.conditioning.num_classes = 10
    config.conditioning.timestep_tokens = 2
    config.conditioning.class_tokens = 3
    return DDT(config, levels)


def _arguments(levels: int = 4) -> tuple[torch.Tensor, ...]:
    values = torch.randn(2, 64, 768)
    timestep = torch.tensor([0.2, 0.8])
    labels = torch.tensor([1, 2])
    weights = torch.zeros(2, levels)
    weights[0, 0] = 1
    weights[1, -1] = 1
    return values, timestep, labels, weights


def test_ddt_shape_zero_initialization_and_encoder_decoder_topology() -> None:
    model = _model()
    output = model(*_arguments())
    assert output.shape == (2, 64, 768)
    assert torch.count_nonzero(output) == 0
    assert len(model.encoder_blocks) == 2
    assert len(model.decoder_blocks) == 1
    assert isinstance(model.encoder_blocks[0], DDTEncoderBlock)
    assert isinstance(model.decoder_blocks[0], DDTDecoderBlock)
    assert model.encoder_blocks[0].attention.rope_cosine.shape == (70, 8)
    assert model.decoder_blocks[0].attention.rope_cosine.shape == (64, 16)


def test_R_is_validated_and_drives_conditioning_gradients() -> None:
    model = _model()
    values, timestep, labels, weights = _arguments()
    with torch.no_grad():
        torch.nn.init.normal_(model.final_layer.projection.weight, std=0.01)
        torch.nn.init.normal_(model.final_layer.adaln[-1].weight, std=0.01)
    output = model(values, timestep, labels, weights)
    output.square().mean().backward()
    gradient = model.representation_embedder.embedding.grad
    assert gradient is not None and torch.count_nonzero(gradient) > 0

    invalid = weights.clone()
    invalid[0, 0] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        model(values, timestep, labels, invalid)
    with pytest.raises(ValueError, match="shape"):
        model(values, timestep, labels, weights[:, :-1])


def test_progressive_and_baseline_are_identical_full_train_architectures() -> None:
    progressive = _model(levels=3)
    baseline = _model(levels=3)
    names = configure_trainability(progressive)
    configure_trainability(baseline)
    assert len(names) == len(list(progressive.named_parameters()))
    assert all(parameter.requires_grad for parameter in progressive.parameters())
    assert parameter_count(progressive) == parameter_count(baseline)
    duplicate = copy.deepcopy(progressive)
    with torch.no_grad():
        next(duplicate.parameters()).add_(1)
    assert not torch.equal(next(progressive.parameters()), next(duplicate.parameters()))
