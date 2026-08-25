from __future__ import annotations

import copy

import torch

from cascade_dit.checkpoint import EMA
from cascade_dit.config import CurriculumConfig, DetailHeadsConfig, ModelConfig
from cascade_dit.curriculum import configure_trainability, transition_to_stage
from cascade_dit.model import LightningDiT, LoRALinear, has_lora
from cascade_dit.optim import parameter_count


def _model() -> LightningDiT:
    return LightningDiT(
        ModelConfig(hidden_size=32, depth=4, num_heads=4, mlp_ratio=2, num_classes=10),
        DetailHeadsConfig(pixel_enabled=False),
    )


def _permutation() -> torch.Tensor:
    return torch.eye(768).roll(17, dims=0)


def _make_nonzero(model: LightningDiT) -> None:
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.01)
    torch.nn.init.normal_(model.token_detail.output_projection.weight, std=0.01)
    for block in model.blocks:
        torch.nn.init.normal_(block.adaln[-1].weight, std=0.01)


def test_lightningdit_shape_zero_initialization_and_official_primitives() -> None:
    model = _model()
    output = model(
        torch.randn(2, 64, 768), torch.tensor([0.2, 0.8]), torch.tensor([1, 2])
    )
    assert output.prediction.shape == (2, 64, 768)
    assert torch.count_nonzero(output.prediction) == 0
    assert isinstance(model.blocks[0].norm1, torch.nn.RMSNorm)
    assert model.rope_cos.shape == (64, 8)
    assert hasattr(model.blocks[0].mlp, "w3")


def test_lora_is_zero_function_then_merges_exactly() -> None:
    model = _model().eval()
    _make_nonzero(model)
    inputs = torch.randn(1, 64, 768)
    arguments = (inputs, torch.tensor([0.4]), torch.tensor([2]))
    before = model(*arguments).prediction
    model.attach_lora([1, 2], rank=4, alpha=4, dropout=0)
    assert has_lora(model)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            assert torch.count_nonzero(module.lora_b) == 0
            torch.nn.init.normal_(module.lora_b, std=0.01)
    adapted = model(*arguments).prediction
    model.merge_lora()
    merged = model(*arguments).prediction
    assert not has_lora(model)
    assert not torch.equal(before, adapted)
    assert torch.allclose(adapted, merged, atol=2e-6, rtol=2e-5)


def test_projector_folding_preserves_rgb_function() -> None:
    model = _model().eval()
    _make_nonzero(model)
    projector = _permutation()
    model.set_stage("rgb", projector)
    values = torch.randn(1, 64, 768)
    before = model(values, torch.tensor([0.3]), torch.tensor([4])).prediction
    model.fold_projector()
    after = model(values, torch.tensor([0.3]), torch.tensor([4])).prediction
    assert torch.allclose(before, after, atol=3e-6, rtol=2e-5)
    assert torch.equal(model.projector.matrix, torch.eye(768))


def test_transition_preserves_aligned_model_and_ema_functions() -> None:
    model = _model().eval()
    _make_nonzero(model)
    ema = EMA(model)
    old_ema = ema.to_model(model).eval()
    alignment = _permutation()
    projectors = {
        "dino12": torch.eye(768),
        "dino8": alignment,
        "dino4": alignment,
        "rgb": alignment,
    }
    curriculum = CurriculumConfig(
        fully_train_blocks=[0, 3], lora_blocks=[1, 2], lora_rank=4, lora_alpha=4
    )
    canonical = torch.randn(1, 64, 768)
    timestep, labels = torch.tensor([0.6]), torch.tensor([3])
    old = model(canonical, timestep, labels).prediction
    old_shadow = old_ema(canonical, timestep, labels).prediction
    assert transition_to_stage(model, ema, 1, projectors, curriculum)
    assert not transition_to_stage(model, ema, 1, projectors, curriculum)
    aligned = canonical @ alignment
    new = model.eval()(aligned, timestep, labels).prediction
    new_shadow = ema.to_model(model).eval()(aligned, timestep, labels).prediction
    assert torch.allclose(new, old @ alignment, atol=3e-6, rtol=2e-5)
    assert torch.allclose(new_shadow, old_shadow @ alignment, atol=3e-6, rtol=2e-5)


def test_trainability_policy_and_exported_baseline_counts_match() -> None:
    model = _model()
    curriculum = CurriculumConfig(
        fully_train_blocks=[0, 3], lora_blocks=[1, 2], lora_rank=4, lora_alpha=4
    )
    model.attach_lora([1, 2], rank=4, alpha=4)
    names = configure_trainability(model, 1, curriculum)
    assert "input_projection.weight" in names
    assert "class_embedding.embedding.weight" not in names
    assert "blocks.0.attn.qkv.weight" in names
    assert "blocks.1.attn.qkv.base.weight" not in names
    assert "blocks.1.attn.qkv.lora_a" in names
    model.merge_lora()
    progressive_count = parameter_count(model)
    baseline = _model()
    configure_trainability(baseline, 3, curriculum, pixel_baseline=True)
    assert progressive_count == parameter_count(baseline)
    assert all(parameter.requires_grad for parameter in baseline.parameters())


def test_copying_model_does_not_share_projector() -> None:
    model = _model()
    duplicate = copy.deepcopy(model)
    duplicate.projector.matrix.zero_()
    assert torch.count_nonzero(model.projector.matrix) == 768
