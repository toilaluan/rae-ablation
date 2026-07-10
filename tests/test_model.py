from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from rae_ablation.config import (
    CompileConfig,
    DDTIGConfig,
    DepthAttentionConfig,
    ModelConfig,
)
from rae_ablation.model.compile import clean_state_dict, compile_model_regions
from rae_ablation.model.ddt import RAEv2DDTIG
from rae_ablation.model.depth_attention import DepthAttention, depth_source_layers
from rae_ablation.model.dit import DiffusionTransformer, SwiGLU, TransformerBlock
from rae_ablation.model.factory import build_diffusion_model
from rae_ablation.model.rope import apply_2d_rope


def tiny_config(*, checkpointing: bool = False, depth_enabled: bool = True) -> ModelConfig:
    return ModelConfig(
        in_channels=4,
        hidden_size=16,
        depth=4,
        num_heads=4,
        mlp_size=32,
        base_grid_size=4,
        early_exit_block=2,
        gradient_checkpointing=checkpointing,
        depth_attention=DepthAttentionConfig(depth_enabled, 2),
    )


def tiny_ddt_config(*, checkpointing: bool = False) -> ModelConfig:
    return ModelConfig(
        architecture="raev2_ddt_ig",
        in_channels=4,
        patch_size=1,
        num_classes=10,
        dropout=0.0,
        gradient_checkpointing=checkpointing,
        ddt_ig=DDTIGConfig(
            encoder_hidden_size=16,
            decoder_hidden_size=32,
            encoder_depth=3,
            decoder_depth=2,
            encoder_num_heads=4,
            decoder_num_heads=8,
            mlp_ratio=2.0,
            base_model_depth=2,
            num_time_tokens=2,
            num_class_tokens=3,
        ),
    )


@pytest.mark.parametrize("grid", [4, 2])
def test_forward_backward_both_grids(grid: int) -> None:
    model = DiffusionTransformer(tiny_config(checkpointing=True))
    latent = torch.randn(2, 4, grid, grid, requires_grad=True)
    output = model(latent, torch.rand(2), torch.tensor([1, 2]))
    assert output.full.shape == latent.shape
    assert output.early.shape == latent.shape
    (output.full.sum() + output.early.sum()).backward()
    assert model.full_head.projection.weight.grad is not None


def test_position_interpolation_and_swiglu_dimensions() -> None:
    model = DiffusionTransformer(tiny_config())
    assert model.interpolated_position_embedding(2, 2).shape == (1, 4, 16)
    mlp = SwiGLU(16, 37)
    assert mlp.up.out_features == 74
    assert mlp.down.in_features == 37


def test_rope_preserves_norm_and_changes_positions() -> None:
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn_like(q)
    qr, kr = apply_2d_rope(q, k, 2, 3)
    assert torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), kr.norm(dim=-1), atol=1e-5)
    assert not torch.equal(q[..., -1, :], qr[..., -1, :])


def test_adaln_zero_block_is_identity() -> None:
    config = tiny_config()
    block = TransformerBlock(config, 0)
    x = torch.randn(2, 16, 16)
    conditioning = torch.randn(2, 16)
    bank = torch.zeros(2, 2, 4, 16, 4)
    mask = torch.zeros(2, dtype=torch.bool)
    output, *_ = block(x, conditioning, 4, 4, bank, bank.clone(), mask)
    assert torch.equal(output, x)


def test_depth_sources_fixed_bank_recursion_and_gradient() -> None:
    assert depth_source_layers(0) == (0,)
    assert depth_source_layers(3) == (0, 3)
    assert depth_source_layers(8) == (0, 4, 8)
    bank = torch.zeros(1, 3, 1, 1, 4)
    mask = torch.zeros(3, dtype=torch.bool)
    zero = torch.zeros(1, 1, 1, 4)
    value0 = torch.ones(1, 1, 1, 4, requires_grad=True)
    mixed0, bank_k, bank_v, mask = DepthAttention(0, 9, 4)(
        zero, zero, value0, bank, bank.clone(), mask
    )
    value4 = 3 * torch.ones_like(value0)
    mixed4, bank_k, bank_v, mask = DepthAttention(4, 9, 4)(zero, zero, value4, bank_k, bank_v, mask)
    assert torch.allclose(mixed0, value0)
    assert torch.allclose(mixed4, torch.full_like(mixed4, 2))
    assert mask.tolist() == [True, True, False]
    mixed4.sum().backward()
    assert torch.allclose(value0.grad, torch.full_like(value0, 0.5))


def test_depth_disabled_is_raw_value() -> None:
    value = torch.randn(1, 2, 3, 4)
    bank = torch.randn(1, 2, 2, 3, 4)
    mask = torch.ones(2, dtype=torch.bool)
    mixed, *_ = DepthAttention(1, 4, 2, enabled=False)(value, value, value, bank, bank, mask)
    assert torch.equal(mixed, value)


def test_checkpointed_matches_plain_outputs_and_gradients() -> None:
    plain = DiffusionTransformer(tiny_config(checkpointing=False))
    checked = copy.deepcopy(plain)
    checked.set_gradient_checkpointing(True)
    latent1 = torch.randn(2, 4, 4, 4, requires_grad=True)
    latent2 = latent1.detach().clone().requires_grad_(True)
    timestep = torch.rand(2)
    labels = torch.tensor([3, 5])
    out1, out2 = plain(latent1, timestep, labels), checked(latent2, timestep, labels)
    assert torch.equal(out1.full, out2.full)
    assert torch.equal(out1.early, out2.early)
    (out1.full.sum() + out1.early.sum()).backward()
    (out2.full.sum() + out2.early.sum()).backward()
    for first, second in zip(plain.parameters(), checked.parameters(), strict=True):
        if first.grad is not None:
            assert torch.equal(first.grad, second.grad)


def test_tiny_real_regional_compile_has_clean_raw_keys() -> None:
    model = DiffusionTransformer(tiny_config(checkpointing=True))
    view = compile_model_regions(model, CompileConfig(True, "eager", "default", False, False))
    output = view(torch.randn(1, 4, 2, 2), torch.rand(1), torch.tensor([1]))
    (output.full.sum() + output.early.sum()).backward()
    state = clean_state_dict(model)
    assert state
    assert not any("_orig_mod" in key for key in state)


def test_model_factory_preserves_legacy_default_and_selects_ddt() -> None:
    assert isinstance(build_diffusion_model(tiny_config()), DiffusionTransformer)
    assert isinstance(build_diffusion_model(tiny_ddt_config()), RAEv2DDTIG)


@pytest.mark.parametrize("grid", [4, 2])
def test_ddt_forward_backward_both_grids(grid: int) -> None:
    model = RAEv2DDTIG(tiny_ddt_config(checkpointing=True))
    latent = torch.randn(2, 4, grid, grid, requires_grad=True)
    output = model(latent, torch.rand(2), torch.tensor([1, 2]))
    assert output.full.shape == latent.shape
    assert output.early.shape == latent.shape
    assert torch.count_nonzero(output.full) == 0
    assert torch.count_nonzero(output.early) == 0
    (output.full.sum() + output.early.sum()).backward()
    assert model.full_head.projection.weight.grad is not None
    assert model.early_head.projection.weight.grad is not None


def test_ddt_condition_token_shapes_and_base_depth_isolation() -> None:
    model = RAEv2DDTIG(tiny_ddt_config())
    timestep_base, timestep_tokens = model.time_embedding(torch.rand(2))
    label_tokens = model.class_embedding(torch.tensor([1, 2]))
    assert timestep_base.shape == (2, 1, 16)
    assert timestep_tokens.shape == (2, 2, 16)
    assert label_tokens.shape == (2, 3, 16)

    nn.init.normal_(model.early_head.projection.weight)
    latent = torch.randn(2, 4, 2, 2)
    timestep = torch.rand(2)
    labels = torch.tensor([1, 2])
    before = model(latent, timestep, labels).early
    with torch.no_grad():
        for parameter in model.encoder_blocks[-1].parameters():
            parameter.add_(torch.randn_like(parameter))
    after = model(latent, timestep, labels).early
    assert torch.equal(before, after)


def test_checkpointed_ddt_matches_plain_outputs_and_gradients() -> None:
    plain = RAEv2DDTIG(tiny_ddt_config(checkpointing=False))
    checked = copy.deepcopy(plain)
    checked.set_gradient_checkpointing(True)
    nn.init.normal_(plain.full_head.projection.weight)
    nn.init.normal_(plain.early_head.projection.weight)
    checked.load_state_dict(plain.state_dict())
    latent1 = torch.randn(2, 4, 2, 2, requires_grad=True)
    latent2 = latent1.detach().clone().requires_grad_(True)
    timestep = torch.rand(2)
    labels = torch.tensor([3, 5])
    out1 = plain(latent1, timestep, labels)
    out2 = checked(latent2, timestep, labels)
    assert torch.equal(out1.full, out2.full)
    assert torch.equal(out1.early, out2.early)
    (out1.full.sum() + out1.early.sum()).backward()
    (out2.full.sum() + out2.early.sum()).backward()
    assert torch.equal(latent1.grad, latent2.grad)
    for first, second in zip(plain.parameters(), checked.parameters(), strict=True):
        if first.grad is not None:
            assert torch.equal(first.grad, second.grad)


def test_tiny_ddt_regional_compile_has_clean_raw_keys() -> None:
    model = RAEv2DDTIG(tiny_ddt_config(checkpointing=True))
    view = compile_model_regions(model, CompileConfig(True, "eager", "default", False, False))
    output = view(torch.randn(1, 4, 2, 2), torch.rand(1), torch.tensor([1]))
    (output.full.sum() + output.early.sum()).backward()
    state = clean_state_dict(model)
    assert state
    assert not any("_orig_mod" in key for key in state)
