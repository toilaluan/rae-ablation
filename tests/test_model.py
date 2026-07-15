from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from rae_ablation.config import CompileConfig, ModelConfig
from rae_ablation.model.compile import clean_state_dict, compile_model_regions
from rae_ablation.model.ddt import DDTOutput, RAEv2DDTIG
from rae_ablation.model.factory import build_diffusion_model
from rae_ablation.model.rope import apply_2d_rope


def tiny_config(*, checkpointing: bool = False) -> ModelConfig:
    return ModelConfig(
        in_channels=4,
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
        num_classes=10,
        dropout=0.0,
        gradient_checkpointing=checkpointing,
    )


def test_rope_preserves_norm_and_changes_positions() -> None:
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn_like(q)
    qr, kr = apply_2d_rope(q, k, 2, 3)
    assert torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), kr.norm(dim=-1), atol=1e-5)
    assert not torch.equal(q[..., -1, :], qr[..., -1, :])


def test_model_factory_builds_ddt() -> None:
    assert isinstance(build_diffusion_model(tiny_config()), RAEv2DDTIG)


@pytest.mark.parametrize("grid", [4, 2])
def test_ddt_forward_backward_both_grids(grid: int) -> None:
    model = RAEv2DDTIG(tiny_config(checkpointing=True))
    latent = torch.randn(2, 4, grid, grid, requires_grad=True)
    output = model(latent, torch.rand(2), torch.tensor([1, 2]))
    assert isinstance(output, DDTOutput)
    assert output.full.shape == latent.shape
    assert output.early.shape == latent.shape
    assert torch.count_nonzero(output.full) == 0
    assert torch.count_nonzero(output.early) == 0
    (output.full.sum() + output.early.sum()).backward()
    assert model.full_head.projection.weight.grad is not None
    assert model.early_head.projection.weight.grad is not None


def test_ddt_condition_token_shapes_and_base_depth_isolation() -> None:
    model = RAEv2DDTIG(tiny_config())
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
    plain = RAEv2DDTIG(tiny_config(checkpointing=False))
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
    model = RAEv2DDTIG(tiny_config(checkpointing=True))
    view = compile_model_regions(model, CompileConfig(True, "eager", "default", False, False))
    output = view(torch.randn(1, 4, 2, 2), torch.rand(1), torch.tensor([1]))
    (output.full.sum() + output.early.sum()).backward()
    state = clean_state_dict(model)
    assert state
    assert not any("_orig_mod" in key for key in state)
