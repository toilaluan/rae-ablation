from __future__ import annotations

import copy

import pytest
import torch

from jit_reg.config import CompileConfig, ModelConfig, REGConfig
from jit_reg.model.compile import clean_state_dict, compile_model_regions
from jit_reg.model.rope import apply_2d_rope
from jit_reg.model.sit import SiT, SiTOutput


def tiny_config(
    patch_size: int = 8,
    *,
    checkpointing: bool = False,
    reg: bool = False,
) -> ModelConfig:
    return ModelConfig(
        image_size=32,
        patch_size=patch_size,
        hidden_size=16,
        depth=4,
        num_heads=4,
        mlp_ratio=2,
        num_classes=10,
        gradient_checkpointing=checkpointing,
        variant="SiT-Tiny",
        reg=REGConfig(
            enabled=reg,
            feature_dim=8,
            repa_block=4,
            repa_hidden_size=16,
        ),
    )


def test_rope_preserves_norm_and_changes_positions() -> None:
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn_like(q)
    qr, kr = apply_2d_rope(q, k, 2, 3)
    torch.testing.assert_close(q.norm(dim=-1), qr.norm(dim=-1))
    torch.testing.assert_close(k.norm(dim=-1), kr.norm(dim=-1))
    assert not torch.equal(q[..., -1, :], qr[..., -1, :])


@pytest.mark.parametrize(
    ("image_size", "patch_size", "tokens"),
    [(256, 16, 256), (256, 32, 64), (256, 64, 16)],
)
def test_configurable_patch_counts_and_reconstructed_shapes(
    image_size: int, patch_size: int, tokens: int
) -> None:
    config = tiny_config()
    config.image_size = image_size
    config.patch_size = patch_size
    config.depth = 1
    model = SiT(config)
    assert model.num_patches == tokens
    image = torch.randn(1, 3, image_size, image_size, requires_grad=True)
    output = model(image, torch.rand(1), torch.tensor([1]))
    assert isinstance(output, SiTOutput)
    assert output.image.shape == image.shape
    assert torch.count_nonzero(output.image) == 0
    output.image.sum().backward()
    assert model.final_layer.projection.weight.grad is not None


def test_baseline_has_no_reg_allocations_or_state() -> None:
    model = SiT(tiny_config())
    assert model.cls_input is None
    assert model.cls_final_layer is None
    assert model.repa_projection is None
    assert not any("cls_input" in key or "repa_projection" in key for key in model.state_dict())


def test_reg_prepends_one_unrotated_token_and_projects_block_four() -> None:
    model = SiT(tiny_config(reg=True))
    output = model(
        torch.randn(2, 3, 32, 32),
        torch.rand(2),
        torch.tensor([1, 2]),
        noisy_cls=torch.randn(2, 8),
        return_repa=True,
    )
    assert output.image.shape == (2, 3, 32, 32)
    assert output.cls is not None and output.cls.shape == (2, 8)
    assert output.repa is not None and output.repa.shape == (2, 17, 8)
    assert torch.count_nonzero(output.image) == 0
    assert torch.count_nonzero(output.cls) == 0


def test_checkpointing_matches_plain_outputs_and_gradients() -> None:
    plain = SiT(tiny_config())
    checked = copy.deepcopy(plain)
    checked.set_gradient_checkpointing(True)
    torch.nn.init.normal_(plain.final_layer.projection.weight)
    checked.load_state_dict(plain.state_dict())
    image1 = torch.randn(2, 3, 32, 32, requires_grad=True)
    image2 = image1.detach().clone().requires_grad_(True)
    timestep = torch.rand(2)
    labels = torch.tensor([3, 5])
    out1 = plain(image1, timestep, labels).image
    out2 = checked(image2, timestep, labels).image
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)
    out1.sum().backward()
    out2.sum().backward()
    torch.testing.assert_close(image1.grad, image2.grad, rtol=0, atol=0)
    for first, second in zip(plain.parameters(), checked.parameters(), strict=True):
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)


def test_tiny_sit_regional_compile_has_clean_raw_keys() -> None:
    model = SiT(tiny_config(checkpointing=True))
    view = compile_model_regions(
        model, CompileConfig(True, "eager", "default", False, False)
    )
    output = view(
        torch.randn(1, 3, 32, 32), torch.rand(1), torch.tensor([1])
    )
    output.image.sum().backward()
    state = clean_state_dict(model)
    assert state and not any("_orig_mod" in key for key in state)
