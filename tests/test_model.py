from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from jit_reg.config import CompileConfig, ModelConfig, REGConfig
from jit_reg.model.compile import clean_state_dict, compile_model_regions
from jit_reg.model.rope import apply_2d_rope
from jit_reg.model.sit import SiT, SiTOutput
from jit_reg.optim import _muon_eligible
from jit_reg.sampling import EulerSample, euler_sample


def tiny_config(
    patch_size: int = 8,
    *,
    checkpointing: bool = False,
    reg: bool = False,
    reg_variant: str = "standard",
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
            cls_predictor=reg_variant,
            cls_head_block=2,
            cls_head_hidden_size=32,
            cls_head_depth=2,
            cls_head_mlp_ratio=2,
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
    assert model.cls_predictor is None
    assert model.cls_clean_projection is None
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


def test_standard_reg_has_original_state_and_no_wide_allocations() -> None:
    model = SiT(tiny_config(reg=True))
    state = model.state_dict()
    assert model.cls_final_layer is not None
    assert model.cls_predictor is None
    assert model.cls_clean_projection is None
    assert any(key.startswith("cls_final_layer.") for key in state)
    assert not any(
        key.startswith("cls_predictor.") or key.startswith("cls_clean_projection.")
        for key in state
    )


def test_wide_reg_predicts_and_reinjects_after_repa() -> None:
    config = tiny_config(reg=True, reg_variant="wide_intermediate")
    config.reg.repa_block = config.reg.cls_head_block
    model = SiT(config)
    assert model.cls_final_layer is None
    assert model.cls_predictor is not None
    assert model.cls_clean_projection is not None
    assert model.cls_clean_projection is not model.cls_input
    assert len(model.cls_predictor.layers) == 2
    assert model.cls_predictor.token_projection.out_features == 32
    assert torch.count_nonzero(model.cls_predictor.output.projection.weight) == 0

    predicted_cls = torch.arange(1, 9, dtype=torch.float32)
    with torch.no_grad():
        model.cls_predictor.output.projection.bias.copy_(predicted_cls)
    seen: dict[str, torch.Tensor] = {}

    def capture_repa(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen["repa"] = inputs[0].detach().clone()

    def capture_next_block(_module: nn.Module, inputs: tuple[object, ...]) -> None:
        assert isinstance(inputs[0], torch.Tensor)
        seen["next"] = inputs[0].detach().clone()

    assert model.repa_projection is not None
    repa_hook = model.repa_projection.register_forward_pre_hook(capture_repa)
    next_hook = model.blocks[config.reg.cls_head_block].register_forward_pre_hook(
        capture_next_block
    )
    output = model(
        torch.randn(2, 3, 32, 32),
        torch.rand(2),
        torch.tensor([1, 2]),
        noisy_cls=torch.randn(2, 8),
        return_repa=True,
    )
    repa_hook.remove()
    next_hook.remove()

    assert output.cls is not None and output.cls.shape == (2, 8)
    torch.testing.assert_close(output.cls, predicted_cls.expand(2, -1))
    expected_clean = model.cls_clean_projection(output.cls)
    torch.testing.assert_close(seen["next"][:, 0], expected_clean)
    torch.testing.assert_close(seen["repa"][:, 1:], seen["next"][:, 1:])
    assert not torch.equal(seen["repa"][:, 0], seen["next"][:, 0])
    assert output.repa is not None and output.repa.shape == (2, 17, 8)


def test_wide_reg_image_and_cls_losses_backpropagate_through_clean_path() -> None:
    model = SiT(tiny_config(reg=True, reg_variant="wide_intermediate"))
    assert model.cls_predictor is not None and model.cls_clean_projection is not None
    with torch.no_grad():
        nn.init.normal_(model.cls_predictor.output.projection.weight)
        nn.init.normal_(model.final_layer.projection.weight)
        hidden = model.config.hidden_size
        model.blocks[model.config.reg.cls_head_block].modulation[-1].bias[
            2 * hidden : 3 * hidden
        ].fill_(1)
    output = model(
        torch.randn(2, 3, 32, 32),
        torch.rand(2),
        torch.tensor([1, 2]),
        noisy_cls=torch.randn(2, 8),
    )
    assert output.cls is not None
    output.image.square().mean().backward(retain_graph=True)
    assert model.cls_clean_projection.weight.grad is not None
    assert torch.count_nonzero(model.cls_clean_projection.weight.grad) > 0
    assert model.cls_predictor.token_projection.weight.grad is not None
    model.zero_grad(set_to_none=True)
    output.cls.square().mean().backward()
    assert model.cls_predictor.output.projection.weight.grad is not None
    assert torch.count_nonzero(model.cls_predictor.output.projection.weight.grad) > 0


def test_wide_swiglu_and_clean_projection_optimizer_policy() -> None:
    model = SiT(tiny_config(reg=True, reg_variant="wide_intermediate"))
    parameters = dict(model.named_parameters())
    ordinary_gate = "blocks.0.feed_forward.gate.weight"
    wide_gate = "cls_predictor.layers.0.feed_forward.gate.weight"
    ordinary_projection = "blocks.0.feed_forward.projection.weight"
    wide_projection = "cls_predictor.layers.0.feed_forward.projection.weight"
    assert _muon_eligible(ordinary_gate, parameters[ordinary_gate])
    assert _muon_eligible(wide_gate, parameters[wide_gate])
    assert not _muon_eligible(ordinary_projection, parameters[ordinary_projection])
    assert not _muon_eligible(wide_projection, parameters[wide_projection])
    clean_projection = "cls_clean_projection.weight"
    assert not _muon_eligible(clean_projection, parameters[clean_projection])


@pytest.mark.parametrize("reg_variant", ["disabled", "standard", "wide_intermediate"])
def test_sit_variants_smoke_euler_sampling(reg_variant: str) -> None:
    enabled = reg_variant != "disabled"
    model = SiT(
        tiny_config(
            reg=enabled,
            reg_variant=reg_variant if enabled else "standard",
        )
    ).eval()
    cls_noise = torch.randn(1, 8) if enabled else None
    sample = euler_sample(
        model,
        torch.randn(1, 3, 32, 32),
        torch.tensor([1]),
        cls_noise=cls_noise,
        steps=1,
    )
    if enabled:
        assert isinstance(sample, EulerSample)
        assert sample.image.shape == (1, 3, 32, 32)
        assert sample.cls.shape == (1, 8)
    else:
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (1, 3, 32, 32)


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


@pytest.mark.parametrize("reg_variant", [None, "wide_intermediate"])
def test_tiny_sit_regional_compile_has_clean_raw_keys(
    reg_variant: str | None,
) -> None:
    model = SiT(
        tiny_config(
            checkpointing=True,
            reg=reg_variant is not None,
            reg_variant=reg_variant or "standard",
        )
    )
    view = compile_model_regions(
        model, CompileConfig(True, "eager", "default", False, False)
    )
    output = view(
        torch.randn(1, 3, 32, 32),
        torch.rand(1),
        torch.tensor([1]),
        noisy_cls=torch.randn(1, 8) if reg_variant is not None else None,
    )
    (output.image.sum() + (output.cls.sum() if output.cls is not None else 0)).backward()
    state = clean_state_dict(model)
    assert state and not any("_orig_mod" in key for key in state)
