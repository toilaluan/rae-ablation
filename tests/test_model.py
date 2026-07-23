from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
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
    token_mode: str = "cls",
    pool_size: int = 2,
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
            token_mode=token_mode,
            pool_size=pool_size,
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
    assert model.reg_position_embedding is None
    assert not any("cls_input" in key or "repa_projection" in key for key in model.state_dict())


def test_cls_reg_prepends_one_unrotated_token_and_projects_image_tokens() -> None:
    model = SiT(tiny_config(reg=True))
    output = model(
        torch.randn(2, 3, 32, 32),
        torch.rand(2),
        torch.tensor([1, 2]),
        noisy_reg=torch.randn(2, 1, 8),
        return_repa=True,
    )
    assert output.image.shape == (2, 3, 32, 32)
    assert output.reg is not None and output.reg.shape == (2, 1, 8)
    assert output.repa is not None and output.repa.shape == (2, 16, 8)
    assert torch.count_nonzero(output.image) == 0
    assert torch.count_nonzero(output.reg) == 0


def test_reg_without_repa_has_no_repa_state_or_output() -> None:
    config = tiny_config(reg=True)
    config.reg.repa_enabled = False
    config.reg.repa_block = 99
    model = SiT(config)
    assert model.repa_projection is None
    assert not any(key.startswith("repa_projection.") for key in model.state_dict())
    output = model(
        torch.randn(2, 3, 32, 32),
        torch.rand(2),
        torch.tensor([1, 2]),
        noisy_reg=torch.randn(2, 1, 8),
        return_repa=True,
    )
    assert output.repa is None


def test_reg_uses_one_shared_time_conditioning() -> None:
    model = SiT(tiny_config(reg=True))
    original_keys = tuple(model.state_dict())
    seen: list[torch.Tensor] = []

    def capture_modulation(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen.append(inputs[0].detach().clone())

    hook = model.blocks[0].modulation.register_forward_pre_hook(capture_modulation)
    model(
        torch.randn(2, 3, 32, 32),
        torch.tensor([0.1, 0.2]),
        torch.tensor([1, 2]),
        noisy_reg=torch.randn(2, 1, 8),
    )
    hook.remove()
    assert len(seen) == 1
    assert tuple(model.state_dict()) == original_keys


def test_spatial_reg_adds_learned_positions_and_projects_only_image_tokens() -> None:
    model = SiT(tiny_config(reg=True, token_mode="cls_spatial", pool_size=2))
    assert model.num_reg_tokens == 5
    assert model.reg_position_embedding is not None
    assert model.reg_position_embedding.shape == (1, 4, 16)
    noisy_reg = torch.randn(2, 5, 8)
    seen: dict[str, torch.Tensor] = {}

    def capture_first_block(_module: nn.Module, inputs: tuple[object, ...]) -> None:
        assert isinstance(inputs[0], torch.Tensor)
        seen["sequence"] = inputs[0].detach().clone()

    hook = model.blocks[0].register_forward_pre_hook(capture_first_block)
    output = model(
        torch.randn(2, 3, 32, 32),
        torch.rand(2),
        torch.tensor([1, 2]),
        noisy_reg=noisy_reg,
        return_repa=True,
    )
    hook.remove()
    assert model.cls_input is not None
    projected = model.cls_input(noisy_reg)
    torch.testing.assert_close(seen["sequence"][:, 0], projected[:, 0])
    torch.testing.assert_close(
        seen["sequence"][:, 1:5],
        projected[:, 1:] + model.reg_position_embedding,
    )
    assert output.reg is not None and output.reg.shape == (2, 5, 8)
    assert output.repa is not None and output.repa.shape == (2, 16, 8)


def test_standard_reg_has_original_state_and_no_wide_allocations() -> None:
    model = SiT(tiny_config(reg=True))
    state = model.state_dict()
    assert model.cls_final_layer is not None
    assert model.cls_predictor is None
    assert model.cls_clean_projection is None
    assert any(key.startswith("cls_final_layer.") for key in state)
    assert not any(
        key.startswith("cls_predictor.") or key.startswith("cls_clean_projection.") for key in state
    )
    assert "reg_position_embedding" not in state
    restored = SiT(tiny_config(reg=True))
    restored.load_state_dict(state)
    assert restored.state_dict().keys() == state.keys()


@pytest.mark.parametrize("token_mode", ["cls", "cls_spatial"])
def test_wide_reg_predicts_and_reinjects_after_repa(token_mode: str) -> None:
    config = tiny_config(reg=True, reg_variant="wide_intermediate", token_mode=token_mode)
    config.reg.repa_block = config.reg.cls_head_block
    model = SiT(config)
    assert model.cls_final_layer is None
    assert model.cls_predictor is not None
    assert model.cls_clean_projection is not None
    assert model.cls_clean_projection is not model.cls_input
    assert len(model.cls_predictor.layers) == 2
    assert model.cls_predictor.token_projection.out_features == 32
    assert torch.count_nonzero(model.cls_predictor.output.projection.weight) == 0

    predicted_reg = torch.arange(1, 9, dtype=torch.float32)
    with torch.no_grad():
        model.cls_predictor.output.projection.bias.copy_(predicted_reg)
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
        noisy_reg=torch.randn(2, model.num_reg_tokens, 8),
        return_repa=True,
    )
    repa_hook.remove()
    next_hook.remove()

    assert output.reg is not None
    assert output.reg.shape == (2, model.num_reg_tokens, 8)
    torch.testing.assert_close(
        output.reg,
        predicted_reg.view(1, 1, -1).expand(2, model.num_reg_tokens, -1),
    )
    expected_clean = model._add_reg_positions(model.cls_clean_projection(output.reg))
    torch.testing.assert_close(seen["next"][:, : model.num_reg_tokens], expected_clean)
    torch.testing.assert_close(seen["repa"], seen["next"][:, model.num_reg_tokens :])
    assert output.repa is not None and output.repa.shape == (2, 16, 8)


def test_wide_reg_image_and_reg_losses_backpropagate_through_clean_path() -> None:
    model = SiT(
        tiny_config(
            reg=True,
            reg_variant="wide_intermediate",
            token_mode="cls_spatial",
        )
    )
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
        noisy_reg=torch.randn(2, model.num_reg_tokens, 8),
    )
    assert output.reg is not None
    output.image.square().mean().backward(retain_graph=True)
    assert model.cls_clean_projection.weight.grad is not None
    assert torch.count_nonzero(model.cls_clean_projection.weight.grad) > 0
    assert model.cls_predictor.token_projection.weight.grad is not None
    model.zero_grad(set_to_none=True)
    output.reg.square().mean().backward()
    assert model.cls_predictor.output.projection.weight.grad is not None
    assert torch.count_nonzero(model.cls_predictor.output.projection.weight.grad) > 0


def test_wide_reg_x0_conditioning_and_configurable_continuation_tokens() -> None:
    config = tiny_config(
        reg=True,
        reg_variant="wide_intermediate",
        token_mode="cls_spatial",
    )
    config.reg.x0_conditioning = True
    config.reg.continuation_tokens = 3
    model = SiT(config)
    assert model.cls_clean_projection is None
    assert model.reg_x0_conditioning_projection is not None
    assert model.reg_continuation_projection is not None
    seen: dict[str, torch.Tensor] = {}

    def capture_latent(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen["latent"] = inputs[0].detach().clone()

    def capture_next(_module: nn.Module, inputs: tuple[object, ...]) -> None:
        assert isinstance(inputs[0], torch.Tensor)
        assert isinstance(inputs[1], torch.Tensor)
        seen["next_sequence"] = inputs[0].detach().clone()
        seen["next_conditioning"] = inputs[1].detach().clone()

    def capture_final(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        seen["final_conditioning"] = inputs[1].detach().clone()

    latent_hook = model.reg_continuation_projection.register_forward_pre_hook(capture_latent)
    next_hook = model.blocks[config.reg.cls_head_block].register_forward_pre_hook(capture_next)
    final_hook = model.final_layer.register_forward_pre_hook(capture_final)
    image = torch.randn(2, 3, 32, 32)
    timestep = torch.tensor([0.2, 0.7])
    labels = torch.tensor([1, 2])
    output = model(
        image,
        timestep,
        labels,
        noisy_reg=torch.randn(2, model.num_reg_tokens, 8),
    )
    latent_hook.remove()
    next_hook.remove()
    final_hook.remove()

    continuation = model.reg_continuation_projection(seen["latent"]).reshape(2, 3, 16)
    torch.testing.assert_close(seen["next_sequence"][:, :3], continuation)
    assert seen["next_sequence"].shape[1] == 3 + model.num_patches
    assert output.reg is not None and output.reg.shape == (2, model.num_reg_tokens, 8)
    base = model.timestep_embedding(timestep) + model.class_embedding(labels)
    expected_conditioning = base + model.reg_x0_conditioning_projection(output.reg[:, 0])
    torch.testing.assert_close(seen["next_conditioning"], expected_conditioning)
    torch.testing.assert_close(seen["final_conditioning"], expected_conditioning)


def test_wide_extension_paths_receive_image_gradients() -> None:
    config = tiny_config(reg=True, reg_variant="wide_intermediate")
    config.reg.x0_conditioning = True
    config.reg.continuation_tokens = 2
    model = SiT(config)
    assert model.reg_x0_conditioning_projection is not None
    assert model.reg_continuation_projection is not None
    assert model.cls_predictor is not None
    with torch.no_grad():
        nn.init.normal_(model.final_layer.projection.weight)
        nn.init.normal_(model.final_layer.modulation[-1].weight)
        hidden = config.hidden_size
        model.blocks[config.reg.cls_head_block].modulation[-1].bias[
            2 * hidden : 3 * hidden
        ].fill_(1)
    output = model(
        torch.randn(2, 3, 32, 32),
        torch.rand(2),
        torch.tensor([1, 2]),
        noisy_reg=torch.randn(2, 1, 8),
    )
    output.image.square().mean().backward()
    assert model.reg_x0_conditioning_projection.weight.grad is not None
    assert model.reg_continuation_projection.weight.grad is not None
    assert model.cls_predictor.output.projection.weight.grad is not None


@pytest.mark.parametrize("source", ["x0_cls", "wide_cls"])
def test_patch_reprojection_recovers_raw_xt_patches_at_wide_head(source: str) -> None:
    config = tiny_config(reg=True, reg_variant="wide_intermediate")
    config.reg.patch_reprojection.enabled = True
    config.reg.patch_reprojection.source = source
    config.reg.patch_reprojection.intermediate_size = 32
    model = SiT(config)
    assert model.reg_patch_reprojection is not None
    assert model.reg_patch_reprojection.value_projection.in_features == 8 * 8 * 3
    assert model.reg_patch_reprojection.value_projection.out_features == 32
    assert model.reg_patch_reprojection.output_projection.out_features == 16
    expected_source_size = 8 if source == "x0_cls" else 32
    assert model.reg_patch_reprojection.source_projection.in_features == expected_source_size
    seen: dict[str, torch.Tensor] = {}

    def capture_head(
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: torch.Tensor,
    ) -> None:
        seen["head_sequence"] = output.detach().clone()

    def capture_reprojection_input(
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        seen["raw_patches"] = inputs[0].detach().clone()
        seen["source"] = inputs[1].detach().clone()
        seen["time"] = inputs[2].detach().clone()

    def capture_reprojection_output(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        seen["delta"] = output.detach().clone()

    def capture_next(_module: nn.Module, inputs: tuple[object, ...]) -> None:
        assert isinstance(inputs[0], torch.Tensor)
        seen["next_sequence"] = inputs[0].detach().clone()

    head_hook = model.blocks[config.reg.cls_head_block - 1].register_forward_hook(capture_head)
    input_hook = model.reg_patch_reprojection.register_forward_pre_hook(
        capture_reprojection_input
    )
    output_hook = model.reg_patch_reprojection.register_forward_hook(
        capture_reprojection_output
    )
    next_hook = model.blocks[config.reg.cls_head_block].register_forward_pre_hook(capture_next)
    image = torch.randn(2, 3, 32, 32)
    timestep = torch.tensor([0.25, 0.75])
    output = model(
        image,
        timestep,
        torch.tensor([1, 2]),
        noisy_reg=torch.randn(2, 1, 8),
    )
    head_hook.remove()
    input_hook.remove()
    output_hook.remove()
    next_hook.remove()

    expected_raw = F.unfold(image, kernel_size=8, stride=8).transpose(1, 2)
    torch.testing.assert_close(seen["raw_patches"], expected_raw)
    torch.testing.assert_close(seen["time"], model.timestep_embedding(timestep))
    if source == "x0_cls":
        assert output.reg is not None
        torch.testing.assert_close(seen["source"], output.reg[:, 0])
    else:
        assert seen["source"].shape == (2, 32)
    expected_image_tokens = seen["head_sequence"][:, 1:] + seen["delta"]
    torch.testing.assert_close(seen["next_sequence"][:, 1:], expected_image_tokens)


def test_patch_reprojection_receives_image_loss_gradients() -> None:
    config = tiny_config(reg=True, reg_variant="wide_intermediate")
    config.reg.patch_reprojection.enabled = True
    config.reg.patch_reprojection.intermediate_size = 32
    model = SiT(config)
    assert model.reg_patch_reprojection is not None
    assert model.cls_predictor is not None
    with torch.no_grad():
        nn.init.normal_(model.final_layer.projection.weight)
        hidden = config.hidden_size
        model.blocks[config.reg.cls_head_block].modulation[-1].bias[
            2 * hidden : 3 * hidden
        ].fill_(1)
    output = model(
        torch.randn(2, 3, 32, 32),
        torch.rand(2),
        torch.tensor([1, 2]),
        noisy_reg=torch.randn(2, 1, 8),
    )
    output.image.square().mean().backward()
    assert model.reg_patch_reprojection.value_projection.weight.grad is not None
    assert model.reg_patch_reprojection.source_projection.weight.grad is not None
    assert model.reg_patch_reprojection.output_projection.weight.grad is not None
    assert model.cls_predictor.output.projection.weight.grad is not None


def test_wide_swiglu_and_clean_projection_optimizer_policy() -> None:
    model = SiT(
        tiny_config(
            reg=True,
            reg_variant="wide_intermediate",
            token_mode="cls_spatial",
        )
    )
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
    assert not _muon_eligible("reg_position_embedding", parameters["reg_position_embedding"])


@pytest.mark.parametrize(
    ("reg_variant", "token_mode"),
    [
        ("disabled", "cls"),
        ("standard", "cls"),
        ("standard", "cls_spatial"),
        ("wide_intermediate", "cls_spatial"),
    ],
)
def test_sit_variants_smoke_euler_sampling(reg_variant: str, token_mode: str) -> None:
    enabled = reg_variant != "disabled"
    model = SiT(
        tiny_config(
            reg=enabled,
            reg_variant=reg_variant if enabled else "standard",
            token_mode=token_mode,
        )
    ).eval()
    reg_noise = torch.randn(1, model.num_reg_tokens, 8) if enabled else None
    sample = euler_sample(
        model,
        torch.randn(1, 3, 32, 32),
        torch.tensor([1]),
        reg_noise=reg_noise,
        steps=1,
    )
    if enabled:
        assert isinstance(sample, EulerSample)
        assert sample.image.shape == (1, 3, 32, 32)
        assert sample.reg.shape == (1, model.num_reg_tokens, 8)
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
            token_mode="cls_spatial" if reg_variant is not None else "cls",
        )
    )
    view = compile_model_regions(model, CompileConfig(True, "eager", "default", False, False))
    output = view(
        torch.randn(1, 3, 32, 32),
        torch.rand(1),
        torch.tensor([1]),
        noisy_reg=torch.randn(1, model.num_reg_tokens, 8) if reg_variant is not None else None,
    )
    (output.image.sum() + (output.reg.sum() if output.reg is not None else 0)).backward()
    state = clean_state_dict(model)
    assert state and not any("_orig_mod" in key for key in state)
