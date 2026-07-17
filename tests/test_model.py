from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from rae_ablation.config import CompileConfig, ModelConfig
from rae_ablation.model.attnres import AttnResRouter
from rae_ablation.model.compile import clean_state_dict, compile_model_regions
from rae_ablation.model.ddt import DDTOutput, RAEv2DDTIG
from rae_ablation.model.factory import build_diffusion_model
from rae_ablation.model.rope import apply_2d_rope


def tiny_config(
    *,
    checkpointing: bool = False,
    patch_forcing: bool = False,
    attnres_block_size: int | None = None,
    encoder_depth: int = 3,
    base_model_depth: int = 2,
) -> ModelConfig:
    return ModelConfig(
        in_channels=4,
        use_patch_forcing=patch_forcing,
        attnres_block_size=attnres_block_size,
        num_input_residuals=4,
        encoder_hidden_size=16,
        decoder_hidden_size=32,
        encoder_depth=encoder_depth,
        decoder_depth=2,
        encoder_num_heads=4,
        decoder_num_heads=8,
        mlp_ratio=2.0,
        base_model_depth=base_model_depth,
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
@pytest.mark.parametrize("patch_forcing", [False, True])
def test_ddt_forward_backward_both_grids(grid: int, patch_forcing: bool) -> None:
    model = RAEv2DDTIG(tiny_config(checkpointing=True, patch_forcing=patch_forcing))
    latent = torch.randn(2, 4, grid, grid, requires_grad=True)
    timestep = torch.rand(2, 1, grid, grid) if patch_forcing else torch.rand(2)
    output = model(latent, timestep, torch.tensor([1, 2]))
    assert isinstance(output, DDTOutput)
    assert output.full.shape == latent.shape
    assert output.early.shape == latent.shape
    assert torch.count_nonzero(output.full) == 0
    assert torch.count_nonzero(output.early) == 0
    (output.full.sum() + output.early.sum()).backward()
    assert model.full_head.projection.weight.grad is not None
    assert model.early_head.projection.weight.grad is not None


@pytest.mark.parametrize("attnres_block_size", [None, 4])
def test_ddt_condition_token_shapes_and_base_depth_isolation(
    attnres_block_size: int | None,
) -> None:
    model = RAEv2DDTIG(tiny_config(attnres_block_size=attnres_block_size))
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


def test_patch_forcing_omits_time_tokens_and_conditions_outputs_per_patch() -> None:
    model = RAEv2DDTIG(tiny_config(patch_forcing=True))
    assert model.time_embedding.learnable_tokens is None

    encoder_inputs: list[torch.Tensor] = []
    decoder_conditioning: list[torch.Tensor] = []
    early_conditioning: list[torch.Tensor] = []
    handles = [
        model.encoder_blocks[0].register_forward_pre_hook(
            lambda _module, args: encoder_inputs.append(args[0].detach().clone())
        ),
        model.decoder_blocks[0].register_forward_pre_hook(
            lambda _module, args: decoder_conditioning.append(args[1].detach().clone())
        ),
        model.early_head.register_forward_pre_hook(
            lambda _module, args: early_conditioning.append(args[1].detach().clone())
        ),
    ]
    latent = torch.randn(2, 4, 2, 2)
    labels = torch.tensor([1, 2])
    first = torch.full((2, 1, 2, 2), 0.2)
    second = first.clone()
    second[:, :, 0, 0] = 0.8
    model(latent, first, labels)
    model(latent, second, labels)
    for handle in handles:
        handle.remove()

    expected_tokens = 2 * 2 + model.config.num_class_tokens
    assert encoder_inputs[0].shape == (2, expected_tokens, 16)
    assert torch.equal(encoder_inputs[0], encoder_inputs[1])
    assert not torch.equal(decoder_conditioning[0], decoder_conditioning[1])
    assert not torch.equal(early_conditioning[0], early_conditioning[1])
    assert torch.equal(decoder_conditioning[0][:, 1:], decoder_conditioning[1][:, 1:])
    assert torch.equal(early_conditioning[0][:, 1:], early_conditioning[1][:, 1:])


def test_patch_forcing_scalar_timestep_matches_constant_map() -> None:
    model = RAEv2DDTIG(tiny_config(patch_forcing=True))
    nn.init.normal_(model.full_head.projection.weight)
    nn.init.normal_(model.early_head.projection.weight)
    latent = torch.randn(2, 4, 2, 2)
    timestep = torch.rand(2)
    labels = torch.tensor([1, 2])
    scalar = model(latent, timestep, labels)
    patchwise = model(
        latent,
        timestep.reshape(2, 1, 1, 1).expand(-1, -1, 2, 2),
        labels,
    )
    torch.testing.assert_close(scalar.full, patchwise.full)
    torch.testing.assert_close(scalar.early, patchwise.early)


def test_timestep_shape_validation() -> None:
    latent = torch.randn(2, 4, 2, 2)
    labels = torch.tensor([1, 2])
    patch_model = RAEv2DDTIG(tiny_config(patch_forcing=True))
    with pytest.raises(ValueError, match="timestep"):
        patch_model(latent, torch.rand(2, 2, 2), labels)
    with pytest.raises(ValueError, match="timestep"):
        patch_model(latent, torch.rand(2, 1, 3, 3), labels)
    baseline = RAEv2DDTIG(tiny_config())
    with pytest.raises(ValueError, match="patchwise"):
        baseline(latent, torch.rand(2, 1, 2, 2), labels)


@pytest.mark.parametrize("patch_forcing", [False, True])
def test_checkpointed_ddt_matches_plain_outputs_and_gradients(
    patch_forcing: bool,
) -> None:
    plain = RAEv2DDTIG(tiny_config(checkpointing=False, patch_forcing=patch_forcing))
    checked = copy.deepcopy(plain)
    checked.set_gradient_checkpointing(True)
    nn.init.normal_(plain.full_head.projection.weight)
    nn.init.normal_(plain.early_head.projection.weight)
    checked.load_state_dict(plain.state_dict())
    latent1 = torch.randn(2, 4, 2, 2, requires_grad=True)
    latent2 = latent1.detach().clone().requires_grad_(True)
    timestep = torch.rand(2, 1, 2, 2) if patch_forcing else torch.rand(2)
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


@pytest.mark.parametrize(
    ("block_size", "expected_final_sources"),
    [(2, 10), (4, 7), (6, 6)],
)
def test_attnres_block_boundaries_and_uniform_initial_weights(
    block_size: int, expected_final_sources: int
) -> None:
    model = RAEv2DDTIG(
        tiny_config(
            attnres_block_size=block_size,
            encoder_depth=6,
            base_model_depth=3,
        )
    )
    latent = torch.randn(2, 4, 2, 2)
    with model.capture_attnres_weights(2, 2) as capture:
        model(latent, torch.rand(2), torch.tensor([1, 2]))

    final = capture.records[-1]
    assert final.site == "encoder.final_output"
    assert len(final.source_labels) == expected_final_sources
    for record in capture.records:
        expected = torch.full_like(record.spatial, 1 / len(record.source_labels))
        torch.testing.assert_close(record.spatial, expected)
        torch.testing.assert_close(record.time, expected)
        torch.testing.assert_close(record.classes, expected)


@pytest.mark.parametrize("patch_forcing", [False, True])
def test_attnres_multiple_input_views_forward_backward_and_encoder_only(
    patch_forcing: bool,
) -> None:
    model = RAEv2DDTIG(
        tiny_config(
            checkpointing=True,
            patch_forcing=patch_forcing,
            attnres_block_size=4,
        )
    )
    assert model.encoder_input is None
    assert model.input_residuals_proj is not None
    assert model.input_residuals_proj.weight.shape == (4 * 16, 4)
    chunks = model.input_residuals_proj.weight.chunk(4, dim=0)
    assert all(chunk.shape == (16, 4) for chunk in chunks)
    assert not torch.equal(chunks[0], chunks[1])
    assert not any(isinstance(module, AttnResRouter) for module in model.decoder_blocks.modules())

    nn.init.normal_(model.full_head.projection.weight)
    nn.init.normal_(model.early_head.projection.weight)
    latent = torch.randn(2, 4, 2, 2, requires_grad=True)
    timestep = torch.rand(2, 1, 2, 2) if patch_forcing else torch.rand(2)
    output = model(latent, timestep, torch.tensor([1, 2]))
    (output.full.sum() + output.early.sum()).backward()
    assert latent.grad is not None and torch.count_nonzero(latent.grad)
    assert model.input_residuals_proj.weight.grad is not None
    assert torch.count_nonzero(model.input_residuals_proj.weight.grad)


def test_checkpointed_attnres_matches_plain_outputs_and_gradients() -> None:
    plain = RAEv2DDTIG(tiny_config(attnres_block_size=4))
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
    torch.testing.assert_close(out1.full, out2.full)
    torch.testing.assert_close(out1.early, out2.early)
    (out1.full.sum() + out1.early.sum()).backward()
    (out2.full.sum() + out2.early.sum()).backward()
    torch.testing.assert_close(latent1.grad, latent2.grad)
    for first, second in zip(plain.parameters(), checked.parameters(), strict=True):
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad)


def test_attnres_tiny_regional_compile() -> None:
    model = RAEv2DDTIG(tiny_config(attnres_block_size=4))
    view = compile_model_regions(model, CompileConfig(True, "eager", "default", False, False))
    output = view(torch.randn(1, 4, 2, 2), torch.rand(1), torch.tensor([1]))
    assert output.full.shape == (1, 4, 2, 2)
