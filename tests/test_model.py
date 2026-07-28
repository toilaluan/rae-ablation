from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import jit_reg
from jit_reg.config import (
    DDTDecoderConfig,
    EncoderConfig,
    FeaturesConfig,
    ModelConfig,
    REGConfig,
    REGLoRAConfig,
    REPAConfig,
)
from jit_reg.model import ddt
from jit_reg.model.ddt import (
    DDT,
    Attention,
    ConditionEmbedder,
    DDTDecoderBlock,
    DDTEncoderBlock,
    DDTInternalREGHead,
    DDTOutput,
    DDTREGOutput,
    FusedSwiGLU,
    GaussianFourierEmbedding,
    LowRankAdapter,
    SwiGLU,
)


def tiny_config(
    *,
    reg: bool = False,
    repa: bool = False,
    scope: str = "image",
    num_img_t_tokens: int = 1,
    num_reg_t_tokens: int = 0,
    num_c_tokens: int = 1,
) -> ModelConfig:
    return ModelConfig(
        image_size=16,
        patch_size=8,
        num_img_t_tokens=num_img_t_tokens,
        num_reg_t_tokens=num_reg_t_tokens,
        num_c_tokens=num_c_tokens,
        hidden_size=16,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        num_classes=3,
        decoder=DDTDecoderConfig(
            hidden_size=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2,
        ),
        features=FeaturesConfig(
            encoder=EncoderConfig(
                feature_dim=8,
                register_tokens=4,
            ),
            reg=REGConfig(
                enabled=reg,
                pool_size=2,
                internal_guidance_block=1,
            ),
            repa=REPAConfig(
                enabled=repa,
                block=1,
                projection_hidden_size=16,
                token_scope=scope,
            ),
        ),
    )


def inputs(config: ModelConfig) -> tuple[torch.Tensor, ...]:
    batch = 2
    image = torch.randn(batch, 3, 16, 16)
    timestep = torch.tensor([0.2, 0.7])
    labels = torch.tensor([0, 2])
    reg = config.features.reg
    noisy_reg = (
        torch.randn(
            batch,
            reg.num_tokens,
            config.features.encoder.feature_dim,
        )
        if reg.enabled
        else None
    )
    return image, timestep, labels, noisy_reg


def run(
    config: ModelConfig,
    *,
    return_repa: bool = False,
) -> DDTOutput:
    model = DDT(config)
    image, timestep, labels, noisy_reg = inputs(config)
    return model(
        image,
        timestep,
        labels,
        noisy_reg=noisy_reg,
        reg_timestep=timestep if noisy_reg is not None else None,
        return_repa=return_repa,
    )


def test_feature_disabled_model_is_baseline_ddt() -> None:
    model = DDT(tiny_config())
    assert len(model.encoder_blocks) == 2
    assert len(model.decoder_blocks) == 2
    assert all(
        isinstance(block, DDTEncoderBlock)
        for block in model.encoder_blocks
    )
    assert all(
        isinstance(block, DDTDecoderBlock)
        for block in model.decoder_blocks
    )
    assert model.encoder_reg_projection is None
    assert model.decoder_reg_projection is None
    assert model.final_layer.reg_projection is None
    assert model.internal_reg_head is None
    assert model.repa_projection is None
    assert not any(
        key.startswith(("encoder_reg_", "decoder_reg_", "repa_"))
        for key in model.state_dict()
    )
    image, timestep, labels, _ = inputs(model.config)
    output = model(image, timestep, labels)
    assert output.image.shape == image.shape
    assert output.reg is None and output.repa is None and output.reg_early is None


def test_default_reg_model_keeps_legacy_modules_and_state_keys() -> None:
    model = DDT(tiny_config(reg=True))
    assert all(
        isinstance(block.feed_forward, SwiGLU)
        for block in (*model.encoder_blocks, *model.decoder_blocks)
    )
    assert all(
        not block.attention.independent_reg_attention
        for block in (*model.encoder_blocks, *model.decoder_blocks)
    )
    assert not any("lora" in key for key in model.state_dict())


def test_fused_swiglu_matches_concatenated_unfused_weights() -> None:
    unfused = SwiGLU(16, 2)
    fused = FusedSwiGLU(16, 2)
    with torch.no_grad():
        fused.gate_value.weight.copy_(
            torch.cat((unfused.gate.weight, unfused.value.weight))
        )
        fused.gate_value.bias.copy_(
            torch.cat((unfused.gate.bias, unfused.value.bias))
        )
        fused.projection.load_state_dict(
            unfused.projection.state_dict()
        )
    unfused_input = torch.randn(2, 7, 16, requires_grad=True)
    fused_input = unfused_input.detach().clone().requires_grad_(True)
    unfused_output = unfused(unfused_input)
    fused_output = fused(fused_input)
    assert torch.allclose(
        unfused_output,
        fused_output,
        atol=1e-6,
        rtol=1e-6,
    )
    unfused_output.sum().backward()
    fused_output.sum().backward()
    assert torch.allclose(
        unfused_input.grad,
        fused_input.grad,
        atol=1e-6,
        rtol=1e-6,
    )


def test_reg_lora_is_zero_initialized_and_follows_mlp_layout() -> None:
    for fused in (False, True):
        config = tiny_config(reg=True)
        config.fused_mlp = fused
        config.features.reg.lora = REGLoRAConfig(
            enabled=True,
            rank=4,
            alpha=4,
            mlp=True,
        )
        model = DDT(config)
        adapters = [
            module
            for module in model.modules()
            if isinstance(module, LowRankAdapter)
        ]
        assert adapters
        assert all(
            torch.count_nonzero(adapter.up.weight) == 0
            for adapter in adapters
        )
        assert all(
            block.attention.reg_qkv_lora is not None
            for block in (*model.encoder_blocks, *model.decoder_blocks)
        )
        block = model.encoder_blocks[0]
        states = torch.randn(
            2,
            block.attention.expected_tokens,
            config.hidden_size,
            requires_grad=True,
        )
        block(states).square().mean().backward()
        assert block.attention.reg_qkv_lora is not None
        assert block.attention.reg_qkv_lora.up.weight.grad is not None
        if fused:
            assert all(
                isinstance(block.feed_forward, FusedSwiGLU)
                and block.feed_forward.reg_gate_value_lora is not None
                for block in (*model.encoder_blocks, *model.decoder_blocks)
            )
            assert isinstance(block.feed_forward, FusedSwiGLU)
            assert (
                block.feed_forward.reg_gate_value_lora is not None
                and block.feed_forward.reg_gate_value_lora.up.weight.grad
                is not None
            )
        else:
            assert all(
                isinstance(block.feed_forward, SwiGLU)
                and block.feed_forward.reg_gate_lora is not None
                and block.feed_forward.reg_value_lora is not None
                for block in (*model.encoder_blocks, *model.decoder_blocks)
            )
            assert isinstance(block.feed_forward, SwiGLU)
            assert (
                block.feed_forward.reg_gate_lora is not None
                and block.feed_forward.reg_gate_lora.up.weight.grad
                is not None
            )


def test_independent_attention_is_asymmetric_and_reg_only_matches() -> None:
    attention = Attention(
        16,
        4,
        grid_size=2,
        reg_pool_size=2,
        condition_tokens=2,
        independent_reg_attention=True,
    )
    x = torch.randn(2, 10, 16)
    changed_image = x.clone()
    changed_image[:, 4:8] = torch.randn_like(changed_image[:, 4:8])
    output = attention(x)
    changed_image_output = attention(changed_image)
    assert torch.allclose(output[:, :4], changed_image_output[:, :4])
    assert torch.allclose(output[:, 8:], changed_image_output[:, 8:])
    assert not torch.allclose(
        output[:, 4:8],
        changed_image_output[:, 4:8],
    )

    changed_reg = x.clone()
    changed_reg[:, :4] = torch.randn_like(changed_reg[:, :4])
    changed_reg_output = attention(changed_reg)
    assert not torch.allclose(output[:, 4:8], changed_reg_output[:, 4:8])
    dropped_output = attention(x, drop_reg_for_image=True)
    changed_reg_dropped_output = attention(
        changed_reg,
        drop_reg_for_image=True,
    )
    assert torch.allclose(
        dropped_output[:, 4:8],
        changed_reg_dropped_output[:, 4:8],
    )

    reg_condition = torch.cat((x[:, :4], x[:, 8:]), dim=1)
    reg_only_output = attention(reg_condition, reg_only=True)
    assert torch.allclose(reg_only_output[:, :4], output[:, :4])
    assert torch.allclose(reg_only_output[:, 4:], output[:, 8:])


def test_ddt_reg_only_forward_skips_image_projections() -> None:
    config = tiny_config(
        reg=True,
        num_reg_t_tokens=1,
    )
    config.features.reg.independent_attention = True
    model = DDT(config)
    image, timestep, labels, noisy_reg = inputs(config)
    assert noisy_reg is not None
    full = model(
        image,
        timestep,
        labels,
        noisy_reg=noisy_reg,
        reg_timestep=timestep,
    )
    calls: list[str] = []
    handles = [
        module.register_forward_hook(
            lambda _module, _args, _output, name=name: calls.append(name)
        )
        for name, module in (
            ("encoder", model.encoder_patch_projection),
            ("decoder", model.decoder_patch_projection),
            ("image_output", model.final_layer.image_projection),
        )
    ]
    reg_only = model.forward_reg(
        noisy_reg,
        timestep,
        labels,
        reg_timestep=timestep,
    )
    for handle in handles:
        handle.remove()
    assert isinstance(reg_only, DDTREGOutput)
    assert calls == []
    assert full.reg is not None and full.reg_early is not None
    assert torch.allclose(full.reg, reg_only.reg)
    assert torch.allclose(full.reg_early, reg_only.reg_early)


@pytest.mark.parametrize(
    ("reg", "scope", "expected_repa_tokens"),
    [
        (False, "image", 4),
        (True, "image", 4),
        (True, "image_and_reg", 8),
    ],
)
def test_output_shapes(
    reg: bool,
    scope: str,
    expected_repa_tokens: int,
) -> None:
    config = tiny_config(
        reg=reg,
        repa=True,
        scope=scope,
    )
    output = run(config, return_repa=True)
    assert output.image.shape == (2, 3, 16, 16)
    assert output.repa is not None
    assert output.repa.shape == (
        2,
        expected_repa_tokens,
        config.features.encoder.feature_dim,
    )
    if reg:
        assert output.reg is not None and output.reg_early is not None
        assert output.reg.shape == (
            2,
            config.features.reg.num_tokens,
            config.features.encoder.feature_dim,
        )
        assert output.reg_early.shape == output.reg.shape
    else:
        assert output.reg is None and output.reg_early is None


def test_reg_requires_explicit_reg_timestep() -> None:
    config = tiny_config(reg=True)
    model = DDT(config)
    image, timestep, labels, noisy_reg = inputs(config)
    with pytest.raises(ValueError, match="reg_timestep"):
        model(
            image,
            timestep,
            labels,
            noisy_reg=noisy_reg,
        )


def test_encoder_is_unconditioned_and_decoder_is_joint() -> None:
    config = tiny_config(
        reg=True,
        num_img_t_tokens=2,
        num_reg_t_tokens=3,
        num_c_tokens=2,
    )
    model = DDT(config)
    assert list(
        inspect.signature(model.encoder_blocks[0].forward).parameters
    ) == ["x", "reg_only", "drop_reg_for_image"]
    assert list(
        inspect.signature(model.decoder_blocks[0].forward).parameters
    ) == [
        "x",
        "conditioning",
        "reg_only",
        "drop_reg_for_image",
    ]

    encoder_inputs: list[torch.Tensor] = []
    decoder_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    encoder_handle = model.encoder_blocks[0].register_forward_pre_hook(
        lambda _module, args: encoder_inputs.append(
            args[0].detach().clone()
        )
    )
    decoder_handle = model.decoder_blocks[0].register_forward_pre_hook(
        lambda _module, args: decoder_inputs.append(
            (
                args[0].detach().clone(),
                args[1].detach().clone(),
            )
        )
    )
    image, timestep, labels, noisy_reg = inputs(config)
    assert noisy_reg is not None
    reg_timestep = torch.tensor([0.8, 0.9])
    model(
        image,
        timestep,
        labels,
        noisy_reg=noisy_reg,
        reg_timestep=reg_timestep,
    )
    encoder_handle.remove()
    decoder_handle.remove()

    reg_result = model.reg_t_embedder(
        reg_timestep,
        return_base_embed=True,
    )
    image_result = model.img_t_embedder(
        timestep,
        return_base_embed=True,
    )
    assert isinstance(reg_result, tuple)
    assert isinstance(image_result, tuple)
    _reg_base, reg_tokens = reg_result
    _image_base, image_tokens = image_result
    assert model.encoder_reg_projection is not None
    expected_encoder = torch.cat(
        (
            model.encoder_reg_projection(noisy_reg),
            model.encoder_patch_projection(image)
            .flatten(2)
            .transpose(1, 2),
            reg_tokens,
            image_tokens,
            model.condition_embedder(labels),
        ),
        dim=1,
    )
    assert len(encoder_inputs) == 1
    assert torch.allclose(encoder_inputs[0], expected_encoder)
    assert encoder_inputs[0].shape == (
        2,
        model.num_primary_tokens + model.num_condition_tokens,
        config.hidden_size,
    )

    assert len(decoder_inputs) == 1
    decoder_state, decoder_context = decoder_inputs[0]
    assert model.decoder_reg_projection is not None
    expected_decoder = torch.cat(
        (
            model.decoder_reg_projection(noisy_reg),
            model.decoder_patch_projection(image)
            .flatten(2)
            .transpose(1, 2),
        ),
        dim=1,
    )
    assert torch.allclose(decoder_state, expected_decoder)
    assert decoder_state.shape == decoder_context.shape == (
        2,
        model.num_primary_tokens,
        config.decoder.hidden_size,
    )


def test_decoder_context_uses_token_specific_base_times() -> None:
    config = tiny_config(
        reg=True,
        num_reg_t_tokens=1,
    )
    model = DDT(config)
    encoder_output: list[torch.Tensor] = []
    decoder_context: list[torch.Tensor] = []
    encoder_handle = model.encoder_blocks[-1].register_forward_hook(
        lambda _module, _args, output: encoder_output.append(
            output.detach().clone()
        )
    )
    decoder_handle = model.decoder_blocks[0].register_forward_pre_hook(
        lambda _module, args: decoder_context.append(
            args[1].detach().clone()
        )
    )
    image, timestep, labels, noisy_reg = inputs(config)
    assert noisy_reg is not None
    reg_timestep = torch.tensor([0.8, 0.9])
    model(
        image,
        timestep,
        labels,
        noisy_reg=noisy_reg,
        reg_timestep=reg_timestep,
    )
    encoder_handle.remove()
    decoder_handle.remove()
    reg_result = model.reg_t_embedder(
        reg_timestep,
        return_base_embed=True,
    )
    image_result = model.img_t_embedder(
        timestep,
        return_base_embed=True,
    )
    assert isinstance(reg_result, tuple)
    assert isinstance(image_result, tuple)
    reg_base, _reg_tokens = reg_result
    image_base, _image_tokens = image_result
    primary = encoder_output[0][:, : model.num_primary_tokens]
    expected = model.context_projection(
        torch.cat(
            (
                F.silu(primary[:, : model.num_reg_tokens] + reg_base),
                F.silu(primary[:, model.image_token_start :] + image_base),
            ),
            dim=1,
        )
    )
    assert torch.allclose(decoder_context[0], expected)


def test_shared_final_layer_emits_image_and_reg_without_reg_head() -> None:
    model = DDT(tiny_config(reg=True))
    assert isinstance(model.final_layer.norm, nn.RMSNorm)
    assert model.final_layer.reg_projection is not None
    assert isinstance(model.internal_reg_head, DDTInternalREGHead)
    assert not hasattr(model, "reg_reinjection")
    state = model.state_dict()
    assert "final_layer.image_projection.weight" in state
    assert "final_layer.reg_projection.weight" in state
    assert "internal_reg_head.projection.weight" in state
    assert not any(
        "reinjection" in key
        or "reg_position_embedding" in key
        for key in state
    )


def test_internal_reg_head_is_zero_initialized_and_uses_selected_block() -> None:
    config = tiny_config(reg=True)
    config.features.reg.internal_guidance_block = 2
    model = DDT(config)
    assert model.internal_reg_head is not None
    assert torch.count_nonzero(model.internal_reg_head.modulation[-1].weight) == 0
    assert torch.count_nonzero(model.internal_reg_head.projection.weight) == 0

    selected: list[torch.Tensor] = []
    handle = model.encoder_blocks[1].register_forward_hook(
        lambda _module, _args, output: selected.append(output.detach().clone())
    )
    image, timestep, labels, noisy_reg = inputs(config)
    output = model(
        image,
        timestep,
        labels,
        noisy_reg=noisy_reg,
        reg_timestep=timestep,
    )
    handle.remove()
    assert output.reg_early is not None
    assert torch.count_nonzero(output.reg_early) == 0
    assert len(selected) == 1


def test_gaussian_fourier_embedding_returns_base_and_context_tokens() -> None:
    embedder = GaussianFourierEmbedding(
        hidden_size=8,
        n_tokens=3,
        embedding_size=4,
        scale=0.5,
    )
    timestep = torch.tensor([0.2, 0.7])
    result = embedder(timestep, return_base_embed=True)
    assert isinstance(result, tuple)
    base, context = result
    assert base.shape == (2, 1, 8)
    assert context.shape == (2, 3, 8)
    assert torch.allclose(
        context,
        embedder.learnable_tokens + base,
    )
    assert not embedder.W.requires_grad


def test_label_condition_embedder_returns_learned_context_tokens() -> None:
    embedder = ConditionEmbedder(
        hidden_size=8,
        num_classes=3,
        condition_type="label",
        n_tokens=2,
    )
    labels = torch.tensor([0, 3])
    base = embedder.base_embedding(labels)
    context = embedder(labels)
    assert base.shape == (2, 8)
    assert context.shape == (2, 2, 8)
    assert torch.allclose(
        context,
        embedder.learnable_tokens + base.unsqueeze(1),
    )


def test_conditioning_tokens_are_excluded_from_repa_outputs() -> None:
    config = tiny_config(
        reg=True,
        repa=True,
        scope="image_and_reg",
        num_img_t_tokens=2,
        num_reg_t_tokens=3,
        num_c_tokens=4,
    )
    output = run(config, return_repa=True)
    assert output.repa is not None
    assert output.repa.shape == (
        2,
        config.features.reg.num_tokens + 4,
        config.features.encoder.feature_dim,
    )


def test_removed_sit_and_independent_reg_components_are_absent() -> None:
    source = inspect.getsource(ddt)
    for retired in (
        "SiT",
        "WideREG",
        "reg_reinjection",
        "early_prediction",
        "reg_position_embedding",
    ):
        assert retired not in source


def test_public_api_exposes_ddt_without_sit_aliases() -> None:
    assert jit_reg.DDT is DDT
    assert jit_reg.DDTOutput is DDTOutput
    assert "DDT" in jit_reg.__all__
    assert "DDTOutput" in jit_reg.__all__
    assert "SiT" not in jit_reg.__all__
    assert "SiTOutput" not in jit_reg.__all__
    assert not hasattr(jit_reg, "SiT")
    assert not hasattr(jit_reg, "SiTOutput")
