from __future__ import annotations

import inspect

import pytest
import torch
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
    REPAStreamConfig,
)
from jit_reg.model.ddt import DDT, DDTOutput, DDTREGOutput
from jit_reg.model.layers import (
    DDTAttention,
    DDTDecoderBlock,
    DDTEncoderBlock,
    FusedSwiGLU,
    LowRankAdapter,
    SwiGLU,
)
from jit_reg.model.tokens import (
    ClassTokenEmbedder,
    ClassTokens,
    DDTTokenLayout,
    TimestepTokenEmbedder,
    TimestepTokens,
)


def tiny_config(
    *,
    reg: bool = False,
    repa: bool = False,
    repa_stream: bool = False,
    scope: str = "image",
    attention_mode: str = "joint",
    image_timestep_tokens: int = 1,
    reg_timestep_tokens: int = 0,
    image_class_tokens: int = 1,
    reg_class_tokens: int = 0,
) -> ModelConfig:
    return ModelConfig(
        image_size=16,
        patch_size=8,
        num_image_timestep_tokens=image_timestep_tokens,
        num_reg_timestep_tokens=reg_timestep_tokens,
        num_image_class_tokens=image_class_tokens,
        num_reg_class_tokens=reg_class_tokens,
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
                attention_mode=attention_mode,
                join_after_block=1,
            ),
            repa=REPAConfig(
                enabled=repa,
                block=1,
                projection_hidden_size=16,
                token_scope=scope,
                repa_stream=REPAStreamConfig(
                    enabled=repa_stream,
                    pool_size=2,
                ),
            ),
        ),
    )


def inputs(
    config: ModelConfig,
) -> tuple[torch.Tensor, ...]:
    image = torch.randn(2, 3, 16, 16)
    timestep = torch.tensor([0.2, 0.7])
    labels = torch.tensor([0, 2])
    noisy_reg = (
        torch.randn(
            2,
            config.features.reg.num_tokens,
            config.features.encoder.feature_dim,
        )
        if config.features.reg.enabled
        else None
    )
    return image, timestep, labels, noisy_reg


@pytest.mark.parametrize(
    "counts",
    [
        (0, 4, 0, 0, 1, 1),
        (0, 4, 0, 0, 0, 0),
        (3, 4, 2, 3, 4, 5),
    ],
)
def test_token_layout_is_the_sequence_authority(
    counts: tuple[int, ...],
) -> None:
    layout = DDTTokenLayout(*counts)
    sizes = [
        layout.reg_data_tokens,
        layout.image_data_tokens,
        layout.reg_timestep_tokens,
        layout.reg_class_tokens,
        layout.image_timestep_tokens,
        layout.image_class_tokens,
    ]
    parts = [
        torch.full((2, size, 7), float(index))
        for index, size in enumerate(sizes)
    ]
    packed = layout.pack(
        reg_data=parts[0],
        image_data=parts[1],
        reg_timestep=parts[2],
        reg_class=parts[3],
        image_timestep=parts[4],
        image_class=parts[5],
    )
    assert packed.shape == (2, sum(sizes), 7)
    assert layout.gather_data(packed).shape[1] == sum(sizes[:2])
    reg_stream = layout.gather_reg_stream(packed)
    image_stream = layout.gather_image_stream(packed)
    assert torch.equal(
        layout.reconstruct_streams(reg_stream, image_stream),
        packed,
    )
    assert packed[:, layout.image_data].shape[1] == sizes[1]


def test_token_layout_inserts_repa_stream_between_image_and_conditions() -> None:
    layout = DDTTokenLayout(
        reg_data_tokens=1,
        image_data_tokens=4,
        reg_timestep_tokens=1,
        reg_class_tokens=1,
        image_timestep_tokens=1,
        image_class_tokens=1,
        repa_stream_tokens=2,
    )
    values = {
        "reg_data": torch.full((2, 1, 7), 1.0),
        "image_data": torch.full((2, 4, 7), 2.0),
        "repa_stream": torch.full((2, 2, 7), 3.0),
        "reg_timestep": torch.full((2, 1, 7), 4.0),
        "reg_class": torch.full((2, 1, 7), 5.0),
        "image_timestep": torch.full((2, 1, 7), 6.0),
        "image_class": torch.full((2, 1, 7), 7.0),
    }
    packed = layout.pack(**values)
    assert torch.equal(packed[:, layout.repa_stream], values["repa_stream"])
    assert layout.gather_data(packed).shape[1] == 5
    assert layout.gather_image_data(packed).shape[1] == 6
    reg_stream = layout.gather_reg_stream(packed)
    image_stream = layout.gather_image_stream(packed)
    assert torch.equal(
        layout.reconstruct_streams(reg_stream, image_stream),
        packed,
    )


def test_timestep_and_class_tokens_are_named_and_stream_private() -> None:
    time = TimestepTokenEmbedder(
        8,
        num_tokens=3,
        embedding_size=4,
    )
    result = time(torch.tensor([0.2, 0.7]))
    assert isinstance(result, TimestepTokens)
    assert result.base.shape == (2, 1, 8)
    assert result.tokens.shape == (2, 3, 8)
    assert torch.allclose(result.tokens, result.base + time.offsets)

    classes = ClassTokenEmbedder(
        8,
        3,
        num_image_tokens=2,
        num_reg_tokens=4,
    )
    labels = torch.tensor([0, 3])
    tokens = classes(labels)
    assert isinstance(tokens, ClassTokens)
    assert tokens.image_tokens.shape == (2, 2, 8)
    assert tokens.reg_tokens.shape == (2, 4, 8)
    base = classes.embedding_table(labels).unsqueeze(1)
    assert torch.allclose(
        tokens.image_tokens,
        base + classes.image_offsets,
    )
    assert torch.allclose(
        tokens.reg_tokens,
        base + classes.reg_offsets,
    )
    assert len(
        [
            module
            for module in classes.modules()
            if isinstance(module, nn.Embedding)
        ]
    ) == 1


def test_modeling_source_has_no_retired_conditioning_or_mode_flags() -> None:
    import jit_reg.model.ddt as ddt
    import jit_reg.model.layers as layers
    import jit_reg.model.tokens as tokens

    source = "\n".join(
        inspect.getsource(module)
        for module in (ddt, layers, tokens)
    )
    for retired in (
        "condition_type",
        "context_dim",
        "num_img_t_tokens",
        "num_reg_t_tokens",
        "num_c_tokens",
        "drop_reg_for_image",
        "isolate_image_from_reg",
    ):
        assert retired not in source


def asymmetric_attention() -> tuple[DDTAttention, DDTTokenLayout]:
    layout = DDTTokenLayout(
        reg_data_tokens=4,
        image_data_tokens=4,
        reg_timestep_tokens=1,
        reg_class_tokens=1,
        image_timestep_tokens=1,
        image_class_tokens=1,
    )
    return (
        DDTAttention(
            16,
            4,
            grid_size=2,
            reg_pool_size=2,
            layout=layout,
            attention_mode="asymmetric",
        ),
        layout,
    )


def test_asymmetric_attention_has_exact_query_kv_policy() -> None:
    attention, layout = asymmetric_attention()
    x = torch.randn(2, layout.total_tokens, 16)
    isolated = attention(x, image_attends_reg=False)

    changed_reg = x.clone()
    changed_reg[:, layout.reg_data] += torch.randn_like(
        changed_reg[:, layout.reg_data]
    )
    changed_reg[:, layout.reg_timestep] += 3
    changed_reg[:, layout.reg_class] -= 2
    changed_isolated = attention(
        changed_reg,
        image_attends_reg=False,
    )
    assert torch.allclose(
        isolated[:, layout.image_data],
        changed_isolated[:, layout.image_data],
    )
    assert torch.allclose(
        isolated[:, layout.image_timestep],
        changed_isolated[:, layout.image_timestep],
    )
    assert torch.allclose(
        isolated[:, layout.image_class],
        changed_isolated[:, layout.image_class],
    )

    joined = attention(x, image_attends_reg=True)
    changed_joined = attention(
        changed_reg,
        image_attends_reg=True,
    )
    assert not torch.allclose(
        joined[:, layout.image_data],
        changed_joined[:, layout.image_data],
    )
    assert torch.allclose(
        joined[:, layout.image_timestep],
        changed_joined[:, layout.image_timestep],
    )
    assert torch.allclose(
        joined[:, layout.image_class],
        changed_joined[:, layout.image_class],
    )

    changed_image = x.clone()
    for token_slice in layout.image_stream_slices:
        changed_image[:, token_slice] += torch.randn_like(
            changed_image[:, token_slice]
        )
    assert torch.allclose(
        joined[:, layout.reg_data],
        attention(
            changed_image,
            image_attends_reg=True,
        )[:, layout.reg_data],
    )


def test_repa_stream_is_image_side_memory_under_asymmetric_attention() -> None:
    layout = DDTTokenLayout(
        reg_data_tokens=4,
        image_data_tokens=4,
        reg_timestep_tokens=1,
        reg_class_tokens=1,
        image_timestep_tokens=1,
        image_class_tokens=1,
        repa_stream_tokens=4,
    )
    attention = DDTAttention(
        16,
        4,
        grid_size=2,
        reg_pool_size=2,
        repa_pool_size=2,
        layout=layout,
        attention_mode="asymmetric",
    )
    x = torch.randn(2, layout.total_tokens, 16)
    isolated = attention(x, image_attends_reg=False)

    changed_repa = x.clone()
    changed_repa[:, layout.repa_stream] += torch.randn_like(
        changed_repa[:, layout.repa_stream]
    )
    changed_repa_output = attention(
        changed_repa,
        image_attends_reg=False,
    )
    assert not torch.allclose(
        isolated[:, layout.image_data],
        changed_repa_output[:, layout.image_data],
    )
    assert torch.allclose(
        isolated[:, layout.reg_data],
        changed_repa_output[:, layout.reg_data],
    )

    changed_image = x.clone()
    changed_image[:, layout.image_data] += torch.randn_like(
        changed_image[:, layout.image_data]
    )
    changed_image_output = attention(
        changed_image,
        image_attends_reg=False,
    )
    assert not torch.allclose(
        isolated[:, layout.repa_stream],
        changed_image_output[:, layout.repa_stream],
    )
    assert torch.allclose(
        isolated[:, layout.reg_data],
        changed_image_output[:, layout.reg_data],
    )

    changed_reg = x.clone()
    for token_slice in layout.reg_stream_slices:
        changed_reg[:, token_slice] += torch.randn_like(
            changed_reg[:, token_slice]
        )
    changed_reg_isolated = attention(
        changed_reg,
        image_attends_reg=False,
    )
    assert torch.allclose(
        isolated[:, layout.image_data],
        changed_reg_isolated[:, layout.image_data],
    )
    assert torch.allclose(
        isolated[:, layout.repa_stream],
        changed_reg_isolated[:, layout.repa_stream],
    )

    joined = attention(x, image_attends_reg=True)
    changed_reg_joined = attention(
        changed_reg,
        image_attends_reg=True,
    )
    assert not torch.allclose(
        joined[:, layout.image_data],
        changed_reg_joined[:, layout.image_data],
    )
    assert not torch.allclose(
        joined[:, layout.repa_stream],
        changed_reg_joined[:, layout.repa_stream],
    )


def test_isolation_remains_leak_free_across_multiple_blocks() -> None:
    config = tiny_config(
        reg=True,
        attention_mode="asymmetric",
        reg_timestep_tokens=1,
        reg_class_tokens=1,
    )
    model = DDT(config)
    layout = model.encoder_layout
    x = torch.randn(2, layout.total_tokens, config.hidden_size)
    changed = x.clone()
    for token_slice in layout.reg_stream_slices:
        changed[:, token_slice] += torch.randn_like(
            changed[:, token_slice]
        )
    for block in model.encoder_blocks:
        x = block(x, image_attends_reg=False)
        changed = block(changed, image_attends_reg=False)
        assert torch.allclose(
            x[:, layout.image_data],
            changed[:, layout.image_data],
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.allclose(
            x[:, layout.image_timestep],
            changed[:, layout.image_timestep],
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.allclose(
            x[:, layout.image_class],
            changed[:, layout.image_class],
            atol=1e-6,
            rtol=1e-6,
        )


def test_image_data_joins_reg_exactly_after_configured_boundary() -> None:
    config = tiny_config(
        reg=True,
        attention_mode="asymmetric",
        reg_timestep_tokens=1,
        reg_class_tokens=1,
    )
    config.features.reg.join_after_block = 1
    model = DDT(config)
    layout = model.encoder_layout
    original = torch.randn(2, layout.total_tokens, config.hidden_size)
    changed = original.clone()
    for token_slice in layout.reg_stream_slices:
        changed[:, token_slice] += torch.randn_like(
            changed[:, token_slice]
        )

    original = model.encoder_blocks[0](
        original,
        image_attends_reg=False,
    )
    changed = model.encoder_blocks[0](
        changed,
        image_attends_reg=False,
    )
    assert torch.allclose(
        original[:, layout.image_data],
        changed[:, layout.image_data],
        atol=1e-6,
        rtol=1e-6,
    )
    original = model.encoder_blocks[1](
        original,
        image_attends_reg=True,
    )
    changed = model.encoder_blocks[1](
        changed,
        image_attends_reg=True,
    )
    assert not torch.allclose(
        original[:, layout.image_data],
        changed[:, layout.image_data],
    )
    assert torch.allclose(
        original[:, layout.image_timestep],
        changed[:, layout.image_timestep],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        original[:, layout.image_class],
        changed[:, layout.image_class],
        atol=1e-6,
        rtol=1e-6,
    )


def test_full_and_reg_forward_match_and_reg_path_skips_image_modules() -> None:
    config = tiny_config(
        reg=True,
        repa=True,
        repa_stream=True,
        scope="image_and_reg",
        attention_mode="asymmetric",
        reg_timestep_tokens=1,
        reg_class_tokens=1,
    )
    model = DDT(config)
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.numel():
                parameter.normal_(std=0.05, generator=generator)
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
            lambda _module, _args, _output, name=name: calls.append(
                name
            )
        )
        for name, module in (
            ("encoder", model.encoder_patch_projection),
            ("decoder", model.decoder_patch_projection),
            ("image_output", model.final_layer.image_projection),
        )
    ]
    reg = model.forward_reg(
        noisy_reg,
        labels,
        reg_timestep=timestep,
    )
    for handle in handles:
        handle.remove()
    assert isinstance(reg, DDTREGOutput)
    assert calls == []
    assert full.reg is not None and full.reg_early is not None
    assert torch.allclose(full.reg, reg.reg)
    assert torch.allclose(full.reg_early, reg.reg_early)


def test_reg_dropout_overrides_encoder_join_and_decoder_join() -> None:
    config = tiny_config(
        reg=True,
        attention_mode="asymmetric",
        reg_timestep_tokens=1,
        reg_class_tokens=1,
    )
    model = DDT(config)
    flags: list[bool] = []
    handles = [
        block.attention.register_forward_pre_hook(
            lambda _module, args, kwargs: flags.append(
                kwargs["image_attends_reg"]
            ),
            with_kwargs=True,
        )
        for block in (*model.encoder_blocks, *model.decoder_blocks)
    ]
    image, timestep, labels, noisy_reg = inputs(config)
    assert noisy_reg is not None
    model(
        image,
        timestep,
        labels,
        noisy_reg=noisy_reg,
        reg_timestep=timestep,
        reg_dropout=True,
    )
    for handle in handles:
        handle.remove()
    assert flags == [False] * 4


@pytest.mark.parametrize(
    ("reg", "scope", "repa_stream", "expected_repa_tokens"),
    [
        (False, "image", False, 4),
        (False, "image", True, 4),
        (True, "image", False, 4),
        (True, "image_and_reg", False, 8),
        (True, "image_and_reg", True, 8),
    ],
)
def test_output_shapes(
    reg: bool,
    scope: str,
    repa_stream: bool,
    expected_repa_tokens: int,
) -> None:
    config = tiny_config(
        reg=reg,
        repa=True,
        repa_stream=repa_stream,
        scope=scope,
        attention_mode="asymmetric" if reg else "joint",
        reg_timestep_tokens=1 if reg else 0,
        reg_class_tokens=1 if reg else 0,
    )
    model = DDT(config)
    image, timestep, labels, noisy_reg = inputs(config)
    output = model(
        image,
        timestep,
        labels,
        noisy_reg=noisy_reg,
        reg_timestep=timestep if reg else None,
        return_repa=True,
    )
    assert isinstance(output, DDTOutput)
    assert output.image.shape == image.shape
    assert output.repa is not None
    assert output.repa.shape == (
        2,
        expected_repa_tokens,
        config.features.encoder.feature_dim,
    )
    if reg:
        assert output.reg is not None
        assert output.reg.shape == (2, 4, 8)
    else:
        assert output.reg is None


def test_repa_stream_is_injected_only_after_repa_block_when_not_returned() -> None:
    config = tiny_config(repa=True, repa_stream=True)
    model = DDT(config)
    token_counts: list[int] = []
    handles = [
        block.register_forward_pre_hook(
            lambda _module, args: token_counts.append(args[0].shape[1])
        )
        for block in model.encoder_blocks
    ]
    image, timestep, labels, _noisy_reg = inputs(config)
    output = model(
        image,
        timestep,
        labels,
        return_repa=False,
    )
    for handle in handles:
        handle.remove()
    assert token_counts == [
        model.encoder_layout.total_tokens,
        model.repa_encoder_layout.total_tokens,
    ]
    assert model.repa_encoder_layout.repa_stream_tokens == 4
    assert output.repa is None


def test_image_loss_backpropagates_through_live_repa_stream() -> None:
    config = tiny_config(repa=True, repa_stream=True)
    model = DDT(config)
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.05, generator=generator)
    image, timestep, labels, _noisy_reg = inputs(config)
    model(
        image,
        timestep,
        labels,
        return_repa=False,
    ).image.square().mean().backward()
    assert model.repa_projection is not None
    assert model.repa_stream_projection is not None
    repa_gradient = model.repa_projection.layers[0].weight.grad
    stream_gradient = model.repa_stream_projection.weight.grad
    assert repa_gradient is not None and torch.count_nonzero(repa_gradient)
    assert stream_gradient is not None and torch.count_nonzero(stream_gradient)


def test_fused_swiglu_matches_unfused() -> None:
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
    x = torch.randn(2, 7, 16)
    assert torch.allclose(
        unfused(x),
        fused(x),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("fused", [False, True])
def test_reg_lora_is_zero_initialized_and_active(
    fused: bool,
) -> None:
    config = tiny_config(
        reg=True,
        attention_mode="asymmetric",
        reg_timestep_tokens=1,
        reg_class_tokens=1,
    )
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
    image, timestep, labels, noisy_reg = inputs(config)
    assert noisy_reg is not None
    model(
        image,
        timestep,
        labels,
        noisy_reg=noisy_reg,
        reg_timestep=timestep,
    ).image.square().mean().backward()
    assert any(
        adapter.up.weight.grad is not None
        for adapter in adapters
    )


def test_joint_attention_and_gradient_checkpointing_run() -> None:
    config = tiny_config(
        reg=True,
        repa=True,
        repa_stream=True,
        attention_mode="joint",
        reg_timestep_tokens=1,
        reg_class_tokens=1,
    )
    config.gradient_checkpointing = True
    model = DDT(config)
    model.train()
    image, timestep, labels, noisy_reg = inputs(config)
    image.requires_grad_(True)
    assert noisy_reg is not None
    output = model(
        image,
        timestep,
        labels,
        noisy_reg=noisy_reg,
        reg_timestep=timestep,
    )
    output.image.square().mean().backward()
    assert image.grad is not None
    with pytest.raises(ValueError, match="asymmetric"):
        model.forward_reg(
            noisy_reg,
            labels,
            reg_timestep=timestep,
        )


def test_public_api_is_intentionally_small() -> None:
    assert jit_reg.DDT is DDT
    assert jit_reg.DDTOutput is DDTOutput
    assert jit_reg.DDTREGOutput is DDTREGOutput
    assert "DDT" in jit_reg.__all__
    assert not hasattr(jit_reg, "DDTAttention")
    assert isinstance(DDT(tiny_config()).encoder_blocks[0], DDTEncoderBlock)
    assert isinstance(DDT(tiny_config()).decoder_blocks[0], DDTDecoderBlock)
