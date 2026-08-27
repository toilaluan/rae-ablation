from __future__ import annotations

from cascade_dit.config import ExperimentConfig
from cascade_dit.flops import ddt_forward_flops, model_flops_per_step


def test_layer_attention_fusion_and_sparse_updates_are_counted() -> None:
    concatenate = ExperimentConfig()
    concatenate_pixels = ddt_forward_flops(concatenate) - ddt_forward_flops(
        concatenate, decode_pixels=False
    )

    layer_attention = ExperimentConfig()
    layer_attention.model.pixel_decoder.fusion.kind = "layer_attention"
    layer_pixels = ddt_forward_flops(layer_attention) - ddt_forward_flops(
        layer_attention, decode_pixels=False
    )
    assert 0 < layer_pixels < concatenate_pixels

    semantic_step = model_flops_per_step(
        layer_attention, decoder_active=False, decoder_updates=0
    )
    one_decoder = model_flops_per_step(
        layer_attention, decoder_active=True, decoder_updates=1
    )
    three_decoders = model_flops_per_step(
        layer_attention, decoder_active=True, decoder_updates=3
    )
    decoder_step = layer_attention.optimizer.global_batch_size * 3 * layer_pixels
    assert one_decoder - semantic_step == decoder_step
    assert three_decoders - semantic_step == 3 * decoder_step


def test_flop_count_rejects_inactive_decoder_updates() -> None:
    config = ExperimentConfig()
    try:
        model_flops_per_step(config, decoder_active=False, decoder_updates=1)
    except ValueError as error:
        assert "inconsistent" in str(error)
    else:
        raise AssertionError("inactive decoder updates must be rejected")
