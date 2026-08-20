from __future__ import annotations

import torch

from jit_reg.config import DINO_CHANNELS, DINO_GRID_SIZE, IMAGE_SIZE, PixelDecoderConfig
from jit_reg.pixel_decoder import (
    PatchDiscriminator,
    TransformerPixelDecoder,
    discriminator_hinge_loss,
    generator_hinge_loss,
)


def test_transformer_pixel_decoder_shape_and_gradients() -> None:
    config = PixelDecoderConfig(width=32, depth=1, num_heads=4, mlp_ratio=2)
    decoder = TransformerPixelDecoder(config)
    tokens = torch.randn(1, DINO_CHANNELS, DINO_GRID_SIZE, DINO_GRID_SIZE)
    output = decoder(tokens)
    assert output.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.all((output >= 0) & (output <= 1))
    output.mean().backward()
    assert decoder.input_projection.weight.grad is not None


def test_patchgan_hinge_losses_are_finite() -> None:
    discriminator = PatchDiscriminator(base_channels=4)
    real = discriminator(torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    fake = discriminator(torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert torch.isfinite(discriminator_hinge_loss(real, fake))
    assert torch.isfinite(generator_hinge_loss(fake))
