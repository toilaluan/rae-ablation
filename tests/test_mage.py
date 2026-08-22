from __future__ import annotations

import torch
from torch import nn

from cascade_dit.mage import MageVAECodec


class FakeMage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.sample_posterior = True

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.avg_pool2d(images.mean(1, keepdim=True), 16)
        return pooled.expand(-1, 128, -1, -1)

    def decode(self, image_latents: torch.Tensor) -> torch.Tensor:
        values = image_latents[:, :1].expand(-1, 3, -1, -1)
        return torch.nn.functional.interpolate(values, scale_factor=16, mode="nearest")


def test_mage_codec_is_deterministic_native_shape_and_decodes_to_unit_range() -> None:
    codec = MageVAECodec(FakeMage())
    images = torch.rand(2, 3, 256, 256).mul(2).sub(1)
    first = codec.encode(images)
    second = codec.encode(images)
    assert first.shape == (2, 128, 16, 16)
    assert torch.equal(first, second)
    assert codec.model.sample_posterior is False
    decoded = codec.decode(first)
    assert decoded.shape == (2, 3, 256, 256)
    assert torch.all((decoded >= 0) & (decoded <= 1))
    assert all(not parameter.requires_grad for parameter in codec.parameters())


def test_dataset_pixel_helper_matches_native_input() -> None:
    codec = MageVAECodec(FakeMage())
    pixels = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8)
    expected = codec.encode(pixels.float().div(255).mul(2).sub(1))
    assert torch.equal(codec.encode_pixels(pixels), expected)
