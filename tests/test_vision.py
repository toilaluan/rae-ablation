from __future__ import annotations

import torch
from torch import nn

from jit_reg.config import IMAGE_SIZE, VisionConfig
from jit_reg.latent_statistics import EndpointLatentNormalizer, LatentStatistics
from jit_reg.vision import DINOEndpointEncoder


class Embeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embeddings = nn.Conv2d(3, 768, 16, 16, bias=False)

    def forward(self, pixels: torch.Tensor, bool_masked_pos: object = None) -> torch.Tensor:
        del bool_masked_pos
        batch = pixels.shape[0]
        patches = (pixels.shape[-1] // 16) ** 2
        return torch.zeros(batch, 5 + patches, 768, device=pixels.device, dtype=pixels.dtype)


class Block(nn.Module):
    def __init__(self, value: int) -> None:
        super().__init__()
        self.value = value

    def forward(
        self,
        states: torch.Tensor,
        attention_mask: object = None,
        position_embeddings: object = None,
    ) -> torch.Tensor:
        del attention_mask, position_embeddings
        return states + self.value


class FakeDINO(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = Embeddings()
        self.rope_embeddings = lambda pixels: (pixels.new_zeros(1), pixels.new_zeros(1))
        self.layer = nn.ModuleList(Block(index) for index in range(1, 13))
        self.norm = nn.Identity()


def _statistics() -> LatentStatistics:
    return LatentStatistics(
        target_mean=torch.ones(768),
        target_std=torch.full((768,), 2.0),
        semantic_mean=torch.full((768,), 10.0),
        semantic_std=torch.full((768,), 4.0),
        pixel_mean=torch.zeros(768),
        pixel_std=torch.ones(768),
        patch_count=2,
        vision_fingerprint="vision",
        data_fingerprint="data",
    )


def test_dino_preserves_raw_block2_patches_and_final_semantics() -> None:
    encoder = DINOEndpointEncoder(FakeDINO(), VisionConfig())
    endpoints = encoder.encode_pair(
        torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8)
    )
    assert endpoints.target.shape == endpoints.semantic.shape == (1, 768, 8, 8)
    assert torch.all(endpoints.target == sum(range(1, 3)))
    assert torch.all(endpoints.semantic == sum(range(1, 13)))
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    encoder.train()
    assert encoder.training is False


def test_target_only_encoder_stops_at_raw_target_layer() -> None:
    encoder = DINOEndpointEncoder(FakeDINO(), VisionConfig(), include_semantic=False)
    target = encoder.encode_target(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert torch.all(target == sum(range(1, 3)))


def test_dino_applies_separate_normalizers() -> None:
    normalizer = EndpointLatentNormalizer(_statistics())
    encoder = DINOEndpointEncoder(
        FakeDINO(),
        VisionConfig(),
        normalizer=normalizer,
    )
    endpoints = encoder.encode_pair(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert torch.all(endpoints.target == (sum(range(1, 3)) - 1) / 2)
    assert torch.all(endpoints.semantic == (sum(range(1, 13)) - 10) / 4)


def test_target_only_encoder_duplicates_target_only_in_semantic_slot() -> None:
    encoder = DINOEndpointEncoder(
        FakeDINO(),
        VisionConfig(),
        include_semantic=False,
        normalizer=EndpointLatentNormalizer(_statistics()),
    )
    endpoints = encoder(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert torch.equal(endpoints.target, endpoints.semantic)
