from __future__ import annotations

import torch
from torch import nn

from cascade_dit.config import SemanticTokenConfig
from cascade_dit.semantic_statistics import SemanticStatistics, SemanticTokenNormalizer
from cascade_dit.vision import DINOSemanticEncoder


class Embeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embeddings = nn.Conv2d(3, 768, 16, 16, bias=False)

    def forward(self, pixels: torch.Tensor, bool_masked_pos: object = None) -> torch.Tensor:
        del bool_masked_pos
        patches = (pixels.shape[-1] // 16) ** 2
        return torch.zeros(pixels.shape[0], 5 + patches, 768, dtype=pixels.dtype)


class Block(nn.Module):
    def forward(
        self, states: torch.Tensor, position_embeddings: object = None
    ) -> torch.Tensor:
        del position_embeddings
        return states + 1


class FakeDINO(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = Embeddings()
        self.rope_embeddings = lambda pixels: pixels.new_zeros(1)
        self.layer = nn.ModuleList(Block() for _ in range(12))
        self.norm = nn.Identity()


def test_dino_extracts_normalized_unpooled_tokens_from_128px_view() -> None:
    statistics = SemanticStatistics(
        mean=torch.full((768,), 4.0),
        std=torch.full((768,), 2.0),
        count=2,
        semantic_encoder_fingerprint="encoder",
        data_fingerprint="data",
    )
    encoder = DINOSemanticEncoder(
        FakeDINO(),
        SemanticTokenConfig(),
        normalizer=SemanticTokenNormalizer(statistics),
    )
    tokens = encoder.encode(torch.zeros(1, 3, 256, 256, dtype=torch.uint8))
    assert tokens.shape == (1, 768, 8, 8)
    assert torch.all(tokens == 4)
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    encoder.train()
    assert encoder.training is False
