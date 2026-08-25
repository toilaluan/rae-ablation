from __future__ import annotations

import torch
from torch import nn

from cascade_dit.config import EncoderConfig
from cascade_dit.vision import DINOMultiLayerEncoder


class Embeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embeddings = nn.Conv2d(3, 768, 16, 16, bias=False)
        self.calls = 0

    def forward(self, pixels: torch.Tensor, bool_masked_pos: object = None) -> torch.Tensor:
        del bool_masked_pos
        self.calls += 1
        return torch.zeros(pixels.shape[0], 69, 768, dtype=pixels.dtype)


class Block(nn.Module):
    def forward(self, states: torch.Tensor, position_embeddings: object = None) -> torch.Tensor:
        del position_embeddings
        return states + 1


class FakeDINO(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = Embeddings()
        self.rope_embeddings = lambda pixels: pixels.new_zeros(1)
        self.layer = nn.ModuleList(Block() for _ in range(12))
        self.norm = lambda values: values * 2


def test_dino_layers_are_one_indexed_final_normed_and_extracted_once() -> None:
    model = FakeDINO()
    encoder = DINOMultiLayerEncoder(model, EncoderConfig())
    outputs = encoder(torch.zeros(2, 3, 128, 128, dtype=torch.uint8))
    assert list(outputs) == [12, 8, 4]
    assert all(value.shape == (2, 64, 768) for value in outputs.values())
    assert outputs[12].unique().item() == 24
    assert outputs[8].unique().item() == 16
    assert outputs[4].unique().item() == 8
    assert model.embeddings.calls == 1
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    encoder.train()
    assert encoder.training is False
