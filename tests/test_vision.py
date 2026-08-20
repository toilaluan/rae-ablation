from __future__ import annotations

import torch
from torch import nn

from jit_reg.config import VisionConfig
from jit_reg.vision import DINOEndpointEncoder


class Embeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embeddings = nn.Conv2d(3, 768, 16, 16, bias=False)

    def forward(self, pixels: torch.Tensor, bool_masked_pos: object = None) -> torch.Tensor:
        del bool_masked_pos
        batch = pixels.shape[0]
        return torch.zeros(batch, 261, 768, device=pixels.device, dtype=pixels.dtype)


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


def test_dino_extracts_both_normalized_patch_endpoints_in_one_pass() -> None:
    model = FakeDINO()
    encoder = DINOEndpointEncoder(model, VisionConfig())
    target, semantic = encoder.encode_pair(torch.zeros(1, 3, 256, 256, dtype=torch.uint8))
    assert target.shape == semantic.shape == (1, 768, 16, 16)
    assert torch.all(target == sum(range(1, 8)))
    assert torch.all(semantic == sum(range(1, 13)))
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    encoder.train()
    assert encoder.training is False


def test_target_only_encoder_stops_at_target_layer() -> None:
    model = FakeDINO()
    encoder = DINOEndpointEncoder(model, VisionConfig(), include_semantic=False)
    target = encoder.encode_target(torch.zeros(1, 3, 256, 256))
    assert torch.all(target == sum(range(1, 8)))
