from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from jit_reg.config import REGConfig
from jit_reg.reg import (
    DINOv3FeatureExtractor,
    prepare_reg_targets,
    resize_spatial_features,
)


class _FakeDINO(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.seen: torch.Tensor | None = None

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        self.seen = images.detach().clone()
        tokens = torch.zeros(images.shape[0], 261, 8, device=images.device)
        tokens[:, 0] = 3
        tokens[:, 1:5] = 99
        values = torch.arange(256, device=images.device).float()
        tokens[:, 5:, 0] = values
        tokens[:, 5:, 1] = 1
        return tokens


def test_dinov3_preprocessing_freezing_and_register_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = _FakeDINO()
    fake_timm = SimpleNamespace(
        create_model=lambda name, pretrained: encoder,
        data=SimpleNamespace(
            resolve_model_data_config=lambda model: {
                "input_size": (3, 256, 256),
                "mean": (0.5, 0.5, 0.5),
                "std": (0.25, 0.25, 0.25),
            }
        ),
    )
    monkeypatch.setitem(sys.modules, "timm", fake_timm)
    extractor = DINOv3FeatureExtractor(REGConfig(feature_dim=8))
    cls, spatial = extractor.extract(torch.ones(2, 3, 128, 128))
    assert cls.shape == (2, 8) and torch.all(cls == 3)
    assert spatial.shape == (2, 256, 8)
    assert spatial[0, 0, 0] == 0 and spatial[0, -1, 0] == 255
    assert encoder.seen is not None
    assert encoder.seen.shape == (2, 3, 256, 256)
    torch.testing.assert_close(encoder.seen, torch.full_like(encoder.seen, 2))
    assert not any(parameter.requires_grad for parameter in extractor.parameters())
    extractor.train()
    assert not extractor.training and not encoder.training


@pytest.mark.parametrize("target_grid", [16, 8, 4])
def test_repa_targets_resize_to_any_jit_grid(target_grid: int) -> None:
    spatial = torch.arange(256 * 8, dtype=torch.float32).reshape(1, 256, 8).add(1)
    cls = torch.arange(8, dtype=torch.float32).reshape(1, 8).add(1)
    targets = prepare_reg_targets(cls, spatial, target_grid)
    assert torch.equal(targets.cls, cls)
    assert targets.spatial.shape == (1, target_grid**2, 8)
    assert targets.sequence.shape == (1, 1 + target_grid**2, 8)
    torch.testing.assert_close(targets.spatial.norm(dim=-1), torch.ones(1, target_grid**2))
    if target_grid == 16:
        torch.testing.assert_close(targets.spatial, F.normalize(spatial, dim=-1))
    else:
        expected = F.interpolate(
            spatial.transpose(1, 2).reshape(1, 8, 16, 16),
            size=(target_grid, target_grid),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).flatten(2).transpose(1, 2)
        torch.testing.assert_close(targets.spatial, F.normalize(expected, dim=-1))


def test_spatial_feature_order_requires_a_square_grid() -> None:
    with pytest.raises(ValueError, match="square"):
        resize_spatial_features(torch.randn(1, 12, 8), 2)
