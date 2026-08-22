from __future__ import annotations

import torch
from PIL import Image

from cascade_dit.semantic_pca import save_semantic_pca_preview, semantic_pca_rgb


def test_per_image_semantic_pca_is_deterministic_and_bounded() -> None:
    features = torch.randn(2, 12, 4, 4, generator=torch.Generator().manual_seed(7))
    first = semantic_pca_rgb(features, 32)
    second = semantic_pca_rgb(features, 32)
    assert first.shape == (2, 3, 32, 32)
    assert torch.equal(first, second)
    assert torch.all((first >= 0) & (first <= 1))


def test_semantic_pca_preview_is_saved_as_labeled_grid(tmp_path) -> None:
    features = torch.randn(2, 8, 4, 4, generator=torch.Generator().manual_seed(9))
    target = tmp_path / "semantic-pca-preview.png"
    save_semantic_pca_preview(features, torch.tensor([4, 8]), target, 24)
    assert target.is_file()
    with Image.open(target) as image:
        assert image.mode == "RGB"
        assert image.size == (96, 176)
