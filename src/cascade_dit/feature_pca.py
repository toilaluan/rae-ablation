from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .sample_runtime import save_preview


def feature_pca_rgb(features: Tensor, image_size: int) -> Tensor:
    """Project each spatial feature map independently onto three PCA components."""
    if (
        features.ndim != 4
        or features.shape[0] < 1
        or features.shape[1] < 3
        or features.shape[2] * features.shape[3] < 3
    ):
        raise ValueError(
            "features must have shape [B,C,H,W] with at least 3 channels and tokens"
        )
    if image_size < 1:
        raise ValueError("PCA preview image size must be positive")
    batch, _channels, height, width = features.shape
    observations = features.detach().float().cpu().permute(0, 2, 3, 1)
    rendered: list[Tensor] = []
    for values in observations:
        matrix = values.reshape(height * width, -1)
        centered = matrix - matrix.mean(dim=0, keepdim=True)
        left, singular_values, right_h = torch.linalg.svd(centered, full_matrices=False)
        scores = left[:, :3] * singular_values[:3]
        loadings = right_h[:3]
        anchor_indices = loadings.abs().argmax(dim=1)
        anchors = loadings[torch.arange(3), anchor_indices]
        signs = torch.where(anchors < 0, -torch.ones_like(anchors), torch.ones_like(anchors))
        scores = scores * signs
        minimum = scores.amin(dim=0, keepdim=True)
        spans = scores.amax(dim=0, keepdim=True) - minimum
        normalized = torch.where(
            spans > torch.finfo(scores.dtype).eps,
            (scores - minimum) / spans.clamp_min(torch.finfo(scores.dtype).eps),
            torch.full_like(scores, 0.5),
        )
        rendered.append(normalized.reshape(height, width, 3).permute(2, 0, 1))
    rgb = torch.stack(rendered).reshape(batch, 3, height, width)
    return F.interpolate(rgb, size=(image_size, image_size), mode="bilinear", align_corners=False)


def save_feature_pca_preview(
    features: Tensor,
    labels: Tensor,
    target: str | Path,
    image_size: int,
) -> None:
    if labels.shape != (features.shape[0],):
        raise ValueError("PCA preview labels must have shape [B]")
    images = feature_pca_rgb(features, image_size).mul(255).round().byte()
    arrays = images.permute(0, 2, 3, 1).contiguous().numpy()
    save_preview(np.asarray(arrays), labels.detach().cpu().numpy(), Path(target))
