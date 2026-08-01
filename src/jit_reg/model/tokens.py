from __future__ import annotations

import torch
from torch import Tensor


def classifier_free_labels(
    labels: Tensor,
    num_classes: int,
    probability: float,
    generator: torch.Generator,
) -> Tensor:
    if labels.ndim != 1:
        raise ValueError("labels must have shape [B]")
    if not 0 <= probability < 1:
        raise ValueError("label dropout probability must be in [0, 1)")
    if probability == 0:
        return labels
    dropped = torch.rand(
        labels.shape,
        device=labels.device,
        generator=generator,
    ) < probability
    return torch.where(dropped, torch.full_like(labels, num_classes), labels)
