from __future__ import annotations

import math

import torch
from torch import nn

from .config import OptimizerConfig


def build_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
) -> torch.optim.AdamW:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    adamw = config.adamw
    return torch.optim.AdamW(
        parameters,
        lr=adamw.lr,
        betas=(adamw.betas[0], adamw.betas[1]),
        weight_decay=adamw.weight_decay,
    )


def build_training_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    kind: str = "cosine",
    steps_per_epoch: int,
    total_epochs: int,
    warmup_epochs: int = 0,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    if kind not in {"constant", "cosine"}:
        raise ValueError("training scheduler kind must be constant or cosine")
    if steps_per_epoch < 1 or total_epochs < 1 or warmup_epochs < 0:
        raise ValueError("invalid training scheduler counts")
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = min(warmup_epochs * steps_per_epoch, total_steps)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1 / warmup_steps)
        if kind == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
