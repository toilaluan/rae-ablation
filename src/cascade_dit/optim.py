from __future__ import annotations

import math

import torch
from torch import nn

from .config import OptimizerConfig


def build_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
    *,
    transitioned: bool = False,
) -> torch.optim.AdamW:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    lr = config.transition_lr if transitioned else config.initial_lr
    return torch.optim.AdamW(
        parameters,
        lr=lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )


def build_stage_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    steps_per_epoch: int,
    stage_epochs: int,
    warmup_epochs: int = 0,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    if steps_per_epoch < 1 or stage_epochs < 1 or warmup_epochs < 0:
        raise ValueError("invalid stage scheduler counts")
    total_steps = steps_per_epoch * stage_epochs
    warmup_steps = min(warmup_epochs * steps_per_epoch, total_steps)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1 / warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
