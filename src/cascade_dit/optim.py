from __future__ import annotations

import math

import torch
from torch import nn

from .config import OptimizerConfig


def build_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
    *,
    include_frozen: bool = False,
) -> torch.optim.AdamW:
    parameters = [
        parameter
        for parameter in model.parameters()
        if include_frozen or parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    adamw = config.adamw
    return torch.optim.AdamW(
        parameters,
        lr=adamw.lr,
        betas=(adamw.betas[0], adamw.betas[1]),
        weight_decay=adamw.weight_decay,
    )


def build_joint_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
) -> torch.optim.AdamW:
    """Build stable flow/adapter/decoder groups, including the frozen decoder."""
    required = (
        "flow_parameter_names",
        "pixel_adapter_parameter_names",
        "pixel_decoder_parameter_names",
    )
    if not all(hasattr(model, name) for name in required):
        raise TypeError("joint optimizer requires a DDT parameter partition")
    named = dict(model.named_parameters())
    partitions = (
        ("flow", model.flow_parameter_names()),  # type: ignore[attr-defined]
        ("pixel_adapter", model.pixel_adapter_parameter_names()),  # type: ignore[attr-defined]
        ("pixel_decoder", model.pixel_decoder_parameter_names()),  # type: ignore[attr-defined]
    )
    groups: list[dict[str, object]] = []
    assigned: set[str] = set()
    for group_name, names in partitions:
        if assigned & names:
            raise ValueError("joint optimizer parameter partitions overlap")
        parameters = [named[name] for name in sorted(names)]
        if not parameters:
            raise ValueError(f"joint optimizer {group_name} partition is empty")
        assigned.update(names)
        groups.append({"params": parameters, "name": group_name})
    if assigned != set(named):
        raise ValueError("joint optimizer parameter partitions are incomplete")
    adamw = config.adamw
    return torch.optim.AdamW(
        groups,
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
