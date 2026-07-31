from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from .config import OptimizerConfig

TORCH_ADAMW = "torch.optim.AdamW"
TORCH_MUON = "torch.optim.Muon"


def _muon_eligible(name: str, parameter: nn.Parameter) -> bool:
    return parameter.ndim == 2 and "class_embed" not in name


class CompositeOptimizer:
    def __init__(
        self,
        optimizers: Iterable[torch.optim.Optimizer],
        implementations: Iterable[str],
    ) -> None:
        self.optimizers = list(optimizers)
        self.implementations = list(implementations)
        if len(self.optimizers) != len(self.implementations):
            raise ValueError("optimizer implementations must identify every optimizer")
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "implementations": list(self.implementations),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        states = state["optimizers"]
        if len(states) != len(self.optimizers):
            raise ValueError("optimizer count changed across resume")
        saved_implementations = state.get("implementations")
        if (
            saved_implementations is not None
            and list(saved_implementations) != self.implementations
        ):
            raise ValueError("optimizer backend changed across resume")
        for optimizer, optimizer_state in zip(self.optimizers, states, strict=True):
            optimizer.load_state_dict(optimizer_state)


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> CompositeOptimizer:
    named = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if config.kind == "adamw":
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in named],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
        return CompositeOptimizer([optimizer], [TORCH_ADAMW])

    muon_parameters = [parameter for name, parameter in named if _muon_eligible(name, parameter)]
    adam_parameters = [
        parameter for name, parameter in named if not _muon_eligible(name, parameter)
    ]
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("optimizer.kind=muon requires torch.optim.Muon (PyTorch 2.12+)")
    muon = torch.optim.Muon(
        muon_parameters,
        lr=config.lr,
        weight_decay=config.weight_decay,
        adjust_lr_fn="match_rms_adamw",
    )
    adam = torch.optim.AdamW(
        adam_parameters,
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    return CompositeOptimizer(
        [muon, adam],
        [TORCH_MUON, TORCH_ADAMW],
    )


def learning_rate_at_epoch(epoch: float, base_lr: float = 2e-4, min_lr: float = 2e-5) -> float:
    if epoch <= 25:
        return base_lr
    if epoch < 50:
        return base_lr + (min_lr - base_lr) * ((epoch - 25) / 25)
    return min_lr


def set_learning_rate(optimizer: CompositeOptimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value
