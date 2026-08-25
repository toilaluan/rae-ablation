from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("invalid LoRA rank, alpha, or dropout")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout)
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, **factory))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, **factory))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, values: Tensor) -> Tensor:
        update = (self.dropout(values) @ self.lora_a.T) @ self.lora_b.T
        return self.base(values) + update * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        self.base.weight.add_((self.lora_b @ self.lora_a).to(self.base.weight), alpha=self.scaling)
        return self.base


_ATTENTION_LINEAR_NAMES = ("qkv", "proj")
_MLP_LINEAR_NAMES = ("w1", "w2", "w3")


def attach_lora(
    blocks: nn.ModuleList,
    block_indices: Iterable[int],
    *,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> None:
    indices = tuple(block_indices)
    if len(indices) != len(set(indices)) or any(index < 0 or index >= len(blocks) for index in indices):
        raise ValueError("LoRA block indices must be unique and within the model depth")
    for index in indices:
        block = blocks[index]
        for module, names in ((block.attn, _ATTENTION_LINEAR_NAMES), (block.mlp, _MLP_LINEAR_NAMES)):
            for name in names:
                layer = getattr(module, name)
                if isinstance(layer, LoRALinear):
                    raise ValueError(f"block {index} already has LoRA adapters")
                if not isinstance(layer, nn.Linear):
                    raise TypeError(f"LoRA target block {index}.{name} is not linear")
                setattr(module, name, LoRALinear(layer, rank, alpha, dropout))


def merge_lora(blocks: nn.ModuleList, block_indices: Iterable[int] | None = None) -> int:
    indices = tuple(range(len(blocks))) if block_indices is None else tuple(block_indices)
    merged = 0
    for index in indices:
        if index < 0 or index >= len(blocks):
            raise ValueError("LoRA merge block index is outside model depth")
        block = blocks[index]
        for module, names in ((block.attn, _ATTENTION_LINEAR_NAMES), (block.mlp, _MLP_LINEAR_NAMES)):
            for name in names:
                layer = getattr(module, name)
                if isinstance(layer, LoRALinear):
                    setattr(module, name, layer.merge())
                    merged += 1
    return merged


def has_lora(module: nn.Module) -> bool:
    return any(isinstance(child, LoRALinear) for child in module.modules())


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for child in module.modules()
        if isinstance(child, LoRALinear)
        for parameter in (child.lora_a, child.lora_b)
    ]
