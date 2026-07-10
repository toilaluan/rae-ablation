from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def depth_source_layers(layer: int, stride: int = 4) -> tuple[int, ...]:
    """Return the causal depth sources, with the current layer listed last."""
    if layer < 0 or stride < 1:
        raise ValueError("layer must be non-negative and stride must be positive")
    return (*range(0, layer, stride), layer)


class DepthAttention(nn.Module):
    """Parameter-free, same-position attention over recursively mixed depth values."""

    def __init__(self, layer: int, depth: int, stride: int = 4, enabled: bool = True) -> None:
        super().__init__()
        self.layer = layer
        self.depth = depth
        self.stride = stride
        self.enabled = enabled
        self.max_sources = math.ceil(depth / stride)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_bank: Tensor,
        value_bank: Tensor,
        bank_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Mix values and functionally update fixed-size ``[B,S,H,N,D]`` banks."""
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError("query, key, and value shapes must match")
        if key_bank.shape != value_bank.shape:
            raise ValueError("depth key and value bank shapes must match")
        if key_bank.shape[1] != self.max_sources or bank_mask.shape != (self.max_sources,):
            raise ValueError("depth bank does not have the configured fixed size")

        if self.enabled:
            # Bank entries are [B,S,H,N,D], while current q/k/v are [B,H,N,D].
            logits_old = torch.einsum("bhnd,bshnd->bhns", query.float(), key_bank.float())
            logits_now = (query.float() * key.float()).sum(dim=-1, keepdim=True)
            logits = torch.cat((logits_old, logits_now), dim=-1) / math.sqrt(query.shape[-1])
            valid = torch.cat(
                (bank_mask.to(device=query.device, dtype=torch.bool), bank_mask.new_ones(1).bool())
            )
            logits = logits.masked_fill(~valid.view(1, 1, 1, -1), -torch.inf)
            weights = logits.softmax(dim=-1, dtype=torch.float32).to(dtype=value.dtype)
            candidates = torch.cat((value_bank, value.unsqueeze(1)), dim=1)
            mixed = torch.einsum("bhns,bshnd->bhnd", weights, candidates)
        else:
            mixed = value

        if self.layer % self.stride == 0:
            slot = self.layer // self.stride
            next_keys = key_bank.clone()
            next_values = value_bank.clone()
            next_mask = bank_mask.clone()
            next_keys[:, slot] = key
            next_values[:, slot] = mixed
            next_mask[slot] = True
        else:
            next_keys, next_values, next_mask = key_bank, value_bank, bank_mask
        return mixed, next_keys, next_values, next_mask
