from __future__ import annotations

import math
from collections.abc import Mapping

from torch import Tensor, nn

from ..config import LoRAConfig


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.adapter_a = nn.Linear(base.in_features, config.rank, bias=False)
        self.adapter_b = nn.Linear(config.rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.scaling = config.alpha / config.rank
        nn.init.kaiming_uniform_(self.adapter_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_b.weight)

    def forward(self, values: Tensor) -> Tensor:
        residual = self.adapter_b(self.adapter_a(self.dropout(values)))
        return self.base(values) + residual * self.scaling


def _target_group(name: str) -> str:
    if ".attention." in name:
        return "attention"
    if (name.startswith("encoder_blocks.") or name.startswith("decoder_blocks.")) and ".mlp." in name:
        return "feedforward"
    if name.startswith("timestep_embedder.mlp.") or ".adaln." in name:
        return "conditioning"
    return "projections"


def apply_lora(model: nn.Module, config: LoRAConfig) -> set[str]:
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("LoRA is already active")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = set(config.target_groups)
    targets = [
        name
        for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear) and _target_group(name) in selected
    ]
    if not targets:
        raise ValueError("LoRA target groups did not match any linear layers")
    for name in targets:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, child_name)
        assert isinstance(base, nn.Linear)
        setattr(parent, child_name, LoRALinear(base, config))
    return {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def has_lora(model: nn.Module) -> bool:
    return any(isinstance(module, LoRALinear) for module in model.modules())


def merged_state_dict(
    model: nn.Module,
    state: Mapping[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    source = model.state_dict() if state is None else state
    lora_modules = {
        name: module
        for name, module in model.named_modules()
        if name and isinstance(module, LoRALinear)
    }
    if not lora_modules:
        return {name: value.detach().contiguous() for name, value in source.items()}
    prefixes = tuple(f"{name}." for name in lora_modules)
    result = {
        name: value.detach().contiguous()
        for name, value in source.items()
        if not name.startswith(prefixes)
    }
    for name, module in lora_modules.items():
        base = source[f"{name}.base.weight"]
        adapter_a = source[f"{name}.adapter_a.weight"]
        adapter_b = source[f"{name}.adapter_b.weight"]
        delta = adapter_b.float() @ adapter_a.float()
        result[f"{name}.weight"] = (
            base.float() + delta * module.scaling
        ).to(base).detach().contiguous()
        bias_key = f"{name}.base.bias"
        if bias_key in source:
            result[f"{name}.bias"] = source[bias_key].detach().contiguous()
    return result


def expand_merged_state_dict(
    model: nn.Module,
    merged: Mapping[str, Tensor],
    *,
    template: Mapping[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    if not has_lora(model):
        return {name: value.detach().contiguous() for name, value in merged.items()}
    expected = set(merged_state_dict(model))
    actual = set(merged)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"checkpoint model keys are incompatible; missing={missing}, unexpected={unexpected}"
        )
    expanded = {
        name: value.detach().clone()
        for name, value in (model.state_dict() if template is None else template).items()
    }
    for name, module in model.named_modules():
        if not name or not isinstance(module, LoRALinear):
            continue
        weight = merged[f"{name}.weight"]
        if weight.shape != expanded[f"{name}.base.weight"].shape:
            raise ValueError(f"checkpoint tensor shape mismatch for {name}.weight")
        expanded[f"{name}.base.weight"] = weight.detach().clone()
        bias_key = f"{name}.bias"
        if bias_key in merged:
            expanded[f"{name}.base.bias"] = merged[bias_key].detach().clone()
        expanded[f"{name}.adapter_b.weight"].zero_()
    lora_prefixes = tuple(
        f"{name}."
        for name, module in model.named_modules()
        if name and isinstance(module, LoRALinear)
    )
    for name, value in merged.items():
        if name.startswith(lora_prefixes):
            continue
        if name not in expanded or expanded[name].shape != value.shape:
            raise ValueError(f"checkpoint tensor shape mismatch for {name}")
        expanded[name] = value.detach().clone()
    return expanded


def load_merged_state_dict(model: nn.Module, state: Mapping[str, Tensor]) -> None:
    if has_lora(model):
        model.load_state_dict(expand_merged_state_dict(model, state))
    else:
        model.load_state_dict(state)
