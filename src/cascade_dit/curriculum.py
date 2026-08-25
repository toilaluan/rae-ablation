from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .config import CurriculumConfig
from .model import LightningDiT, LoRALinear, has_lora

if TYPE_CHECKING:
    from .checkpoint import EMA


@dataclass(frozen=True)
class StagePosition:
    index: int
    name: str
    first_epoch: int
    final_epoch: int
    epoch_within_stage: int


def stage_at_epoch(epoch: int, config: CurriculumConfig) -> StagePosition:
    if epoch < 0 or epoch >= sum(config.epochs):
        raise ValueError("epoch is outside the curriculum")
    beginning = 0
    for index, (name, duration) in enumerate(zip(config.stages, config.epochs, strict=True)):
        end = beginning + duration
        if epoch < end:
            return StagePosition(index, name, beginning, end - 1, epoch - beginning)
        beginning = end
    raise AssertionError("unreachable curriculum position")


def _set_module_trainable(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def configure_trainability(
    model: LightningDiT,
    stage_index: int,
    config: CurriculumConfig,
    *,
    pixel_baseline: bool = False,
) -> set[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if pixel_baseline:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif stage_index == 0:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        if model.pixel_detail is not None:
            _set_module_trainable(model.pixel_detail, False)
    else:
        for module in (model.input_projection, model.final_layer, model.token_detail):
            _set_module_trainable(module, True)
        if stage_index == 3 and model.pixel_detail is not None:
            _set_module_trainable(model.pixel_detail, True)
        if config.later_stage_full_finetune:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        else:
            for index in config.fully_train_blocks:
                _set_module_trainable(model.blocks[index], True)
            for index in config.lora_blocks:
                for module in model.blocks[index].modules():
                    if isinstance(module, LoRALinear):
                        module.lora_a.requires_grad_(True)
                        module.lora_b.requires_grad_(True)
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def transition_to_stage(
    model: LightningDiT,
    ema: EMA | LightningDiT,
    target_index: int,
    projectors: dict[str, Tensor],
    config: CurriculumConfig,
) -> bool:
    """Merge old adapters, rotate stages, and install fresh adapters exactly once."""
    if target_index <= 0 or target_index >= len(config.stages):
        raise ValueError("transition target must be a later curriculum stage")
    target = config.stages[target_index]
    if model.stage == target:
        return False
    expected_source = config.stages[target_index - 1]
    if model.stage != expected_source:
        raise ValueError(f"cannot transition directly from {model.stage} to {target}")

    ema_model = ema if isinstance(ema, LightningDiT) else ema.to_model(model)
    model.merge_lora()
    ema_model.merge_lora()
    model.set_stage(target, projectors[target])
    ema_model.set_stage(target, projectors[target])
    if not config.later_stage_full_finetune:
        model.attach_lora(
            config.lora_blocks,
            config.lora_rank,
            config.lora_alpha,
            config.lora_dropout,
        )
        ema_model.attach_lora(
            config.lora_blocks,
            config.lora_rank,
            config.lora_alpha,
            config.lora_dropout,
        )
    model_parameters = dict(model.named_parameters())
    for name, parameter in ema_model.named_parameters():
        if ".lora_" in name or name.startswith("pixel_detail."):
            parameter.data.copy_(model_parameters[name].data.to(parameter))
    configure_trainability(model, target_index, config)
    if not isinstance(ema, LightningDiT):
        ema.replace_from_model(ema_model)
    return True


def export_progressive_model(model: LightningDiT) -> LightningDiT:
    model.merge_lora()
    model.fold_projector()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def assert_fresh_lora(model: LightningDiT) -> None:
    if not has_lora(model):
        raise ValueError("later progressive stage has no LoRA adapters")
    for module in model.modules():
        if isinstance(module, LoRALinear) and torch.count_nonzero(module.lora_b):
            raise ValueError("fresh LoRA B matrices must be zero initialized")
