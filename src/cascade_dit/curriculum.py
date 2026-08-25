from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .config import ExperimentConfig


@dataclass(frozen=True)
class StagePosition:
    index: int
    name: str
    first_epoch: int
    final_epoch: int
    epoch_within_stage: int


def stage_at_epoch(epoch: int, config: ExperimentConfig) -> StagePosition:
    if epoch < 0 or epoch >= config.total_epochs:
        raise ValueError("epoch is outside the curriculum")
    beginning = 0
    for index, stage in enumerate(config.stages):
        end = beginning + stage.epochs
        if epoch < end:
            return StagePosition(index, stage.name, beginning, end - 1, epoch - beginning)
        beginning = end
    raise AssertionError("unreachable curriculum position")


def configure_trainability(model: nn.Module) -> set[str]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}
