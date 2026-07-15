from __future__ import annotations

from torch import nn

from ..config import ModelConfig
from .ddt import RAEv2DDTIG


def build_diffusion_model(config: ModelConfig) -> nn.Module:
    return RAEv2DDTIG(config)
