from __future__ import annotations

from torch import nn

from ..config import ModelConfig
from .ddt import RAEv2DDTIG
from .dit import DiffusionTransformer


def build_diffusion_model(config: ModelConfig) -> nn.Module:
    if config.architecture == "depth_attention_dit":
        return DiffusionTransformer(config)
    if config.architecture == "raev2_ddt_ig":
        return RAEv2DDTIG(config)
    raise ValueError(f"unsupported model architecture: {config.architecture}")
