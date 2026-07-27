"""Pixel- and FLUX.2-latent DDT training with optional DINOv3 REG and REPA."""

from .config import (
    DDTDecoderConfig,
    ExperimentConfig,
    FeaturesConfig,
    ModelConfig,
    REGConfig,
    VAEConfig,
    load_config,
)
from .model import DDT, DDTOutput

__all__ = [
    "DDT",
    "DDTDecoderConfig",
    "DDTOutput",
    "ExperimentConfig",
    "FeaturesConfig",
    "ModelConfig",
    "REGConfig",
    "VAEConfig",
    "load_config",
]
__version__ = "0.1.0"
