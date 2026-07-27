"""Pixel- and FLUX.2-latent JiT training with optional DINOv3 REG and REPA."""

from .config import (
    ExperimentConfig,
    FeaturesConfig,
    ModelConfig,
    REGConfig,
    VAEConfig,
    load_config,
)
from .model import SiT, SiTOutput

__all__ = [
    "ExperimentConfig",
    "FeaturesConfig",
    "ModelConfig",
    "REGConfig",
    "SiT",
    "SiTOutput",
    "VAEConfig",
    "load_config",
]
__version__ = "0.1.0"
