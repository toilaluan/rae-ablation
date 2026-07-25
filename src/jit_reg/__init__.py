"""Pixel-space JiT training with optional DINOv3 REG and REPA."""

from .config import ExperimentConfig, FeaturesConfig, ModelConfig, REGConfig, load_config
from .model import SiT, SiTOutput

__all__ = [
    "ExperimentConfig",
    "FeaturesConfig",
    "ModelConfig",
    "REGConfig",
    "SiT",
    "SiTOutput",
    "load_config",
]
__version__ = "0.1.0"
