"""Pixel-space JiT training with optional DINOv3 REG and REPA."""

from .config import ExperimentConfig, ModelConfig, REGConfig, load_config
from .model import SiT, SiTOutput

__all__ = [
    "ExperimentConfig",
    "ModelConfig",
    "REGConfig",
    "SiT",
    "SiTOutput",
    "load_config",
]
__version__ = "0.1.0"
