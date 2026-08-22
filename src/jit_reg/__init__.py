"""Gaussian DINOv3 semantic-conditioned target and pixel flow matching."""

from .config import (
    DiTConfig,
    EndpointCacheConfig,
    ExperimentConfig,
    FlowConfig,
    VisionConfig,
    load_config,
)
from .model import SemanticDiT

__all__ = [
    "DiTConfig",
    "EndpointCacheConfig",
    "ExperimentConfig",
    "FlowConfig",
    "SemanticDiT",
    "VisionConfig",
    "load_config",
]
__version__ = "0.6.0"
