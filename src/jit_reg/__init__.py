"""Pixel- and FLUX.2-latent DDT training with optional DINOv3 REG and REPA."""

from .config import (
    DDTDecoderConfig,
    ExperimentConfig,
    FeaturesConfig,
    ModelConfig,
    REGConfig,
    REGLoRAConfig,
    VAEConfig,
    load_config,
)
from .model import (
    DDT,
    DDTOutput,
    DDTREGOutput,
    classifier_free_labels,
)

__all__ = [
    "DDT",
    "DDTDecoderConfig",
    "DDTOutput",
    "DDTREGOutput",
    "ExperimentConfig",
    "FeaturesConfig",
    "ModelConfig",
    "REGConfig",
    "REGLoRAConfig",
    "VAEConfig",
    "classifier_free_labels",
    "load_config",
]
__version__ = "0.1.0"
