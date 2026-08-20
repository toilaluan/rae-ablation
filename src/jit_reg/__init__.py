"""DINOv3 token-space diffusion with token mixing and REPA alignment."""

from .config import DDTConfig, ExperimentConfig, MixingConfig, RepaConfig, VisionConfig, load_config
from .model import DDT, DDTOutput, classifier_free_labels

__all__ = [
    "DDT",
    "DDTConfig",
    "DDTOutput",
    "ExperimentConfig",
    "MixingConfig",
    "RepaConfig",
    "VisionConfig",
    "classifier_free_labels",
    "load_config",
]
__version__ = "0.3.0"
