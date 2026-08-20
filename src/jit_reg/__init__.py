"""RAE/pixel token interpolation training with DDT internal guidance."""

from .config import (
    DDTConfig,
    ExperimentConfig,
    MixingConfig,
    PixelCodecConfig,
    RAEConfig,
    load_config,
)
from .model import DDT, DDTOutput, DiTwDDTHeadIG, classifier_free_labels

__all__ = [
    "DDT",
    "DDTConfig",
    "DDTOutput",
    "DiTwDDTHeadIG",
    "ExperimentConfig",
    "MixingConfig",
    "PixelCodecConfig",
    "RAEConfig",
    "classifier_free_labels",
    "load_config",
]
__version__ = "0.3.0"
