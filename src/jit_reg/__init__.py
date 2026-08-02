"""RAEv2-compressed ImageNet training with DDT internal guidance and K exploration."""

from .config import DDTConfig, ExperimentConfig, ExplorationConfig, RAEConfig, load_config
from .model import DDT, DDTOutput, DiTwDDTHeadIG, classifier_free_labels

__all__ = [
    "DDT",
    "DDTConfig",
    "DDTOutput",
    "DiTwDDTHeadIG",
    "ExperimentConfig",
    "ExplorationConfig",
    "RAEConfig",
    "classifier_free_labels",
    "load_config",
]
__version__ = "0.3.0"
