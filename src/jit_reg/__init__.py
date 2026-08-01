"""High-throughput SiT training with optional REPA and K-exploration."""

from .config import (
    ExperimentConfig,
    ExplorationConfig,
    ModelConfig,
    REPAConfig,
    VAEConfig,
    load_config,
)
from .model import SiT, SiTOutput, classifier_free_labels

__all__ = [
    "ExperimentConfig",
    "ExplorationConfig",
    "ModelConfig",
    "REPAConfig",
    "SiT",
    "SiTOutput",
    "VAEConfig",
    "classifier_free_labels",
    "load_config",
]
__version__ = "0.2.0"
