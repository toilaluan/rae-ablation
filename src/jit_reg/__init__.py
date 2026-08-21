"""DINOv3 semantic-waypoint diffusion with internal guidance and REPA."""

from .config import (
    DDTConfig,
    EndpointCacheConfig,
    ExperimentConfig,
    RepaConfig,
    VisionConfig,
    WaypointConfig,
    load_config,
)
from .model import DDT, DDTOutput, classifier_free_labels

__all__ = [
    "DDT",
    "DDTConfig",
    "DDTOutput",
    "EndpointCacheConfig",
    "ExperimentConfig",
    "RepaConfig",
    "VisionConfig",
    "WaypointConfig",
    "classifier_free_labels",
    "load_config",
]
__version__ = "0.3.0"
