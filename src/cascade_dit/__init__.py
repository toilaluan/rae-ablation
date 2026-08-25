"""Adjacent-flow DINO-to-pixel DDT experiment."""

from .config import ExperimentConfig, ModelConfig, load_config
from .model import DDT, build_model
from .representations import ChannelStatistics, patchify_rgb, unpatchify_rgb
from .vision import DINOMultiLayerEncoder

__all__ = [
    "DDT",
    "ChannelStatistics",
    "DINOMultiLayerEncoder",
    "ExperimentConfig",
    "ModelConfig",
    "build_model",
    "load_config",
    "patchify_rgb",
    "unpatchify_rgb",
]

__version__ = "0.3.0"
