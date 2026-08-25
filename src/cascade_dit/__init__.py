"""Progressive DINO-to-pixel LightningDiT experiment."""

from .alignment import OrthogonalProjector, StreamingProcrustes, cumulative_projectors
from .config import ExperimentConfig, ModelConfig, load_config
from .model import LightningDiT, PixelPatchDetailer, TokenDetailHead, build_model
from .representations import ChannelStatistics, patchify_rgb, unpatchify_rgb
from .vision import DINOMultiLayerEncoder

__all__ = [
    "ChannelStatistics",
    "DINOMultiLayerEncoder",
    "ExperimentConfig",
    "LightningDiT",
    "ModelConfig",
    "OrthogonalProjector",
    "PixelPatchDetailer",
    "StreamingProcrustes",
    "TokenDetailHead",
    "build_model",
    "cumulative_projectors",
    "load_config",
    "patchify_rgb",
    "unpatchify_rgb",
]

__version__ = "0.2.0"
