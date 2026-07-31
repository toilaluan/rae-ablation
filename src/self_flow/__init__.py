"""Self-Flow and SIGReg experiments with a FLUX.2-latent SiT backbone."""

from .config import ExperimentConfig, ModelConfig, ObjectiveConfig
from .model import SiT, SiTOutput

__all__ = ["ExperimentConfig", "ModelConfig", "ObjectiveConfig", "SiT", "SiTOutput"]
