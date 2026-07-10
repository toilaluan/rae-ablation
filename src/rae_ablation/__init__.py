"""RAE versus compact-RAE diffusion ablation."""

from .config import DDTIGConfig, ExperimentConfig, load_config
from .model import DiffusionTransformer, DiTOutput, RAEv2DDTIG, build_diffusion_model

__all__ = [
    "DDTIGConfig",
    "DiTOutput",
    "DiffusionTransformer",
    "ExperimentConfig",
    "RAEv2DDTIG",
    "build_diffusion_model",
    "load_config",
]
__version__ = "0.1.0"
