from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = 2

IMAGE_SIZE = 256
IMAGE_LATENT_CHANNELS = 128
IMAGE_LATENT_SIZE = 16
IMAGE_LATENT_SHAPE = (IMAGE_LATENT_CHANNELS, IMAGE_LATENT_SIZE, IMAGE_LATENT_SIZE)

SEMANTIC_IMAGE_SIZE = 128
SEMANTIC_CHANNELS = 768
SEMANTIC_PATCH_SIZE = 16
SEMANTIC_GRID_SIZE = 8
SEMANTIC_TOKEN_SHAPE = (SEMANTIC_CHANNELS, SEMANTIC_GRID_SIZE, SEMANTIC_GRID_SIZE)

MAGE_REPOSITORY = "mage-flow-community/Mage-Flow"
MAGE_REVISION = "5edf242ad658cb85050ee077c6629d59e5388fe7"
MAGE_WEIGHT_FILE = "vae/diffusion_pytorch_model.safetensors"
DINO_REPOSITORY = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"

def _positive(name: str, *values: int | float) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} must be positive")


@dataclass
class CompileConfig:
    enabled: bool = True
    backend: str = "inductor"
    mode: str = "default"
    fullgraph: bool = True
    dynamic: bool = False


@dataclass
class DataConfig:
    repository: str = "benjamin-paine/imagenet-1k-256x256"
    revision: str = "1bd0400450249a7fe90c0aece37d0d03e7ea956a"
    split: str = "train"
    validation_split: str = "validation"
    workers: int = 16
    prefetch_factor: int = 4
    max_samples: int | None = None


@dataclass
class CacheConfig:
    enabled: bool = False
    directory: str = "data/cache"
    batch_size: int = 128
    log_interval: int = 10
    asynchronous_writes: bool = True

    def validate(self, path: str) -> None:
        if not self.directory:
            raise ValueError(f"{path}.directory cannot be empty")
        _positive(f"{path} settings", self.batch_size, self.log_interval)


@dataclass
class ImageLatentConfig:
    repository: str = MAGE_REPOSITORY
    revision: str = MAGE_REVISION
    weight_file: str = MAGE_WEIGHT_FILE
    image_size: int = IMAGE_SIZE
    channels: int = IMAGE_LATENT_CHANNELS
    latent_size: int = IMAGE_LATENT_SIZE
    posterior: str = "mean"
    compile: CompileConfig = field(default_factory=CompileConfig)
    cache: CacheConfig = field(
        default_factory=lambda: CacheConfig(directory="data/image-latents")
    )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.channels, self.latent_size, self.latent_size

    def validate(self) -> None:
        if self.repository != MAGE_REPOSITORY or self.revision != MAGE_REVISION:
            raise ValueError("image_latents must use the pinned native Mage-VAE")
        if self.weight_file != MAGE_WEIGHT_FILE:
            raise ValueError(f"image_latents.weight_file must be {MAGE_WEIGHT_FILE}")
        if (self.image_size, self.channels, self.latent_size) != (
            IMAGE_SIZE,
            IMAGE_LATENT_CHANNELS,
            IMAGE_LATENT_SIZE,
        ):
            raise ValueError("Mage-VAE image latent schema must be [128,16,16] at 256px")
        if self.posterior != "mean":
            raise ValueError("image_latents.posterior must be mean for deterministic encoding")
        self.cache.validate("image_latents.cache")


@dataclass
class SemanticTokenConfig:
    repository: str = DINO_REPOSITORY
    revision: str = DINO_REVISION
    image_size: int = SEMANTIC_IMAGE_SIZE
    patch_size: int = SEMANTIC_PATCH_SIZE
    channels: int = SEMANTIC_CHANNELS
    grid_size: int = SEMANTIC_GRID_SIZE
    num_register_tokens: int = 4
    statistics_path: str = "data/semantic-statistics.safetensors"
    compile: CompileConfig = field(default_factory=CompileConfig)
    cache: CacheConfig = field(
        default_factory=lambda: CacheConfig(directory="data/semantic-tokens")
    )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.channels, self.grid_size, self.grid_size

    def validate(self) -> None:
        if self.repository != DINO_REPOSITORY or self.revision != DINO_REVISION:
            raise ValueError("semantic_tokens must use a pinned DINOv3-B/16 checkpoint")
        if (self.image_size, self.patch_size, self.channels, self.grid_size) != (
            SEMANTIC_IMAGE_SIZE,
            SEMANTIC_PATCH_SIZE,
            SEMANTIC_CHANNELS,
            SEMANTIC_GRID_SIZE,
        ):
            raise ValueError("semantic token schema must be [768,8,8] from a 128px view")
        if self.num_register_tokens != 4:
            raise ValueError("the pinned DINOv3-B/16 checkpoint must have four registers")
        if not self.statistics_path:
            raise ValueError("semantic_tokens.statistics_path cannot be empty")
        self.cache.validate("semantic_tokens.cache")


@dataclass
class ModelConfig:
    architecture: str = "shared_cascade"
    image_latent_patch_size: int = 2
    semantic_patch_size: int = 1
    hidden_size: list[int] = field(default_factory=lambda: [1152, 2048])
    depth: list[int] = field(default_factory=lambda: [28, 2])
    num_heads: list[int] = field(default_factory=lambda: [16, 16])
    mlp_ratio: float = 4.0
    base_model_depth: int = 8
    num_classes: int = 1000
    class_dropout_probability: float = 0.1
    num_timestep_tokens: int = 4
    num_class_tokens: int = 8
    gradient_checkpointing: bool = False

    @property
    def token_grid_size(self) -> int:
        return SEMANTIC_GRID_SIZE

    @property
    def image_latent_shape(self) -> tuple[int, int, int]:
        return IMAGE_LATENT_SHAPE

    @property
    def semantic_token_shape(self) -> tuple[int, int, int]:
        return SEMANTIC_TOKEN_SHAPE

    def validate(self) -> None:
        if self.architecture not in {"latent", "shared_cascade", "decoupled_cascade"}:
            raise ValueError(
                "model.architecture must be latent, shared_cascade, or decoupled_cascade"
            )
        paired = (self.hidden_size, self.depth, self.num_heads)
        if not all(len(values) == 2 for values in paired):
            raise ValueError(
                "model.hidden_size, model.depth, and model.num_heads must be "
                "encoder/decoder pairs"
            )
        _positive(
            "model dimensions",
            self.image_latent_patch_size,
            self.semantic_patch_size,
            *self.hidden_size,
            *self.depth,
            *self.num_heads,
            self.mlp_ratio,
            self.base_model_depth,
            self.num_classes,
            self.num_timestep_tokens,
            self.num_class_tokens,
        )
        if (self.image_latent_patch_size, self.semantic_patch_size) != (2, 1):
            raise ValueError(
                "model patches must align native image latents and semantic tokens on 8x8"
            )
        for route, width, heads in zip(
            ("encoder", "decoder"), self.hidden_size, self.num_heads, strict=True
        ):
            if width % heads:
                raise ValueError(
                    f"model {route} hidden size must be divisible by its head count"
                )
            if (width // heads) % 4:
                raise ValueError(
                    f"model {route} attention head size must be divisible by four"
                )
        if self.base_model_depth > self.depth[0]:
            raise ValueError("model.base_model_depth must be within encoder depth")
        if not 0 <= self.class_dropout_probability < 1:
            raise ValueError("model.class_dropout_probability must be in [0,1)")


@dataclass
class OptimizerConfig:
    kind: str = "adamw"
    lr: float = 2e-4
    min_lr: float = 2e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95


@dataclass
class TrainConfig:
    epochs: int = 80
    global_batch_size: int = 1024
    max_steps: int | None = None
    grad_clip: float = 1.0
    ema_decay: float = 0.9995
    seed: int = 0
    timestep_loc: float = 0.0
    timestep_scale: float = 1.0
    timestep_shift: float = 4.0
    velocity_denominator_floor: float = 0.05
    semantic_base_loss_weight: float = 1.0
    predicted_semantic_condition_probability: float = 0.25
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class SamplingConfig:
    num_samples: int = 50_000
    batch_size: int = 32
    semantic_steps: int = 10
    image_steps: int = 40
    solver: str = "euler"
    timestep_shift: float = 1.0
    semantic_guidance_scale: float = 1.78
    semantic_guidance_start: float = 0.1
    semantic_guidance_end: float = 1.0
    class_guidance_scale: float = 1.0
    class_guidance_start: float = 0.0
    class_guidance_end: float = 1.0
    log_interval: int = 10


@dataclass
class EvaluationConfig:
    reference_npz: str = "data/VIRTUAL_imagenet256_labeled.npz"
    batch_size: int = 32
    device: str = "auto"
    log_interval: int = 10


@dataclass
class ValidationConfig:
    interval: int = 5
    periodic_samples: int = 10_000
    final_samples: int = 50_000
    plan_path: str = "validation-plan.pt"
    validate_now: bool = False


@dataclass
class CheckpointConfig:
    directory: str = "checkpoints"
    interval: int = 5
    keep_last_n: int = 8
    save_best_fid: bool = True


@dataclass
class WandbConfig:
    enabled: bool = False
    entity: str | None = None
    project: str = "cascade-dit"
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"
    checkpoint_artifacts: bool = False


@dataclass
class ExperimentConfig:
    schema_version: int = CONFIG_SCHEMA_VERSION
    data: DataConfig = field(default_factory=DataConfig)
    image_latents: ImageLatentConfig = field(default_factory=ImageLatentConfig)
    semantic_tokens: SemanticTokenConfig = field(default_factory=SemanticTokenConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"configuration schema {self.schema_version!r} is unsupported; "
                f"schema_version: {CONFIG_SCHEMA_VERSION} is required (clean break)"
            )
        self.image_latents.validate()
        self.semantic_tokens.validate()
        self.model.validate()
        if self.train.optimizer.kind not in {"adamw", "muon", "fused_muon"}:
            raise ValueError("optimizer.kind must be adamw, muon, or fused_muon")
        if self.wandb.project != "cascade-dit":
            raise ValueError("wandb.project must be cascade-dit")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")
        if self.data.workers < 0 or self.data.prefetch_factor < 1:
            raise ValueError("data workers must be non-negative and prefetch_factor positive")
        if self.data.max_samples is not None and self.data.max_samples < 1:
            raise ValueError("data.max_samples must be positive")
        if not all((self.data.repository, self.data.revision, self.data.split)):
            raise ValueError("dataset provenance cannot be empty")
        _positive("training counts", self.train.epochs, self.train.global_batch_size)
        if self.train.max_steps is not None and self.train.max_steps < 1:
            raise ValueError("train.max_steps must be positive")
        if not 0 < self.train.ema_decay < 1:
            raise ValueError("train.ema_decay must be in (0,1)")
        if not math.isfinite(self.train.timestep_loc):
            raise ValueError("train.timestep_loc must be finite")
        _positive(
            "training flow settings",
            self.train.timestep_scale,
            self.train.timestep_shift,
            self.train.velocity_denominator_floor,
        )
        if self.train.semantic_base_loss_weight < 0:
            raise ValueError("train.semantic_base_loss_weight cannot be negative")
        if not 0 <= self.train.predicted_semantic_condition_probability <= 1:
            raise ValueError(
                "train.predicted_semantic_condition_probability must be in [0,1]"
            )
        for name, value in (
            ("validation.interval", self.validation.interval),
            ("checkpoint.interval", self.checkpoint.interval),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.validation.periodic_samples > self.validation.final_samples:
            raise ValueError("periodic validation must be a subset of final validation")
        if self.checkpoint.keep_last_n < 0:
            raise ValueError("checkpoint.keep_last_n cannot be negative")
        self._validate_sampling()

    def _validate_sampling(self) -> None:
        values = self.sampling
        _positive(
            "sampling counts",
            values.num_samples,
            values.batch_size,
            values.image_steps,
            values.log_interval,
        )
        if self.model.architecture != "latent":
            _positive("sampling.semantic_steps", values.semantic_steps)
        elif values.semantic_steps < 0:
            raise ValueError("sampling.semantic_steps cannot be negative")
        if values.solver not in {"euler", "heun"}:
            raise ValueError("sampling.solver must be euler or heun")
        if values.num_samples % self.model.num_classes:
            raise ValueError("sampling.num_samples must be class balanced")
        if not math.isfinite(values.timestep_shift) or values.timestep_shift <= 0:
            raise ValueError("sampling.timestep_shift must be finite and positive")
        for scale in (
            values.semantic_guidance_scale,
            values.class_guidance_scale,
        ):
            if not math.isfinite(scale):
                raise ValueError("sampling guidance scales must be finite")
        if not 0 <= values.semantic_guidance_start < values.semantic_guidance_end <= 1:
            raise ValueError("semantic guidance interval must satisfy 0 <= start < end <= 1")
        if not 0 <= values.class_guidance_start < values.class_guidance_end <= 1:
            raise ValueError("class guidance interval must satisfy 0 <= start < end <= 1")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_current_schema(values: Any, source: str | Path) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ValueError(f"{source} must contain a configuration object")
    if values.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{source} uses unsupported configuration schema "
            f"{values.get('schema_version')!r}; schema_version: "
            f"{CONFIG_SCHEMA_VERSION} is required"
        )
    return values


def _structured_config(values: dict[str, Any]) -> ExperimentConfig:
    from omegaconf import OmegaConf

    try:
        config = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), values)
        OmegaConf.set_struct(config, True)
        result = OmegaConf.to_object(config)
    except Exception as error:
        raise ValueError(
            f"invalid schema-v{CONFIG_SCHEMA_VERSION} configuration: {error}"
        ) from error
    if not isinstance(result, ExperimentConfig):
        raise TypeError("configuration did not resolve to ExperimentConfig")
    result.validate()
    return result


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None
) -> ExperimentConfig:
    from omegaconf import OmegaConf

    values: dict[str, Any] = {"schema_version": CONFIG_SCHEMA_VERSION}
    if path is not None:
        loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        values = _require_current_schema(loaded, path)
    if overrides:
        override_values = OmegaConf.to_container(OmegaConf.from_dotlist(overrides), resolve=True)
        if not isinstance(override_values, dict):
            raise ValueError("configuration overrides must form an object")
        values = OmegaConf.to_container(OmegaConf.merge(values, override_values), resolve=True)
        assert isinstance(values, dict)
    return _structured_config(values)


def load_checkpoint_config(path: str | Path) -> ExperimentConfig:
    values = json.loads(Path(path).read_text())
    return _structured_config(deepcopy(_require_current_schema(values, path)))
