from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = 16
IMAGE_SIZE = 128
DINO_CHANNELS = 768
DINO_PATCH_SIZE = 16
DINO_GRID_SIZE = IMAGE_SIZE // DINO_PATCH_SIZE
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
class EndpointCacheConfig:
    enabled: bool = False
    directory: str = "data/dino-endpoints"
    batch_size: int = 128
    log_interval: int = 10

    def validate(self) -> None:
        if not self.directory:
            raise ValueError("vision.cache.directory cannot be empty")
        _positive("vision cache settings", self.batch_size, self.log_interval)


@dataclass
class VisionConfig:
    repository: str = DINO_REPOSITORY
    revision: str = DINO_REVISION
    target_layer: int = 2
    semantic_layer: int = 12
    image_size: int = IMAGE_SIZE
    patch_size: int = DINO_PATCH_SIZE
    hidden_size: int = DINO_CHANNELS
    num_register_tokens: int = 4
    statistics_path: str = "data/dino-endpoint-statistics.safetensors"
    compile: CompileConfig = field(default_factory=CompileConfig)
    cache: EndpointCacheConfig = field(default_factory=EndpointCacheConfig)

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def token_shape(self) -> tuple[int, int, int]:
        return self.hidden_size, self.grid_size, self.grid_size

    def validate(self) -> None:
        if self.repository != DINO_REPOSITORY:
            raise ValueError(f"vision.repository must be {DINO_REPOSITORY}")
        if not self.revision:
            raise ValueError("vision.revision must pin an immutable model revision")
        if (self.image_size, self.patch_size, self.hidden_size) != (IMAGE_SIZE, DINO_PATCH_SIZE, DINO_CHANNELS):
            raise ValueError(
                f"v16 requires DINOv3-B/16 at {IMAGE_SIZE}px with width {DINO_CHANNELS}"
            )
        if not 1 <= self.target_layer < self.semantic_layer <= 12:
            raise ValueError("vision layers must satisfy 1 <= target < semantic <= 12")
        if self.num_register_tokens != 4:
            raise ValueError("the pinned DINOv3-B/16 checkpoint must have four registers")
        if not self.statistics_path:
            raise ValueError("vision.statistics_path cannot be empty")
        self.cache.validate()
@dataclass
class DecoderLossConfig:
    l1_weight: float = 1.0
    lpips_weight: float = 1.0
    gan_weight: float = 0.1
    gan_enabled: bool = True
    gan_start_fraction: float = 0.2


@dataclass
class DecoderTrainConfig:
    epochs: int = 40
    global_batch_size: int = 1024
    max_steps: int | None = None
    lr: float = 2e-4
    min_lr: float = 2e-5
    grad_clip: float = 1.0
    ema_decay: float = 0.9995
    seed: int = 0
    validation_interval: int = 1
    validation_batches: int = 128
    sample_interval: int = 1_000


@dataclass
class PixelDecoderConfig:
    checkpoint: str | None = None
    width: int = 768
    depth: int = 8
    num_heads: int = 12
    mlp_ratio: float = 4.0
    output_patch_size: int = 16
    discriminator_channels: int = 64
    loss: DecoderLossConfig = field(default_factory=DecoderLossConfig)
    train: DecoderTrainConfig = field(default_factory=DecoderTrainConfig)

    def validate(self) -> None:
        _positive("decoder dimensions", self.width, self.depth, self.num_heads, self.mlp_ratio, self.output_patch_size, self.discriminator_channels)
        if self.width % self.num_heads:
            raise ValueError("decoder.width must be divisible by decoder.num_heads")
        if self.output_patch_size != DINO_PATCH_SIZE:
            raise ValueError("decoder.output_patch_size must match DINO patch size")
        loss = self.loss
        if min(loss.l1_weight, loss.lpips_weight, loss.gan_weight) < 0:
            raise ValueError("decoder loss weights cannot be negative")
        if not 0 <= loss.gan_start_fraction < 1:
            raise ValueError("decoder.loss.gan_start_fraction must be in [0,1)")
        train = self.train
        _positive("decoder training settings", train.epochs, train.global_batch_size, train.lr, train.min_lr, train.grad_clip, train.validation_interval, train.validation_batches, train.sample_interval)
        if train.max_steps is not None and train.max_steps < 1:
            raise ValueError("decoder.train.max_steps must be positive")
        if not 0 < train.ema_decay < 1:
            raise ValueError("decoder.train.ema_decay must be in (0,1)")


@dataclass
class DDTConfig:
    input_size: int = DINO_GRID_SIZE
    patch_size: list[int] = field(default_factory=lambda: [1, 1])
    in_channels: int = DINO_CHANNELS
    hidden_size: list[int] = field(default_factory=lambda: [384, 1536])
    depth: list[int] = field(default_factory=lambda: [12, 2])
    num_heads: list[int] = field(default_factory=lambda: [6, 16])
    mlp_ratio: float = 4.0
    base_model_depth: int = 4
    num_classes: int = 1000
    cfg_dropout_prob: float = 0.1
    num_timestep_tokens: int = 4
    num_stage_tokens: int = 4
    num_class_tokens: int = 8
    gradient_checkpointing: bool = False
    variant: str = "DDT-B/1-DINOv3B-8x8-Semantic-Cascade-IG"

    @property
    def grid_size(self) -> int:
        return self.input_size // self.patch_size[0]

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self.in_channels, self.input_size, self.input_size

    @property
    def model_variant(self) -> str:
        return self.variant

    def validate(self) -> None:
        if any(len(value) != 2 for value in (self.patch_size, self.hidden_size, self.depth, self.num_heads)):
            raise ValueError("model paired dimensions must contain two values")
        _positive("DDT dimensions", self.input_size, *self.patch_size, self.in_channels, *self.hidden_size, *self.depth, *self.num_heads, self.mlp_ratio, self.base_model_depth, self.num_classes, self.num_timestep_tokens, self.num_stage_tokens, self.num_class_tokens)
        if any(self.input_size % patch for patch in self.patch_size):
            raise ValueError("model input size must be divisible by both patch sizes")
        if self.patch_size[0] != self.patch_size[1]:
            raise ValueError("encoder and decoder patch sizes must match")
        for width, heads in zip(self.hidden_size, self.num_heads, strict=True):
            if width % heads or (width // heads) % 4:
                raise ValueError("attention head dimensions must be divisible by four")
        if self.base_model_depth > self.depth[0]:
            raise ValueError("model.base_model_depth must be within encoder depth")
        if not 0 <= self.cfg_dropout_prob < 1:
            raise ValueError("model.cfg_dropout_prob must be in [0,1)")
        if not self.variant:
            raise ValueError("model.variant cannot be empty")


ModelConfig = DDTConfig


@dataclass
class CascadeConfig:
    pool_size: int = 2
    downsample_mode: str = "area"
    upsample_mode: str = "bilinear"
    align_corners: bool = False
    semantic_stage_probability: float = 0.5

    def validate(self) -> None:
        _positive("cascade.pool_size", self.pool_size)
        if DINO_GRID_SIZE % self.pool_size:
            raise ValueError("cascade.pool_size must divide the DINO token grid")
        if self.downsample_mode not in {"area", "bilinear"}:
            raise ValueError("cascade.downsample_mode must be area or bilinear")
        if self.upsample_mode not in {"nearest", "bilinear"}:
            raise ValueError("cascade.upsample_mode must be nearest or bilinear")
        if not 0 < self.semantic_stage_probability < 1:
            raise ValueError("cascade.semantic_stage_probability must be in (0,1)")


@dataclass
class RepaConfig:
    enabled: bool = True
    layer_depth: int = 8
    projector_width: int = 2048
    coefficient: float = 0.5

    def validate(self, model_depth: int) -> None:
        if not 1 <= self.layer_depth <= model_depth:
            raise ValueError("repa.layer_depth must be within the diffusion encoder")
        _positive("REPA settings", self.projector_width, self.coefficient)


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
    base_loss_weight: float = 1.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class SamplingConfig:
    num_samples: int = 50_000
    batch_size: int = 32
    semantic_steps: int = 10
    target_steps: int = 40
    solver: str = "euler"
    timestep_shift: float = 1.0
    ig_scale: float = 1.78
    ig_start: float = 0.1
    ig_end: float = 1.0
    cfg_scale: float = 1.0
    cfg_start: float = 0.0
    cfg_end: float = 1.0
    log_interval: int = 10


@dataclass
class EvaluationConfig:
    reference_npz: str = "data/VIRTUAL_imagenet128_labeled.npz"
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
    project: str = "dino-semantic-cascade"
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"
    checkpoint_artifacts: bool = False


@dataclass
class ExperimentConfig:
    schema_version: int = CONFIG_SCHEMA_VERSION
    data: DataConfig = field(default_factory=DataConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    decoder: PixelDecoderConfig = field(default_factory=PixelDecoderConfig)
    model: DDTConfig = field(default_factory=DDTConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    repa: RepaConfig = field(default_factory=RepaConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"configuration schema version {self.schema_version!r} is unsupported; schema_version: {CONFIG_SCHEMA_VERSION} is required (clean break)")
        self.vision.validate()
        self.decoder.validate()
        self.model.validate()
        self.cascade.validate()
        self.repa.validate(self.model.depth[0])
        if self.model.input_shape != self.vision.token_shape:
            raise ValueError("vision and diffusion token shapes must match")
        if self.train.optimizer.kind not in {"adamw", "muon", "fused_muon"}:
            raise ValueError("optimizer.kind must be adamw, muon, or fused_muon")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")
        if self.data.workers < 0 or self.data.prefetch_factor < 1:
            raise ValueError("data workers must be non-negative and prefetch_factor positive")
        if self.data.max_samples is not None and self.data.max_samples < 1:
            raise ValueError("data.max_samples must be positive")
        if not all((self.data.repository, self.data.revision, self.data.split, self.data.validation_split)):
            raise ValueError("dataset provenance cannot be empty")
        _positive("training counts", self.train.epochs, self.train.global_batch_size)
        if self.train.max_steps is not None and self.train.max_steps < 1:
            raise ValueError("train.max_steps must be positive")
        if not 0 < self.train.ema_decay < 1:
            raise ValueError("train.ema_decay must be in (0,1)")
        if not math.isfinite(self.train.timestep_loc):
            raise ValueError("train.timestep_loc must be finite")
        _positive("training flow settings", self.train.timestep_scale, self.train.timestep_shift, self.train.velocity_denominator_floor)
        if self.train.base_loss_weight < 0:
            raise ValueError("train.base_loss_weight cannot be negative")
        for name, value in (("validation.interval", self.validation.interval), ("checkpoint.interval", self.checkpoint.interval)):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.validation.periodic_samples > self.validation.final_samples:
            raise ValueError("periodic validation must be a subset of final validation")
        if self.validation.periodic_samples % self.model.num_classes or self.validation.final_samples % self.model.num_classes:
            raise ValueError("validation sample counts must be class balanced")
        if self.checkpoint.keep_last_n < 0:
            raise ValueError("checkpoint.keep_last_n cannot be negative")
        self._validate_sampling()

    def _validate_sampling(self) -> None:
        s = self.sampling
        _positive("sampling counts", s.num_samples, s.batch_size, s.semantic_steps, s.target_steps, s.log_interval)
        if s.solver not in {"euler", "heun"}:
            raise ValueError("sampling.solver must be euler or heun")
        if s.num_samples % self.model.num_classes:
            raise ValueError("sampling.num_samples must be class balanced")
        if not math.isfinite(s.timestep_shift) or s.timestep_shift <= 0:
            raise ValueError("sampling.timestep_shift must be finite and positive")
        if not math.isfinite(s.ig_scale) or not math.isfinite(s.cfg_scale):
            raise ValueError("sampling guidance scales must be finite")
        if not 0 <= s.ig_start < s.ig_end <= 1:
            raise ValueError("sampling IG interval must satisfy 0 <= start < end <= 1")
        if not 0 <= s.cfg_start < s.cfg_end <= 1:
            raise ValueError("sampling CFG interval must satisfy 0 <= start < end <= 1")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_current_schema(values: Any, source: str | Path) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ValueError(f"{source} must contain a configuration object")
    if values.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"{source} uses unsupported configuration schema {values.get('schema_version')!r}; schema_version: {CONFIG_SCHEMA_VERSION} is required")
    return values


def _structured_config(values: dict[str, Any]) -> ExperimentConfig:
    from omegaconf import OmegaConf

    try:
        cfg = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), values)
        OmegaConf.set_struct(cfg, True)
        result = OmegaConf.to_object(cfg)
    except Exception as error:
        raise ValueError(f"invalid schema-v{CONFIG_SCHEMA_VERSION} configuration: {error}") from error
    if not isinstance(result, ExperimentConfig):
        raise TypeError("configuration did not resolve to ExperimentConfig")
    result.validate()
    return result


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
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
    return _structured_config(deepcopy(_require_current_schema(json.loads(Path(path).read_text()), path)))
