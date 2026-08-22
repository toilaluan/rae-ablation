from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = 18
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
    validation_size: int = 50_000
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
        if (self.image_size, self.patch_size, self.hidden_size) != (
            IMAGE_SIZE,
            DINO_PATCH_SIZE,
            DINO_CHANNELS,
        ):
            raise ValueError(
                f"v18 requires DINOv3-B/16 at {IMAGE_SIZE}px with width {DINO_CHANNELS}"
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
    epochs: int = 10
    global_batch_size: int = 1024
    max_steps: int | None = None
    lr: float = 2e-4
    min_lr: float = 2e-5
    grad_clip: float = 1.0
    ema_decay: float = 0.9995
    seed: int = 0
    validation_interval: int = 1
    validation_batches: int = 128
    sample_interval: int = 1000


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
        _positive(
            "decoder dimensions",
            self.width,
            self.depth,
            self.num_heads,
            self.mlp_ratio,
            self.output_patch_size,
            self.discriminator_channels,
        )
        if self.width % self.num_heads:
            raise ValueError("decoder.width must be divisible by decoder.num_heads")
        if self.output_patch_size != DINO_PATCH_SIZE:
            raise ValueError("decoder.output_patch_size must match DINO patch size")
        if min(
            self.loss.l1_weight,
            self.loss.lpips_weight,
            self.loss.gan_weight,
        ) < 0:
            raise ValueError("decoder loss weights cannot be negative")
        if not 0 <= self.loss.gan_start_fraction < 1:
            raise ValueError("decoder.loss.gan_start_fraction must be in [0,1)")
        train = self.train
        _positive(
            "decoder training settings",
            train.epochs,
            train.global_batch_size,
            train.lr,
            train.min_lr,
            train.grad_clip,
            train.validation_interval,
            train.validation_batches,
            train.sample_interval,
        )
        if train.max_steps is not None and train.max_steps < 1:
            raise ValueError("decoder.train.max_steps must be positive")
        if not 0 < train.ema_decay < 1:
            raise ValueError("decoder.train.ema_decay must be in (0,1)")


@dataclass
class DiTConfig:
    input_size: int = DINO_GRID_SIZE
    patch_size: int = 1
    in_channels: int = DINO_CHANNELS
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    gradient_checkpointing: bool = True
    variant: str = "LightningDiT-B/1-DINOv3B-8x8-Semantic-AdaLN"

    @property
    def grid_size(self) -> int:
        return self.input_size // self.patch_size

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self.in_channels, self.input_size, self.input_size

    @property
    def model_variant(self) -> str:
        return self.variant

    def validate(self) -> None:
        _positive(
            "DiT dimensions",
            self.input_size,
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            self.depth,
            self.num_heads,
            self.mlp_ratio,
        )
        if self.input_size % self.patch_size:
            raise ValueError("model input size must be divisible by patch size")
        if self.hidden_size % self.num_heads or (
            self.hidden_size // self.num_heads
        ) % 4:
            raise ValueError("attention head dimensions must be divisible by four")
        if self.hidden_size % 4:
            raise ValueError("model hidden size must be divisible by four")
        if not self.variant:
            raise ValueError("model.variant cannot be empty")


ModelConfig = DiTConfig


@dataclass
class OptimizerConfig:
    kind: str = "adamw"
    lr: float = 2e-4
    min_lr: float = 2e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95


@dataclass
class FlowTrainConfig:
    epochs: int = 20
    global_batch_size: int = 1024
    max_steps: int | None = None
    grad_clip: float = 1.0
    ema_decay: float = 0.9995
    seed: int = 0
    timestep_loc: float = 0.0
    timestep_scale: float = 1.0
    timestep_shift: float = 4.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class FlowConfig:
    endpoint: str = "target"
    train: FlowTrainConfig = field(default_factory=FlowTrainConfig)

    def validate(self) -> None:
        if self.endpoint not in {"target", "pixel"}:
            raise ValueError("flow.endpoint must be target or pixel")


@dataclass
class SamplingConfig:
    batch_size: int = 32
    steps: int = 40
    solver: str = "euler"
    timestep_shift: float = 4.0
    noise_seed: int = 0
    log_interval: int = 10


@dataclass
class EvaluationConfig:
    reference_npz: str = "data/VIRTUAL_imagenet128_labeled.npz"
    batch_size: int = 32
    device: str = "auto"
    log_interval: int = 10
    reconstruction_fid: bool = True


@dataclass
class ValidationConfig:
    interval: int = 2
    periodic_samples: int = 10_000
    final_samples: int = 50_000
    plan_path: str = "validation-plan.pt"
    validate_now: bool = False


@dataclass
class CheckpointConfig:
    directory: str = "checkpoints"
    interval: int = 2
    keep_last_n: int = 8
    save_best_lpips: bool = True


@dataclass
class WandbConfig:
    enabled: bool = False
    entity: str | None = None
    project: str = "dino-semantic-bridge"
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
    model: DiTConfig = field(default_factory=DiTConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"configuration schema version {self.schema_version!r} is unsupported; "
                f"schema_version: {CONFIG_SCHEMA_VERSION} is required (clean break)"
            )
        self.vision.validate()
        self.decoder.validate()
        self.model.validate()
        self.flow.validate()
        if self.model.input_shape != self.vision.token_shape:
            raise ValueError("vision and flow token shapes must match")
        train = self.flow.train
        if train.optimizer.kind not in {"adamw", "muon", "fused_muon"}:
            raise ValueError("optimizer.kind must be adamw, muon, or fused_muon")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")
        if self.data.workers < 0 or self.data.prefetch_factor < 1:
            raise ValueError("data workers must be non-negative and prefetch_factor positive")
        _positive("validation dataset size", self.data.validation_size)
        if self.data.max_samples is not None and self.data.max_samples < 1:
            raise ValueError("data.max_samples must be positive")
        if not all(
            (
                self.data.repository,
                self.data.revision,
                self.data.split,
                self.data.validation_split,
            )
        ):
            raise ValueError("dataset provenance cannot be empty")
        _positive("flow training counts", train.epochs, train.global_batch_size)
        if train.max_steps is not None and train.max_steps < 1:
            raise ValueError("flow.train.max_steps must be positive")
        if not 0 < train.ema_decay < 1:
            raise ValueError("flow.train.ema_decay must be in (0,1)")
        if not math.isfinite(train.timestep_loc):
            raise ValueError("flow.train.timestep_loc must be finite")
        _positive(
            "flow training settings",
            train.timestep_scale,
            train.timestep_shift,
            train.grad_clip,
        )
        if min(self.validation.interval, self.checkpoint.interval) < 0:
            raise ValueError("validation and checkpoint intervals cannot be negative")
        if self.validation.periodic_samples > self.validation.final_samples:
            raise ValueError("periodic validation must be a subset of final validation")
        effective_validation_size = min(
            self.data.validation_size,
            self.data.max_samples or self.data.validation_size,
        )
        if not 1 < self.validation.periodic_samples <= self.validation.final_samples:
            raise ValueError("validation sample counts must satisfy 1 < periodic <= final")
        if self.validation.final_samples > effective_validation_size:
            raise ValueError("final validation samples exceed the configured dataset size")
        if self.checkpoint.keep_last_n < 0:
            raise ValueError("checkpoint.keep_last_n cannot be negative")
        self._validate_sampling()

    def _validate_sampling(self) -> None:
        sampling = self.sampling
        _positive(
            "sampling settings",
            sampling.batch_size,
            sampling.steps,
            sampling.log_interval,
            sampling.timestep_shift,
        )
        if sampling.solver not in {"euler", "heun"}:
            raise ValueError("sampling.solver must be euler or heun")
        if sampling.noise_seed < 0:
            raise ValueError("sampling.noise_seed cannot be negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, overrides: list[str]) -> ExperimentConfig:
        from omegaconf import OmegaConf

        base = OmegaConf.structured(deepcopy(self))
        if overrides:
            base = OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))
        result = OmegaConf.to_object(base)
        if not isinstance(result, ExperimentConfig):
            raise TypeError("configuration overrides did not produce ExperimentConfig")
        result.validate()
        return result


def _structured_config(values: dict[str, Any], source: str) -> ExperimentConfig:
    from omegaconf import OmegaConf

    if values.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{source} uses unsupported configuration schema "
            f"{values.get('schema_version')!r}; schema_version: "
            f"{CONFIG_SCHEMA_VERSION} is required"
        )
    try:
        merged = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), values)
        result = OmegaConf.to_object(merged)
    except Exception as error:
        raise ValueError(
            f"invalid schema-v{CONFIG_SCHEMA_VERSION} configuration: {error}"
        ) from error
    if not isinstance(result, ExperimentConfig):
        raise TypeError("configuration did not produce ExperimentConfig")
    result.validate()
    return result


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    from omegaconf import OmegaConf

    source = Path(path)
    values = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    if not isinstance(values, dict):
        raise ValueError(f"configuration must be a mapping: {source}")
    config = _structured_config(values, str(source))
    return config.with_overrides(overrides or [])


def load_checkpoint_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    values = json.loads(source.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"checkpoint configuration must be a mapping: {source}")
    return _structured_config(values, str(source))


def default_config_dict() -> dict[str, Any]:
    values: dict[str, Any] = {"schema_version": CONFIG_SCHEMA_VERSION}
    return _structured_config(values, "defaults").as_dict()
