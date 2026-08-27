from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = 9

IMAGE_SIZE = 128
PATCH_SIZE = 16
GRID_SIZE = 8
TOKEN_COUNT = GRID_SIZE * GRID_SIZE
TOKEN_DIM = 768

DINO_REPOSITORY = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
IMAGENET_REPOSITORY = "benjamin-paine/imagenet-1k-256x256"
IMAGENET_REVISION = "1bd0400450249a7fe90c0aece37d0d03e7ea956a"


def _positive(name: str, *values: int | float) -> None:
    if any(not math.isfinite(float(value)) or value <= 0 for value in values):
        raise ValueError(f"{name} must be finite and positive")


def _non_negative(name: str, *values: int | float) -> None:
    if any(not math.isfinite(float(value)) or value < 0 for value in values):
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass
class ExperimentOptions:
    name: str = "joint-semantic-pixel-ddt"
    seed: int = 0
    output_directory: str = "runs/experiment"


@dataclass
class DataConfig:
    repository: str = IMAGENET_REPOSITORY
    revision: str = IMAGENET_REVISION
    split: str = "train"
    image_size: int = IMAGE_SIZE
    workers: int = 16
    prefetch_factor: int = 4
    max_samples: int | None = None
    shuffle: bool = True


@dataclass
class UnlabeledDataConfig:
    repository: str = IMAGENET_REPOSITORY
    revision: str = IMAGENET_REVISION
    train_split: str = "train"
    validation_split: str = "validation"
    image_column: str = "image"
    image_size: int = IMAGE_SIZE
    workers: int = 16
    prefetch_factor: int = 4
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    shuffle: bool = True


@dataclass
class EncoderConfig:
    repository: str = DINO_REPOSITORY
    revision: str = DINO_REVISION
    patch_size: int = PATCH_SIZE
    hidden_size: int = TOKEN_DIM
    num_register_tokens: int = 4
    apply_final_norm: bool = True


@dataclass
class DecoderOptimizationConfig:
    pretrained_checkpoint: str = "artifacts/pixel-decoder/checkpoints/epoch-0050"
    pretrained_weights: str = "ema"
    frozen_epochs: int = 20
    update_interval: int = 1
    updates_per_active_step: int = 1
    backpropagate_to_flow: bool = True
    samples_per_batch: int | None = None


@dataclass
class TrainingConfig:
    epochs: int = 200
    timestep_shift: float = 4.0
    decoder: DecoderOptimizationConfig = field(
        default_factory=DecoderOptimizationConfig
    )


@dataclass
class SemanticRepresentationConfig:
    name: str = "dino12"
    layer: int = 12


@dataclass
class RepresentationsConfig:
    directory: str = "data/direct-representations"
    manifest: str = "manifest.json"
    semantic: SemanticRepresentationConfig = field(default_factory=SemanticRepresentationConfig)
    cache_batch_size: int = 256
    normalization_epsilon: float = 1e-6
    require_complete: bool = True


@dataclass
class DDTPathConfig:
    width: int = 768
    depth: int = 12
    heads: int = 12


@dataclass
class SemanticDecoderConfig:
    width: int = 2048
    depth: int = 2
    heads: int = 16


@dataclass
class LayerAttentionFusionConfig:
    width: int = 768
    depth: int = 1
    heads: int = 12
    mlp_ratio: float = 2.0


@dataclass
class PixelFusionConfig:
    kind: str = "concatenate"
    layer_attention: LayerAttentionFusionConfig = field(
        default_factory=LayerAttentionFusionConfig
    )


@dataclass
class PixelDecoderConfig:
    width: int = 2048
    depth: int = 3
    heads: int = 16
    encoder_features: list[int] = field(default_factory=lambda: [4, 8, 12])
    decoder_features: list[int] = field(default_factory=lambda: [1, 2])
    fusion: PixelFusionConfig = field(default_factory=PixelFusionConfig)


# Retained as a constructor alias; the phase-era ``decoder`` field itself is gone.
DDTDecoderConfig = SemanticDecoderConfig


@dataclass
class DDTFeedForwardConfig:
    mlp_ratio: float = 4.0


@dataclass
class DDTAttentionConfig:
    qk_norm: bool = True
    rope: bool = True
    sdpa: bool = True


@dataclass
class DDTConditioningConfig:
    num_classes: int = 1000
    class_dropout_probability: float = 0.1
    timestep_tokens: int = 4
    class_tokens: int = 8


@dataclass
class ModelConfig:
    architecture: str = "ddt"
    token_dim: int = TOKEN_DIM
    base_prediction_depth: int = 8
    encoder: DDTPathConfig = field(default_factory=DDTPathConfig)
    semantic_decoder: SemanticDecoderConfig = field(default_factory=SemanticDecoderConfig)
    pixel_decoder: PixelDecoderConfig = field(default_factory=PixelDecoderConfig)
    feedforward: DDTFeedForwardConfig = field(default_factory=DDTFeedForwardConfig)
    attention: DDTAttentionConfig = field(default_factory=DDTAttentionConfig)
    conditioning: DDTConditioningConfig = field(default_factory=DDTConditioningConfig)
    gradient_checkpointing: bool = False

    @property
    def num_classes(self) -> int:
        return self.conditioning.num_classes

    @property
    def class_dropout_probability(self) -> float:
        return self.conditioning.class_dropout_probability


@dataclass
class ObjectiveConfig:
    timestep_loc: float = 0.0
    timestep_scale: float = 1.0
    denominator_floor: float = 0.05
    base_velocity_weight: float = 1.0
    pixel_l1_weight: float = 1.0
    pixel_lpips_weight: float = 1.0


@dataclass
class AdamWConfig:
    lr: float = 2e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig:
    kind: str = "cosine"
    warmup_epochs: int = 0
    min_lr_ratio: float = 0.0


@dataclass
class OptimizerConfig:
    kind: str = "adamw"
    adamw: AdamWConfig = field(default_factory=AdamWConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    gradient_clip: float = 1.0
    global_batch_size: int = 1024
    ema_decay: float = 0.9999


def _pixel_pretraining_optimizer() -> OptimizerConfig:
    return OptimizerConfig(
        scheduler=SchedulerConfig(kind="cosine", warmup_epochs=5, min_lr_ratio=0.0)
    )


@dataclass
class PixelPretrainingConfig:
    epochs: int = 50
    semantic_input_mode: str = "cached"
    data: UnlabeledDataConfig = field(default_factory=UnlabeledDataConfig)
    train_cache_directory: str = "data/direct-representations"
    validation_cache_directory: str = "data/pixel-pretraining/validation"
    manifest: str = "manifest.json"
    cache_batch_size: int = 256
    normalization_epsilon: float = 1e-6
    validation_interval_epochs: int = 1
    validation_batches: int = 32
    checkpoint_interval_epochs: int = 5
    optimizer: OptimizerConfig = field(default_factory=_pixel_pretraining_optimizer)


@dataclass
class InternalGuidanceConfig:
    scale: float = 1.78
    t_min: float = 0.10
    t_max: float = 1.0


@dataclass
class SamplingConfig:
    steps: int = 40
    solver: str = "euler"
    timestep_shift: float = 1.0
    internal_guidance: InternalGuidanceConfig = field(default_factory=InternalGuidanceConfig)
    batch_size: int = 32
    seed: int = 0


@dataclass
class ValidationConfig:
    interval_epochs: int = 5
    batches: int = 32
    periodic_samples: int = 10_000


@dataclass
class CheckpointConfig:
    directory: str = "checkpoints"
    interval_epochs: int = 5
    resume: str | None = None


@dataclass
class CompileConfig:
    enabled: bool = True
    backend: str = "inductor"
    mode: str = "default"
    fullgraph: bool = True
    dynamic: bool = False


@dataclass
class WandbConfig:
    enabled: bool = False
    entity: str | None = None
    project: str = "joint-semantic-pixel-ddt"
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"


@dataclass
class EvaluationConfig:
    num_samples: int = 50_000
    reference_npz: str = "data/VIRTUAL_imagenet128_labeled.npz"
    batch_size: int = 32
    device: str = "auto"


@dataclass
class ExperimentConfig:
    schema_version: int = CONFIG_SCHEMA_VERSION
    experiment: ExperimentOptions = field(default_factory=ExperimentOptions)
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    representations: RepresentationsConfig = field(default_factory=RepresentationsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    pixel_pretraining: PixelPretrainingConfig = field(
        default_factory=PixelPretrainingConfig
    )
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def total_epochs(self) -> int:
        return self.training.epochs

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"configuration schema {self.schema_version!r} is unsupported; "
                f"schema_version: {CONFIG_SCHEMA_VERSION} is required"
            )
        _positive("training settings", self.training.epochs, self.training.timestep_shift)
        _positive(
            "decoder optimization settings",
            self.training.decoder.update_interval,
            self.training.decoder.updates_per_active_step,
        )
        decoder = self.training.decoder
        if not decoder.pretrained_checkpoint:
            raise ValueError("training.decoder.pretrained_checkpoint cannot be empty")
        if decoder.pretrained_weights not in {"ema", "model"}:
            raise ValueError("training.decoder.pretrained_weights must be ema or model")
        if not 0 <= decoder.frozen_epochs < self.training.epochs:
            raise ValueError(
                "training.decoder.frozen_epochs must be non-negative and below total epochs"
            )
        if (
            self.training.decoder.samples_per_batch is not None
            and self.training.decoder.samples_per_batch < 1
        ):
            raise ValueError("training.decoder.samples_per_batch must be positive")
        if self.data.image_size != IMAGE_SIZE:
            raise ValueError("data.image_size must be 128")
        if (self.data.repository, self.data.revision) != (
            IMAGENET_REPOSITORY,
            IMAGENET_REVISION,
        ):
            raise ValueError("data must use the pinned ImageNet repository revision")
        if (self.encoder.repository, self.encoder.revision) != (
            DINO_REPOSITORY,
            DINO_REVISION,
        ):
            raise ValueError("encoder must use the pinned DINOv3-B/16 revision")
        if (self.encoder.patch_size, self.encoder.hidden_size) != (PATCH_SIZE, TOKEN_DIM):
            raise ValueError("DINO representations must use 16px patches and width 768")
        if self.encoder.num_register_tokens != 4 or not self.encoder.apply_final_norm:
            raise ValueError("DINO extraction requires four registers and final normalization")
        semantic = self.representations.semantic
        if not semantic.name or not 1 <= semantic.layer <= 12:
            raise ValueError("the selected semantic representation must be a DINO layer in [1,12]")
        _positive(
            "representation settings",
            self.representations.cache_batch_size,
            self.representations.normalization_epsilon,
        )
        if not self.representations.require_complete:
            raise ValueError("joint training requires a complete semantic representation cache")
        self._validate_model()
        self._validate_pixel_pretraining()
        _positive(
            "objective settings",
            self.objective.timestep_scale,
            self.objective.denominator_floor,
        )
        _non_negative(
            "objective loss weights",
            self.objective.base_velocity_weight,
            self.objective.pixel_l1_weight,
            self.objective.pixel_lpips_weight,
        )
        self._validate_optimizer()
        if self.sampling.solver not in {"euler", "heun"}:
            raise ValueError("sampling.solver must be euler or heun")
        _positive(
            "sampling settings",
            self.sampling.steps,
            self.sampling.batch_size,
            self.sampling.timestep_shift,
            self.sampling.internal_guidance.scale,
        )
        guidance = self.sampling.internal_guidance
        if not 0 <= guidance.t_min < guidance.t_max <= 1:
            raise ValueError("internal-guidance interval must satisfy 0 <= t_min < t_max <= 1")
        if self.data.workers < 0 or self.data.prefetch_factor < 1:
            raise ValueError("data workers must be non-negative and prefetch_factor positive")
        if self.data.max_samples is not None and self.data.max_samples < 1:
            raise ValueError("data.max_samples must be positive")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")
        if self.evaluation.num_samples % self.model.num_classes:
            raise ValueError("evaluation.num_samples must be class balanced")
        if self.validation.periodic_samples % self.model.num_classes:
            raise ValueError("validation.periodic_samples must be class balanced")
        _positive(
            "validation settings",
            self.validation.batches,
            self.validation.periodic_samples,
            self.evaluation.num_samples,
            self.evaluation.batch_size,
        )
        if self.validation.periodic_samples > self.evaluation.num_samples:
            raise ValueError("periodic validation samples cannot exceed final evaluation samples")

    def _validate_model(self) -> None:
        model = self.model
        if model.architecture != "ddt" or model.token_dim != TOKEN_DIM:
            raise ValueError("model must be a 768-dimensional DDT")
        paths = (
            ("encoder", model.encoder),
            ("semantic_decoder", model.semantic_decoder),
            ("pixel_decoder", model.pixel_decoder),
        )
        for name, path in paths:
            _positive(f"model.{name}", path.width, path.depth, path.heads)
            if path.width % path.heads or (path.width // path.heads) % 4:
                raise ValueError(f"model.{name} head width must be divisible by four")
        if not 1 <= model.base_prediction_depth <= model.encoder.depth:
            raise ValueError("model.base_prediction_depth must select an encoder block")
        for name, selected, depth in (
            ("encoder_features", model.pixel_decoder.encoder_features, model.encoder.depth),
            (
                "decoder_features",
                model.pixel_decoder.decoder_features,
                model.semantic_decoder.depth,
            ),
        ):
            if not selected:
                raise ValueError(f"model.pixel_decoder.{name} must not be empty")
            if len(set(selected)) != len(selected):
                raise ValueError(f"model.pixel_decoder.{name} must contain unique blocks")
            if any(not 1 <= block <= depth for block in selected):
                raise ValueError(f"model.pixel_decoder.{name} contains an out-of-range block")
        fusion = model.pixel_decoder.fusion
        if fusion.kind not in {"concatenate", "layer_attention"}:
            raise ValueError(
                "model.pixel_decoder.fusion.kind must be concatenate or layer_attention"
            )
        layer_attention = fusion.layer_attention
        _positive(
            "model.pixel_decoder.fusion.layer_attention",
            layer_attention.width,
            layer_attention.depth,
            layer_attention.heads,
            layer_attention.mlp_ratio,
        )
        if (
            layer_attention.width % layer_attention.heads
            or (layer_attention.width // layer_attention.heads) % 4
        ):
            raise ValueError(
                "model.pixel_decoder.fusion.layer_attention head width must be divisible by four"
            )
        _positive(
            "model settings",
            model.feedforward.mlp_ratio,
            model.num_classes,
            model.conditioning.timestep_tokens,
            model.conditioning.class_tokens,
        )
        if not 0 <= model.class_dropout_probability < 1:
            raise ValueError("class dropout probability must be in [0,1)")

    def _validate_optimizer(self) -> None:
        self._validate_optimizer_config("optimizer", self.optimizer)

    @staticmethod
    def _validate_optimizer_config(name: str, values: OptimizerConfig) -> None:
        if values.kind != "adamw":
            raise ValueError(f"{name}.kind must be adamw")
        if values.scheduler.kind not in {"constant", "cosine"}:
            raise ValueError(f"{name}.scheduler.kind must be constant or cosine")
        if len(values.adamw.betas) != 2 or not all(0 <= beta < 1 for beta in values.adamw.betas):
            raise ValueError(f"{name}.adamw.betas must contain two values in [0,1)")
        _positive(
            f"{name} settings",
            values.adamw.lr,
            values.global_batch_size,
            values.gradient_clip,
        )
        if values.scheduler.warmup_epochs < 0:
            raise ValueError(f"{name} scheduler warmup epochs cannot be negative")
        if not 0 <= values.scheduler.min_lr_ratio <= 1:
            raise ValueError(f"{name} scheduler min_lr_ratio must be in [0,1]")
        if not 0 < values.ema_decay < 1:
            raise ValueError(f"{name}.ema_decay must be in (0,1)")

    def _validate_pixel_pretraining(self) -> None:
        values = self.pixel_pretraining
        if values.semantic_input_mode not in {"cached", "online"}:
            raise ValueError(
                "pixel_pretraining.semantic_input_mode must be cached or online"
            )
        data = values.data
        if not all(
            (
                data.repository,
                data.revision,
                data.train_split,
                data.validation_split,
                data.image_column,
            )
        ):
            raise ValueError("pixel pretraining data fields cannot be empty")
        if data.image_size != IMAGE_SIZE:
            raise ValueError("pixel pretraining images must be 128x128")
        if data.workers < 0 or data.prefetch_factor < 1:
            raise ValueError("pixel pretraining workers and prefetch settings are invalid")
        for name, maximum in (
            ("max_train_samples", data.max_train_samples),
            ("max_validation_samples", data.max_validation_samples),
        ):
            if maximum is not None and maximum < 1:
                raise ValueError(f"pixel_pretraining.data.{name} must be positive")
        _positive(
            "pixel pretraining settings",
            values.epochs,
            values.cache_batch_size,
            values.normalization_epsilon,
            values.validation_interval_epochs,
            values.validation_batches,
            values.checkpoint_interval_epochs,
        )
        if values.optimizer.scheduler.warmup_epochs >= values.epochs:
            raise ValueError("pixel pretraining warmup must be below its epoch count")
        self._validate_optimizer_config(
            "pixel_pretraining.optimizer", values.optimizer
        )


def _structured_config(values: dict[str, Any]) -> ExperimentConfig:
    from omegaconf import OmegaConf

    try:
        config = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), values)
        OmegaConf.set_struct(config, True)
        result = OmegaConf.to_object(config)
    except Exception as error:
        raise ValueError(f"invalid schema-v{CONFIG_SCHEMA_VERSION} configuration: {error}") from error
    if not isinstance(result, ExperimentConfig):
        raise TypeError("configuration did not resolve to ExperimentConfig")
    result.validate()
    return result


def _require_current(values: object, source: str | Path) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ValueError(f"{source} must contain a configuration object")
    version = values.get("schema_version")
    if version != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{source} uses unsupported configuration schema {version!r}; "
            f"schema_version: {CONFIG_SCHEMA_VERSION} is required. Older "
            "configurations are intentionally not migrated."
        )
    return values


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None
) -> ExperimentConfig:
    from omegaconf import OmegaConf

    values: dict[str, Any] = {"schema_version": CONFIG_SCHEMA_VERSION}
    if path is not None:
        loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        values = _require_current(loaded, path)
    if overrides:
        override_values = OmegaConf.to_container(OmegaConf.from_dotlist(overrides), resolve=True)
        if not isinstance(override_values, dict):
            raise ValueError("configuration overrides must form an object")
        values = OmegaConf.to_container(OmegaConf.merge(values, override_values), resolve=True)
        assert isinstance(values, dict)
    return _structured_config(values)


def load_checkpoint_config(path: str | Path) -> ExperimentConfig:
    return _structured_config(_require_current(json.loads(Path(path).read_text()), path))
