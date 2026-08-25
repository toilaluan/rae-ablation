from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = 5

IMAGE_SIZE = 128
PATCH_SIZE = 16
GRID_SIZE = 8
TOKEN_COUNT = GRID_SIZE * GRID_SIZE
TOKEN_DIM = 768
RGB_CHANNELS = 3
RGB_PATCH_DIM = RGB_CHANNELS * PATCH_SIZE * PATCH_SIZE

DINO_REPOSITORY = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
IMAGENET_REPOSITORY = "benjamin-paine/imagenet-1k-256x256"
IMAGENET_REVISION = "1bd0400450249a7fe90c0aece37d0d03e7ea956a"


def _positive(name: str, *values: int | float) -> None:
    if any(not math.isfinite(float(value)) or value <= 0 for value in values):
        raise ValueError(f"{name} must be finite and positive")


@dataclass
class ExperimentOptions:
    mode: str = "progressive"
    name: str = "adjacent-flow-ddt"
    seed: int = 0
    output_directory: str = "runs/experiment"


@dataclass
class DataConfig:
    repository: str = IMAGENET_REPOSITORY
    revision: str = IMAGENET_REVISION
    split: str = "train"
    validation_split: str = "validation"
    image_size: int = IMAGE_SIZE
    workers: int = 16
    prefetch_factor: int = 4
    max_samples: int | None = None
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
class RepresentationSourceConfig:
    kind: str = "dino"
    layer: int | None = None


@dataclass
class RepresentationStageConfig:
    name: str = ""
    epochs: int = 20
    source: RepresentationSourceConfig = field(default_factory=RepresentationSourceConfig)
    timestep_shift: float = 4.0


def _default_stages() -> list[RepresentationStageConfig]:
    return [
        RepresentationStageConfig("dino12", 20, RepresentationSourceConfig("dino", 12), 4.0),
        RepresentationStageConfig("dino8", 20, RepresentationSourceConfig("dino", 8), 4.0),
        RepresentationStageConfig("dino4", 20, RepresentationSourceConfig("dino", 4), 4.0),
        RepresentationStageConfig("rgb", 20, RepresentationSourceConfig("rgb", None), 4.0),
    ]


@dataclass
class RepresentationsConfig:
    directory: str = "data/direct-representations"
    manifest: str = "manifest.json"
    stages: list[RepresentationStageConfig] = field(default_factory=_default_stages)
    dtype: str = "bfloat16"
    cache_batch_size: int = 128
    asynchronous_writes: bool = True
    statistics_dtype: str = "float64"
    normalization_epsilon: float = 1e-6
    require_complete: bool = True


@dataclass
class DDTPathConfig:
    width: int = 768
    depth: int = 12
    heads: int = 12


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
    encoder: DDTPathConfig = field(default_factory=DDTPathConfig)
    decoder: DDTPathConfig = field(
        default_factory=lambda: DDTPathConfig(width=2048, depth=2, heads=16)
    )
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
class AdjacentMixConfig:
    probability: float = 0.10
    distribution: str = "uniform"
    minimum: float = 0.0
    maximum: float = 1.0


@dataclass
class AllLevelsMixConfig:
    probability: float = 0.025
    distribution: str = "uniform_simplex"
    concentration: float = 1.0


@dataclass
class MixingConfig:
    adjacent: AdjacentMixConfig = field(default_factory=AdjacentMixConfig)
    pure_next_probability: float = 0.025
    all_levels: AllLevelsMixConfig = field(default_factory=AllLevelsMixConfig)
    final_stage: str = "pure"

    @property
    def pure_current_probability(self) -> float:
        return 1.0 - (
            self.adjacent.probability
            + self.pure_next_probability
            + self.all_levels.probability
        )


@dataclass
class CurriculumConfig:
    mixing: MixingConfig = field(default_factory=MixingConfig)
    source_consistency_weight: float = 0.0


@dataclass
class ObjectiveConfig:
    prediction: str = "x0"
    loss_space: str = "velocity"
    timestep_distribution: str = "shifted_logit_normal"
    timestep_loc: float = 0.0
    timestep_scale: float = 1.0
    denominator_floor: float = 0.05
    cosine_loss_weight: float = 0.0
    cosine_loss_epsilon: float = 1e-8


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


@dataclass
class SamplingConfig:
    steps: int = 40
    solver: str = "euler"
    timestep_shift: float = 1.0
    guidance_scale: float = 1.0
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    batch_size: int = 32
    seed: int = 0


@dataclass
class ValidationConfig:
    interval_epochs: int = 5
    batches: int = 32
    pca_preview_samples: int = 16
    periodic_samples: int = 10_000


@dataclass
class CheckpointConfig:
    directory: str = "checkpoints"
    interval_epochs: int = 5
    save_stage_boundaries: bool = True
    keep_last_n: int = 8
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
    project: str = "adjacent-flow-ddt"
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"
    checkpoint_artifacts: bool = False


@dataclass
class EvaluationConfig:
    num_samples: int = 50_000
    reference_npz: str = "data/VIRTUAL_imagenet128_labeled.npz"
    metrics: list[str] = field(default_factory=lambda: ["fid", "is"])
    batch_size: int = 32
    device: str = "auto"
    report_flops: bool = True
    report_peak_memory: bool = True


@dataclass
class ExperimentConfig:
    schema_version: int = CONFIG_SCHEMA_VERSION
    experiment: ExperimentOptions = field(default_factory=ExperimentOptions)
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    representations: RepresentationsConfig = field(default_factory=RepresentationsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def stages(self) -> list[RepresentationStageConfig]:
        return self.representations.stages

    @property
    def stage_names(self) -> list[str]:
        return [stage.name for stage in self.stages]

    @property
    def dino_layers(self) -> list[int]:
        return [
            int(stage.source.layer)
            for stage in self.stages
            if stage.source.kind == "dino" and stage.source.layer is not None
        ]

    @property
    def total_epochs(self) -> int:
        return sum(stage.epochs for stage in self.stages)

    @property
    def num_representation_levels(self) -> int:
        return len(self.stages)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"configuration schema {self.schema_version!r} is unsupported; "
                f"schema_version: {CONFIG_SCHEMA_VERSION} is required"
            )
        if self.experiment.mode not in {"progressive", "pixel_baseline"}:
            raise ValueError("experiment.mode must be progressive or pixel_baseline")
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
        self._validate_stages()
        if self.representations.dtype != "bfloat16":
            raise ValueError("representations.dtype must be bfloat16")
        if self.representations.statistics_dtype != "float64":
            raise ValueError("representation statistics must accumulate in float64")
        if not self.representations.require_complete:
            raise ValueError("training must require complete representation caches")
        _positive(
            "representation settings",
            self.representations.cache_batch_size,
            self.representations.normalization_epsilon,
        )
        self._validate_model()
        self._validate_mixing()
        if self.curriculum.source_consistency_weight < 0:
            raise ValueError("source consistency weight cannot be negative")
        if self.objective.prediction != "x0" or self.objective.loss_space != "velocity":
            raise ValueError("the experiment requires x0 prediction with velocity loss")
        if self.objective.timestep_distribution != "shifted_logit_normal":
            raise ValueError("objective timestep distribution must be shifted_logit_normal")
        _positive(
            "objective settings",
            self.objective.timestep_scale,
            self.objective.denominator_floor,
        )
        self._validate_optimizer()
        if self.sampling.solver not in {"euler", "heun"}:
            raise ValueError("sampling.solver must be euler or heun")
        _positive(
            "sampling settings",
            self.sampling.steps,
            self.sampling.batch_size,
            self.sampling.timestep_shift,
        )
        if not 0 <= self.sampling.guidance_start < self.sampling.guidance_end <= 1:
            raise ValueError("sampling guidance interval must satisfy 0 <= start < end <= 1")
        if self.data.workers < 0 or self.data.prefetch_factor < 1:
            raise ValueError("data workers must be non-negative and prefetch_factor positive")
        if self.data.max_samples is not None and self.data.max_samples < 1:
            raise ValueError("data.max_samples must be positive")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")
        if set(self.evaluation.metrics) != {"fid", "is"}:
            raise ValueError("evaluation.metrics must contain fid and is")
        if self.evaluation.num_samples % self.model.num_classes:
            raise ValueError("evaluation.num_samples must be class balanced")
        if self.validation.periodic_samples % self.model.num_classes:
            raise ValueError("validation.periodic_samples must be class balanced")
        _positive(
            "validation settings",
            self.validation.batches,
            self.validation.pca_preview_samples,
            self.validation.periodic_samples,
            self.evaluation.num_samples,
            self.evaluation.batch_size,
        )
        if self.validation.periodic_samples > self.evaluation.num_samples:
            raise ValueError("periodic validation samples cannot exceed final evaluation samples")

    def _validate_stages(self) -> None:
        if len(self.stages) < 2:
            raise ValueError("at least two ordered representation stages are required")
        names = self.stage_names
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("representation stage names must be non-empty and unique")
        if any(stage.epochs < 1 for stage in self.stages):
            raise ValueError("every representation stage must have positive epochs")
        _positive("representation stage timestep shifts", *(stage.timestep_shift for stage in self.stages))
        rgb = [index for index, stage in enumerate(self.stages) if stage.source.kind == "rgb"]
        if rgb != [len(self.stages) - 1]:
            raise ValueError("exactly one RGB representation is required as the final stage")
        layers: list[int] = []
        for stage in self.stages:
            source = stage.source
            if source.kind == "dino":
                if source.layer is None or not 1 <= source.layer <= 12:
                    raise ValueError("DINO stages require a one-indexed layer in [1,12]")
                layers.append(source.layer)
            elif source.kind == "rgb":
                if source.layer is not None:
                    raise ValueError("RGB stage source.layer must be null")
            else:
                raise ValueError("representation source kind must be dino or rgb")
        if len(layers) != len(set(layers)):
            raise ValueError("DINO representation layers must be unique")

    def _validate_model(self) -> None:
        if self.model.architecture != "ddt" or self.model.token_dim != TOKEN_DIM:
            raise ValueError("model must be a 768-dimensional universal DDT")
        for name, path in (("encoder", self.model.encoder), ("decoder", self.model.decoder)):
            _positive(f"model {name}", path.width, path.depth, path.heads)
            if path.width % path.heads or (path.width // path.heads) % 4:
                raise ValueError(f"model.{name} head width must be divisible by four")
        _positive(
            "model settings",
            self.model.feedforward.mlp_ratio,
            self.model.num_classes,
            self.model.conditioning.timestep_tokens,
            self.model.conditioning.class_tokens,
        )
        if not 0 <= self.model.class_dropout_probability < 1:
            raise ValueError("class dropout probability must be in [0,1)")

    def _validate_mixing(self) -> None:
        mixing = self.curriculum.mixing
        probabilities = (
            mixing.adjacent.probability,
            mixing.pure_next_probability,
            mixing.all_levels.probability,
        )
        if any(not math.isfinite(value) or value < 0 for value in probabilities):
            raise ValueError("mixture probabilities must be finite and non-negative")
        if mixing.pure_current_probability < 0:
            raise ValueError("mixture probabilities cannot sum above one")
        adjacent = mixing.adjacent
        if adjacent.distribution != "uniform":
            raise ValueError("adjacent mixture distribution must be uniform")
        if not 0 <= adjacent.minimum < adjacent.maximum <= 1:
            raise ValueError("adjacent uniform bounds must satisfy 0 <= min < max <= 1")
        if mixing.all_levels.distribution != "uniform_simplex":
            raise ValueError("all-level distribution must be uniform_simplex")
        _positive("all-level concentration", mixing.all_levels.concentration)
        if mixing.final_stage != "pure":
            raise ValueError("the final representation stage must be pure")

    def _validate_optimizer(self) -> None:
        values = self.optimizer
        if values.kind != "adamw":
            raise ValueError("optimizer.kind must be adamw")
        if values.scheduler.kind not in {"constant", "cosine"}:
            raise ValueError("optimizer.scheduler.kind must be constant or cosine")
        if len(values.adamw.betas) != 2 or not all(
            0 <= beta < 1 for beta in values.adamw.betas
        ):
            raise ValueError("optimizer.adamw.betas must contain two values in [0,1)")
        _positive(
            "optimizer settings",
            values.adamw.lr,
            values.global_batch_size,
            values.gradient_clip,
        )
        if values.scheduler.warmup_epochs < 0:
            raise ValueError("scheduler warmup epochs cannot be negative")
        if not 0 <= values.scheduler.min_lr_ratio <= 1:
            raise ValueError("scheduler min_lr_ratio must be in [0,1]")
        if not 0 < values.ema_decay < 1:
            raise ValueError("optimizer.ema_decay must be in (0,1)")


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
    if values.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{source} uses unsupported configuration schema "
            f"{values.get('schema_version')!r}; schema_version: "
            f"{CONFIG_SCHEMA_VERSION} is required. Alignment-era LightningDiT and "
            "legacy DDT/Mage configurations are not supported."
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
