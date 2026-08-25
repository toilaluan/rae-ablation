from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = 3

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
    name: str = "dino-to-pixel-lightningdit-b1"
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
    layers: list[int] = field(default_factory=lambda: [12, 8, 4])
    patch_size: int = PATCH_SIZE
    hidden_size: int = TOKEN_DIM
    num_register_tokens: int = 4
    apply_final_norm: bool = True


@dataclass
class RepresentationsConfig:
    directory: str = "data/representations"
    manifest: str = "manifest.json"
    dtype: str = "bfloat16"
    cache_batch_size: int = 128
    asynchronous_writes: bool = True
    statistics_dtype: str = "float64"
    normalization_epsilon: float = 1e-6
    require_complete: bool = True


@dataclass
class AlignmentConfig:
    enabled: bool = True
    subset_size: int = 50_000
    subset_seed: int = 17
    heldout_size: int = 10_000
    pairs: list[str] = field(
        default_factory=lambda: ["dino12->dino8", "dino8->dino4", "dino4->rgb"]
    )
    identity: bool = False
    direct_layer12_to_rgb: bool = False
    svd_driver: str = "gesvd"


@dataclass
class ModelConfig:
    token_dim: int = TOKEN_DIM
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    class_dropout_probability: float = 0.1
    qk_norm: bool = True
    rms_norm: bool = True
    rope: bool = True
    swiglu: bool = True
    adaln_shift: bool = True
    gradient_checkpointing: bool = False
    sdpa: bool = True


@dataclass
class DetailHeadsConfig:
    token_enabled: bool = True
    token_depth: int = 2
    token_expansion: float = 2.0
    pixel_enabled: bool = True
    pixel_channels: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    pixel_bottleneck_context: int = TOKEN_DIM
    zero_initialize: bool = True


@dataclass
class CurriculumConfig:
    stages: list[str] = field(
        default_factory=lambda: ["dino12", "dino8", "dino4", "rgb"]
    )
    epochs: list[int] = field(default_factory=lambda: [20, 20, 20, 20])
    transition_warmup_epochs: int = 1
    fully_train_blocks: list[int] = field(default_factory=lambda: [0, 1, 10, 11])
    lora_blocks: list[int] = field(default_factory=lambda: list(range(2, 10)))
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    later_stage_full_finetune: bool = False
    source_consistency_weight: float = 0.0


@dataclass
class ObjectiveConfig:
    prediction: str = "x0"
    loss_space: str = "velocity"
    timestep_distribution: str = "shifted_logit_normal"
    timestep_loc: float = 0.0
    timestep_scale: float = 1.0
    timestep_shift: float = 4.0
    denominator_floor: float = 0.05
    cosine_loss_weight: float = 0.0
    cosine_loss_epsilon: float = 1e-8


@dataclass
class OptimizerConfig:
    kind: str = "adamw"
    initial_lr: float = 2e-4
    transition_lr: float = 5e-5
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    global_batch_size: int = 1024
    ema_decay: float = 0.9999
    schedule: str = "cosine"
    min_lr_ratio: float = 0.0


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


@dataclass
class CheckpointConfig:
    directory: str = "checkpoints"
    interval_epochs: int = 5
    save_boundary_before: bool = True
    save_boundary_after: bool = True
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
    project: str = "dino-to-pixel-lightningdit"
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
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    detail_heads: DetailHeadsConfig = field(default_factory=DetailHeadsConfig)
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
    def total_epochs(self) -> int:
        return sum(self.curriculum.epochs)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"configuration schema {self.schema_version!r} is unsupported; "
                f"schema_version: {CONFIG_SCHEMA_VERSION} is required. DDT/Mage "
                "configs cannot be migrated."
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
        if self.encoder.layers != [12, 8, 4]:
            raise ValueError("encoder.layers must be the one-indexed layers [12, 8, 4]")
        if (self.encoder.patch_size, self.encoder.hidden_size) != (PATCH_SIZE, TOKEN_DIM):
            raise ValueError("DINOv3-B/16 representations must use 16px patches and width 768")
        if self.encoder.num_register_tokens != 4:
            raise ValueError("the pinned DINOv3-B/16 encoder must use four register tokens")
        if not self.encoder.apply_final_norm:
            raise ValueError("every selected DINO intermediate must use the final norm")
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
        if self.model.token_dim != TOKEN_DIM:
            raise ValueError("model.token_dim must be 768 for raw DINO/RGB patch tokens")
        if (self.model.depth, self.model.hidden_size, self.model.num_heads) != (12, 768, 12):
            raise ValueError("the experiment requires official LightningDiT-B/1 dimensions")
        if self.model.num_classes != 1000:
            raise ValueError("class-conditional ImageNet requires model.num_classes=1000")
        if not all(
            (
                self.model.qk_norm,
                self.model.rms_norm,
                self.model.rope,
                self.model.swiglu,
                self.model.adaln_shift,
                self.model.sdpa,
            )
        ):
            raise ValueError("LightningDiT must retain QK/RMS norm, RoPE, SwiGLU, AdaLN, and SDPA")
        _positive(
            "model dimensions", self.model.hidden_size, self.model.depth,
            self.model.num_heads, self.model.mlp_ratio, self.model.num_classes,
        )
        if self.model.hidden_size % self.model.num_heads:
            raise ValueError("model.hidden_size must be divisible by model.num_heads")
        if (self.model.hidden_size // self.model.num_heads) % 4:
            raise ValueError("the attention head dimension must be divisible by four for 2D RoPE")
        if not 0 <= self.model.class_dropout_probability < 1:
            raise ValueError("model.class_dropout_probability must be in [0,1)")
        if self.detail_heads.token_depth != 2:
            raise ValueError("detail_heads.token_depth must be 2")
        if not self.detail_heads.token_enabled or not self.detail_heads.pixel_enabled:
            raise ValueError("the primary architecture requires both detail heads")
        if not self.detail_heads.zero_initialize:
            raise ValueError("detail-head output projections must be zero initialized")
        if self.detail_heads.pixel_bottleneck_context != TOKEN_DIM:
            raise ValueError("the RGB detail bottleneck context must have width 768")
        if self.detail_heads.pixel_channels != [64, 128, 256, 512]:
            raise ValueError("detail_heads.pixel_channels must be [64, 128, 256, 512]")
        if self.curriculum.stages != ["dino12", "dino8", "dino4", "rgb"]:
            raise ValueError("curriculum.stages must be dino12, dino8, dino4, rgb")
        if len(self.curriculum.epochs) != 4 or any(epoch <= 0 for epoch in self.curriculum.epochs):
            raise ValueError("curriculum.epochs must contain four positive epoch counts")
        all_blocks = self.curriculum.fully_train_blocks + self.curriculum.lora_blocks
        if len(all_blocks) != len(set(all_blocks)):
            raise ValueError("curriculum full-train and LoRA block lists must be disjoint")
        if any(index < 0 or index >= self.model.depth for index in all_blocks):
            raise ValueError("curriculum block index is outside model.depth")
        if self.curriculum.later_stage_full_finetune and self.curriculum.lora_blocks:
            raise ValueError("full-finetune override requires curriculum.lora_blocks=[]")
        _positive("LoRA settings", self.curriculum.lora_rank, self.curriculum.lora_alpha)
        if not 0 <= self.curriculum.lora_dropout < 1:
            raise ValueError("curriculum.lora_dropout must be in [0,1)")
        if self.objective.prediction != "x0" or self.objective.loss_space != "velocity":
            raise ValueError("the experiment requires clean-x0 prediction with velocity-space loss")
        if self.objective.timestep_distribution != "shifted_logit_normal":
            raise ValueError("objective.timestep_distribution must be shifted_logit_normal")
        _positive(
            "objective settings", self.objective.timestep_scale,
            self.objective.timestep_shift, self.objective.denominator_floor,
        )
        if self.optimizer.kind != "adamw" or self.optimizer.schedule != "cosine":
            raise ValueError("this experiment uses AdamW with per-stage cosine decay")
        _positive(
            "optimizer settings", self.optimizer.initial_lr, self.optimizer.transition_lr,
            self.optimizer.global_batch_size, self.optimizer.gradient_clip,
        )
        if not 0 < self.optimizer.ema_decay < 1:
            raise ValueError("optimizer.ema_decay must be in (0,1)")
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
        if not all(
            math.isfinite(value)
            for value in (self.sampling.guidance_scale, self.sampling.timestep_shift)
        ):
            raise ValueError("sampling guidance and timestep shift must be finite")
        if self.data.workers < 0 or self.data.prefetch_factor < 1:
            raise ValueError("data workers must be non-negative and prefetch_factor positive")
        if self.data.max_samples is not None and self.data.max_samples < 1:
            raise ValueError("data.max_samples must be positive")
        if self.alignment.subset_size < 1 or self.alignment.heldout_size < 1:
            raise ValueError("alignment subset and held-out sizes must be positive")
        required_pairs = {"dino12->dino8", "dino8->dino4", "dino4->rgb"}
        if not required_pairs.issubset(self.alignment.pairs):
            raise ValueError("alignment.pairs must contain all consecutive curriculum pairs")
        if len(self.alignment.pairs) != len(set(self.alignment.pairs)):
            raise ValueError("alignment.pairs cannot contain duplicates")
        if self.alignment.svd_driver not in {"gesvd", "gesvdj"}:
            raise ValueError("alignment.svd_driver must be gesvd or gesvdj")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")
        if set(self.evaluation.metrics) != {"fid", "is"}:
            raise ValueError("evaluation.metrics must contain fid and is")
        if self.evaluation.num_samples % self.model.num_classes:
            raise ValueError("evaluation.num_samples must be class balanced")


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
        legacy = {"image_latents", "semantic_tokens", "train"} & set(values)
        suffix = " DDT/Mage checkpoints and configs are intentionally unsupported." if legacy else ""
        raise ValueError(
            f"{source} uses unsupported configuration schema {values.get('schema_version')!r}; "
            f"schema_version: {CONFIG_SCHEMA_VERSION} is required.{suffix}"
        )
    return values


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    from omegaconf import OmegaConf

    values: dict[str, Any] = {"schema_version": CONFIG_SCHEMA_VERSION}
    if path is not None:
        values = _require_current(OmegaConf.to_container(OmegaConf.load(path), resolve=True), path)
    if overrides:
        override_values = OmegaConf.to_container(OmegaConf.from_dotlist(overrides), resolve=True)
        if not isinstance(override_values, dict):
            raise ValueError("configuration overrides must form an object")
        values = OmegaConf.to_container(OmegaConf.merge(values, override_values), resolve=True)
        assert isinstance(values, dict)
    return _structured_config(values)


def load_checkpoint_config(path: str | Path) -> ExperimentConfig:
    return _structured_config(_require_current(json.loads(Path(path).read_text()), path))
