from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = 3


def _positive(name: str, *values: int | float) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} must be positive")


@dataclass
class EncoderConfig:
    model_name: str = "vit_base_patch16_dinov3.lvd1689m"
    feature_dim: int = 768
    register_tokens: int = 4


@dataclass
class REGHeadConfig:
    kind: str = "standard"
    hidden_size: int = 2048
    depth: int = 2
    mlp_ratio: float = 4.0


@dataclass
class REGEarlyPredictionConfig:
    block: int | None = None
    projection_hidden_size: int = 2048
    reinjection_mode: str = "replace"


@dataclass
class REGTimestepConfig:
    train_ratio_min: float = 1.0
    train_ratio_max: float = 1.0
    sample_ratio: float = 1.0


@dataclass
class REGConfig:
    enabled: bool = False
    pool_size: int = 4
    target_std: float = 0.53
    spatial_rope: bool = True
    loss_weight: float = 0.03
    head: REGHeadConfig = field(default_factory=REGHeadConfig)
    timestep: REGTimestepConfig = field(default_factory=REGTimestepConfig)
    early_prediction: REGEarlyPredictionConfig = field(
        default_factory=REGEarlyPredictionConfig
    )

    @property
    def num_tokens(self) -> int:
        return self.pool_size**2


@dataclass
class REPAConfig:
    enabled: bool = False
    block: int = 4
    projection_hidden_size: int = 2048
    loss_weight: float = 0.5
    token_scope: str = "image"


@dataclass
class FeaturesConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    reg: REGConfig = field(default_factory=REGConfig)
    repa: REPAConfig = field(default_factory=REPAConfig)

    @property
    def enabled(self) -> bool:
        return self.reg.enabled or self.repa.enabled


@dataclass
class ModelConfig:
    image_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    gradient_checkpointing: bool = False
    variant: str = "SiT-B"
    features: FeaturesConfig = field(default_factory=FeaturesConfig)

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def model_variant(self) -> str:
        return f"{self.variant}/{self.patch_size}"


@dataclass
class CompileConfig:
    enabled: bool = True
    backend: str = "inductor"
    mode: str = "default"
    fullgraph: bool = False
    dynamic: bool = False


@dataclass
class DataConfig:
    repository: str = "benjamin-paine/imagenet-1k-256x256"
    revision: str = "1bd0400450249a7fe90c0aece37d0d03e7ea956a"
    split: str = "train"
    workers: int = 16
    prefetch_factor: int = 4
    max_samples: int | None = None


@dataclass
class FeatureCacheConfig:
    enabled: bool = False
    directory: str = "data/features"
    batch_size: int = 128
    log_interval: int = 10


@dataclass
class OptimizerConfig:
    kind: str = "muon"
    lr: float = 2e-4
    min_lr: float = 2e-5
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95


@dataclass
class TrainConfig:
    epochs: int = 80
    global_batch_size: int = 1024
    micro_batch_size: int = 256
    max_steps: int | None = None
    grad_clip: float = 1.0
    ema_decay: float = 0.9995
    seed: int = 0
    timestep_loc: float = 0.0
    timestep_scale: float = 1.0
    horizontal_flip_probability: float = 0.5
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class SamplingConfig:
    num_samples: int = 50_000
    batch_size: int = 32
    sampling_steps: int = 50
    solver: str = "euler"
    guidance_scale: float = 1.78
    timestep_shift: float = 1.0
    cfg_start: float = 0.0
    cfg_end: float = 0.9
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
    project: str = "jit-reg"
    name: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"
    checkpoint_artifacts: bool = False


@dataclass
class ExperimentConfig:
    schema_version: int = CONFIG_SCHEMA_VERSION
    data: DataConfig = field(default_factory=DataConfig)
    feature_cache: FeatureCacheConfig = field(default_factory=FeatureCacheConfig)
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
                f"configuration schema version {self.schema_version!r} is unsupported; "
                f"schema_version: {CONFIG_SCHEMA_VERSION} is required (clean break; migration "
                "is not supported)"
            )
        m = self.model
        _positive(
            "model dimensions",
            m.image_size,
            m.patch_size,
            m.hidden_size,
            m.depth,
            m.num_heads,
            m.num_classes,
            m.mlp_ratio,
        )
        if m.image_size % m.patch_size:
            raise ValueError("model.image_size must be exactly divisible by model.patch_size")
        if m.in_channels != 3:
            raise ValueError("model.in_channels must be 3 for RGB pixel training")
        if m.hidden_size % m.num_heads:
            raise ValueError("model.hidden_size must be divisible by model.num_heads")
        if (m.hidden_size // m.num_heads) % 4:
            raise ValueError("model attention head dimension must be divisible by 4 for 2D RoPE")
        if not 0 <= m.class_dropout_prob < 1:
            raise ValueError("model.class_dropout_prob must be in [0, 1)")
        if not m.variant:
            raise ValueError("model.variant cannot be empty")

        features = m.features
        # A true baseline does not depend on DINO, feature timing, or cache settings.
        if features.enabled:
            encoder = features.encoder
            _positive(
                "model.features.encoder dimensions",
                encoder.feature_dim,
                encoder.register_tokens,
            )
            if not encoder.model_name:
                raise ValueError("model.features.encoder.model_name cannot be empty")
        if features.reg.enabled:
            reg = features.reg
            if reg.pool_size < 1 or reg.pool_size > 16:
                raise ValueError(
                    "model.features.reg.pool_size must be within the native 16x16 grid"
                )
            if not math.isfinite(reg.target_std) or reg.target_std <= 0:
                raise ValueError(
                    "model.features.reg.target_std must be finite and positive"
                )
            if reg.loss_weight < 0:
                raise ValueError("model.features.reg.loss_weight cannot be negative")
            if reg.head.kind not in {"standard", "wide_mlp"}:
                raise ValueError(
                    "model.features.reg.head.kind must be standard or wide_mlp"
                )
            if reg.head.kind == "wide_mlp":
                _positive(
                    "model.features.reg.head dimensions",
                    reg.head.hidden_size,
                    reg.head.depth,
                    reg.head.mlp_ratio,
                )
            block = reg.early_prediction.block
            if block is not None and not 1 <= block < m.depth:
                raise ValueError(
                    "model.features.reg.early_prediction.block must be null or before "
                    "model.depth"
                )
            if block is not None:
                _positive(
                    "model.features.reg.early_prediction.projection_hidden_size",
                    reg.early_prediction.projection_hidden_size,
                )
            if reg.early_prediction.reinjection_mode not in {"replace", "add"}:
                raise ValueError(
                    "model.features.reg.early_prediction.reinjection_mode "
                    "must be replace or add"
                )
            timing = reg.timestep
            values = (
                timing.train_ratio_min,
                timing.train_ratio_max,
                timing.sample_ratio,
            )
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("REG timestep ratios must be finite and positive")
            if timing.train_ratio_min > timing.train_ratio_max:
                raise ValueError("REG train timestep ratio minimum cannot exceed maximum")
        if features.repa.enabled:
            repa = features.repa
            if not 1 <= repa.block <= m.depth:
                raise ValueError("model.features.repa.block must be within model.depth")
            if repa.projection_hidden_size < 1:
                raise ValueError(
                    "model.features.repa.projection_hidden_size must be positive"
                )
            if repa.loss_weight < 0:
                raise ValueError("model.features.repa.loss_weight cannot be negative")
            if repa.token_scope not in {"image", "image_and_reg"}:
                raise ValueError(
                    "model.features.repa.token_scope must be image or image_and_reg"
                )
            if repa.token_scope == "image_and_reg" and not features.reg.enabled:
                raise ValueError("REPA image_and_reg token scope requires REG")

        if self.train.optimizer.kind not in {"adamw", "muon"}:
            raise ValueError("optimizer.kind must be adamw or muon")
        if self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")
        if self.data.workers < 0 or self.data.prefetch_factor < 1:
            raise ValueError("data workers must be non-negative and prefetch_factor positive")
        if self.data.max_samples is not None and self.data.max_samples < 1:
            raise ValueError("data.max_samples must be positive")
        if not self.data.repository or not self.data.revision or not self.data.split:
            raise ValueError("data repository, revision, and split cannot be empty")
        for name, value in (
            ("validation.interval", self.validation.interval),
            ("checkpoint.interval", self.checkpoint.interval),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.validation.periodic_samples > self.validation.final_samples:
            raise ValueError("periodic validation must be a subset of final validation")
        if (
            self.validation.periodic_samples % m.num_classes
            or self.validation.final_samples % m.num_classes
        ):
            raise ValueError("validation sample counts must be class balanced")
        if self.train.global_batch_size < self.train.micro_batch_size:
            raise ValueError("global batch size must be at least the micro batch size")
        if self.train.epochs < 1 or self.train.micro_batch_size < 1:
            raise ValueError("training epochs and batch sizes must be positive")
        if self.train.max_steps is not None and self.train.max_steps < 1:
            raise ValueError("train.max_steps must be positive")
        if not 0 < self.train.ema_decay < 1:
            raise ValueError("train.ema_decay must be in (0, 1)")
        if not math.isfinite(self.train.timestep_loc):
            raise ValueError("train.timestep_loc must be finite")
        if not math.isfinite(self.train.timestep_scale) or self.train.timestep_scale <= 0:
            raise ValueError("train.timestep_scale must be finite and positive")
        if not 0 <= self.train.horizontal_flip_probability <= 1:
            raise ValueError("train.horizontal_flip_probability must be in [0, 1]")
        if features.enabled:
            if not self.feature_cache.directory:
                raise ValueError("feature_cache.directory cannot be empty")
            _positive(
                "feature-cache batch size and log interval",
                self.feature_cache.batch_size,
                self.feature_cache.log_interval,
            )
            if (
                self.feature_cache.enabled
                and self.train.horizontal_flip_probability != 0
            ):
                raise ValueError(
                    "cached feature training requires train.horizontal_flip_probability=0"
                )
        if self.checkpoint.keep_last_n < 0:
            raise ValueError("checkpoint.keep_last_n cannot be negative")
        self._validate_sampling()
        if self.evaluation.batch_size < 1 or self.evaluation.log_interval < 1:
            raise ValueError("evaluation batch size and log interval must be positive")
        if not self.evaluation.reference_npz or not self.evaluation.device:
            raise ValueError("evaluation reference_npz and device cannot be empty")

    def _validate_sampling(self) -> None:
        sampling = self.sampling
        _positive(
            "sampling counts",
            sampling.num_samples,
            sampling.batch_size,
            sampling.sampling_steps,
            sampling.log_interval,
        )
        if sampling.solver not in {"euler", "heun"}:
            raise ValueError("sampling.solver must be euler or heun")
        if sampling.num_samples % self.model.num_classes:
            raise ValueError("sampling.num_samples must be class balanced")
        if not math.isfinite(sampling.guidance_scale):
            raise ValueError("sampling.guidance_scale must be finite")
        if not math.isfinite(sampling.timestep_shift) or sampling.timestep_shift <= 0:
            raise ValueError("sampling.timestep_shift must be finite and positive")
        if not 0 <= sampling.cfg_start <= sampling.cfg_end <= 1:
            raise ValueError("sampling CFG interval must satisfy 0 <= start <= end <= 1")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_current_schema(values: Any, source: str | Path) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ValueError(f"{source} must contain a configuration object")
    version = values.get("schema_version")
    if version != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{source} uses unsupported configuration schema {version!r}; "
            f"schema_version: {CONFIG_SCHEMA_VERSION} is required (clean break; old configs "
            "and checkpoints cannot be migrated)"
        )
    return values


def _structured_config(values: dict[str, Any]) -> ExperimentConfig:
    from omegaconf import OmegaConf

    try:
        cfg = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), values)
        OmegaConf.set_struct(cfg, True)
        result = OmegaConf.to_object(cfg)
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
    """Load the current structured YAML schema and reject old or unknown fields."""
    from omegaconf import OmegaConf

    values: dict[str, Any] = {"schema_version": CONFIG_SCHEMA_VERSION}
    if path is not None:
        loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        values = _require_current_schema(loaded, path)
    if overrides:
        override_values = OmegaConf.to_container(
            OmegaConf.from_dotlist(overrides), resolve=True
        )
        if not isinstance(override_values, dict):
            raise ValueError("configuration overrides must form an object")
        values = OmegaConf.to_container(
            OmegaConf.merge(values, override_values), resolve=True
        )
        assert isinstance(values, dict)
    return _structured_config(values)


def load_checkpoint_config(path: str | Path) -> ExperimentConfig:
    """Load a resolved current-schema checkpoint config."""
    values = deepcopy(
        _require_current_schema(json.loads(Path(path).read_text()), path)
    )
    return _structured_config(values)
