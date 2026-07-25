from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from jit_reg.checkpoint import EMA, CheckpointManager
from jit_reg.checkpoint_bundle import load_checkpoint_bundle
from jit_reg.config import (
    EncoderConfig,
    ExperimentConfig,
    FeaturesConfig,
    ModelConfig,
    REGConfig,
    REGEarlyPredictionConfig,
    REGHeadConfig,
    REPAConfig,
    WandbConfig,
    load_config,
)
from jit_reg.data import data_fingerprint, random_horizontal_flip
from jit_reg.feature_cache import (
    FeatureCache,
    FeatureComponentCacheWriter,
    component_path,
    component_spec,
)
from jit_reg.model.sit import SiT
from jit_reg.optim import build_optimizer
from jit_reg.reg import FeatureTargets
from jit_reg.rng import RNGStreams
from jit_reg.training import make_train_loader, train
from jit_reg.validation import create_validation_plan
from jit_reg.wandb_utils import RunLogger

ROOT = Path(__file__).resolve().parents[1]


def tiny_experiment(kind: str = "baseline") -> ExperimentConfig:
    reg = kind != "baseline"
    repa = kind == "paper"
    early = kind == "early"
    config = ExperimentConfig(
        model=ModelConfig(
            image_size=16,
            patch_size=8,
            hidden_size=16,
            depth=2 if early else 1,
            num_heads=4,
            mlp_ratio=2,
            num_classes=2,
            features=FeaturesConfig(
                encoder=EncoderConfig(feature_dim=8),
                reg=REGConfig(
                    enabled=reg,
                    token_mode="cls_spatial" if early else "cls",
                    pool_size=2,
                    head=REGHeadConfig(
                        kind="wide_mlp" if early else "standard",
                        hidden_size=32,
                        depth=2,
                        mlp_ratio=2,
                    ),
                    early_prediction=REGEarlyPredictionConfig(
                        block=1 if early else None,
                        projection_hidden_size=32,
                    ),
                ),
                repa=REPAConfig(
                    enabled=repa,
                    block=1,
                    projection_hidden_size=16,
                    token_scope="image_and_reg" if repa else "image",
                ),
            ),
        )
    )
    config.compile.enabled = False
    config.train.optimizer.kind = "adamw"
    config.train.epochs = 1
    config.train.global_batch_size = 2
    config.train.micro_batch_size = 2
    config.train.max_steps = 1
    config.data.workers = 0
    config.validation.interval = 0
    config.validation.periodic_samples = 2
    config.validation.final_samples = 2
    return config


class TinyTarget(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config

    def targets(self, images: torch.Tensor, target_grid: int) -> FeatureTargets:
        batch = images.shape[0]
        features = self.config.model.features
        reg = (
            torch.zeros(batch, features.reg.num_tokens, 8)
            if features.reg.enabled
            else None
        )
        repa = (
            torch.nn.functional.normalize(
                torch.ones(batch, target_grid**2, 8), dim=-1
            )
            if features.repa.enabled
            else None
        )
        return FeatureTargets(reg, repa)


class RecordingLogger(RunLogger):
    def __init__(self) -> None:
        super().__init__(WandbConfig(), {}, True, log_interval=100)
        self.metrics: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.metrics.append(metrics)


def save_checkpoint(tmp_path: Path) -> tuple[CheckpointManager, Path, SiT]:
    config = tiny_experiment()
    model = SiT(config.model)
    plan = create_validation_plan(2, 2, 2, seed=7)
    plan_path = tmp_path / "plan.pt"
    plan.save(plan_path)
    manager = CheckpointManager(tmp_path / "checkpoints")
    path = manager.save(
        completed_epoch=1,
        global_step=1,
        model=model,
        ema=EMA(model),
        optimizer_state={"optimizers": []},
        rng_streams=RNGStreams.create(1).state_dict(),
        data_fingerprint="data",
        validation_plan_hash=plan.plan_hash,
        validation_plan_path=plan_path,
        last_completed_validation_epoch=None,
        wandb_run_id=None,
        resolved_config=config.as_dict(),
    )
    return manager, path, model


def test_v2_checkpoint_bundle_and_resume(tmp_path: Path) -> None:
    manager, checkpoint, model = save_checkpoint(tmp_path)
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    assert metadata["schema_version"] == 2
    training = torch.load(checkpoint / "training.pt", weights_only=False)
    assert training["schema_version"] == 2
    restored = SiT(model.config)
    resumed = manager.load(
        checkpoint,
        restored,
        EMA(restored),
        "data",
        metadata["validation_plan_hash"],
    )
    assert resumed.epoch == 1
    bundle = load_checkpoint_bundle(tmp_path / "checkpoints")
    assert bundle.directory == checkpoint.resolve()
    assert bundle.config.schema_version == 2


def test_schema_v1_checkpoint_is_explicitly_rejected(tmp_path: Path) -> None:
    manager, checkpoint, model = save_checkpoint(tmp_path)
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    metadata["schema_version"] = 1
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    restored = SiT(model.config)
    with pytest.raises(ValueError, match="schema-v1"):
        manager.load(checkpoint, restored, EMA(restored))
    with pytest.raises(ValueError, match="schema-v1"):
        load_checkpoint_bundle(checkpoint)


def test_rng_streams_checkpoint_relative_reg_timing() -> None:
    streams = RNGStreams.create(12)
    saved = streams.state_dict()
    first = torch.rand(5, generator=streams.reg_timesteps)
    streams.load_state_dict(saved)
    second = torch.rand(5, generator=streams.reg_timesteps)
    assert torch.equal(first, second)
    old = dict(saved)
    old.pop("reg_timesteps")
    with pytest.raises(ValueError, match="incompatible with schema v2"):
        streams.load_state_dict(old)


@pytest.mark.parametrize(
    ("kind", "cached"),
    [
        ("baseline", False),
        ("paper", False),
        ("early", False),
        ("paper", True),
    ],
)
def test_required_preset_shapes_run_one_training_step(
    tmp_path: Path,
    kind: str,
    cached: bool,
) -> None:
    preset = {
        "baseline": "baseline-p16.yaml",
        "paper": "reg-repa-p16.yaml",
        "early": "early-reg-p16.yaml",
    }[kind]
    config = load_config(
        ROOT / "configs" / preset,
        [
            "model.image_size=16",
            "model.patch_size=8",
            "model.hidden_size=16",
            "model.depth=2",
            "model.num_heads=4",
            "model.mlp_ratio=2",
            "model.num_classes=2",
            "model.features.encoder.feature_dim=8",
            "model.features.reg.pool_size=2",
            "model.features.reg.head.hidden_size=32",
            "model.features.reg.head.depth=2",
            "model.features.reg.head.mlp_ratio=2",
            "model.features.reg.early_prediction.projection_hidden_size=32",
            (
                "model.features.reg.early_prediction.block=1"
                if kind == "early"
                else "model.features.reg.early_prediction.block=null"
            ),
            "model.features.repa.block=1",
            "model.features.repa.projection_hidden_size=16",
            "compile.enabled=false",
            "train.optimizer.kind=adamw",
            "train.epochs=1",
            "train.global_batch_size=2",
            "train.micro_batch_size=2",
            "train.max_steps=1",
            "data.workers=0",
            "feature_cache.enabled=false",
            "validation.interval=0",
            "validation.periodic_samples=2",
            "validation.final_samples=2",
        ],
    )
    case_name = f"{kind}-cached" if cached else kind
    config.checkpoint.directory = str(tmp_path / case_name / "checkpoints")
    feature_cache = None
    if cached:
        config.feature_cache.enabled = True
        config.feature_cache.directory = str(tmp_path / case_name / "features")
        config.train.horizontal_flip_probability = 0
        config.validate()
        for component in ("reg", "repa"):
            spec = component_spec(config, 2, component)
            path = component_path(config, 2, component)
            FeatureComponentCacheWriter.initialize(path, spec)
            writer = FeatureComponentCacheWriter(path, spec)
            values = torch.nn.functional.normalize(torch.ones(writer.shape), dim=-1)
            writer.write(torch.arange(2), values)
            writer.finalize()
        feature_cache = FeatureCache(config, 2)
    plan = create_validation_plan(2, 2, 2, seed=5)
    plan_path = tmp_path / case_name / "plan.pt"
    plan.save(plan_path)
    dataset = TensorDataset(
        torch.rand(2, 3, 16, 16),
        torch.tensor([0, 1]),
        torch.arange(2),
    )
    streams = RNGStreams.create(3)
    loader, sampler = make_train_loader(dataset, config, streams)
    model = SiT(config.model)
    logger = RecordingLogger()
    state = train(
        canonical_model=model,
        execution_model=model,
        loader=loader,
        sampler=sampler,
        optimizer=build_optimizer(model, config.train.optimizer),
        ema=EMA(model),
        streams=streams,
        config=config,
        checkpoint_manager=CheckpointManager(config.checkpoint.directory),
        data_fingerprint="data",
        validation_plan_hash=plan.plan_hash,
        validation_plan_path=plan_path,
        logger=logger,
        feature_encoder=TinyTarget(config)
        if config.model.features.enabled and feature_cache is None
        else None,
        feature_cache=feature_cache,
    )
    assert state.global_step == 1
    assert (Path(config.checkpoint.directory) / "epoch-0001").is_dir()
    logged_names = {name for metrics in logger.metrics for name in metrics}
    assert {
        "throughput/data_wait_seconds",
        "throughput/host_batch_seconds",
        "throughput/optimizer_host_seconds",
        "throughput/logging_seconds",
        "throughput/gpu_input_seconds",
        "throughput/gpu_forward_backward_seconds",
        "throughput/gpu_optimizer_seconds",
        "throughput/gpu_step_seconds",
        "throughput/checkpoint_seconds",
    } <= logged_names


def test_data_fingerprint_and_flip_are_stable() -> None:
    config = tiny_experiment()
    first = data_fingerprint(config.data, config.model.image_size)
    assert first == data_fingerprint(config.data, config.model.image_size)
    assert first != data_fingerprint(config.data, 32)
    values = torch.arange(12).reshape(1, 1, 3, 4)
    assert torch.equal(
        random_horizontal_flip(
            values, torch.Generator().manual_seed(1), probability=1
        ),
        values.flip(-1),
    )
