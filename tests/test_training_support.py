from __future__ import annotations

import json

import torch

from jit_reg.checkpoint import EMA, CheckpointManager
from jit_reg.config import ExperimentConfig
from jit_reg.flops import model_flops_per_step
from jit_reg.model.sit import SiT
from jit_reg.optim import build_optimizer
from jit_reg.rng import RNGStreams
from jit_reg.training import (
    TrainBatch,
    local_batch_size,
    make_train_loader,
    prepare_train_batch,
    train,
)
from jit_reg.validation import create_validation_plan
from jit_reg.wandb_utils import RunLogger


def test_batch_preparation_keeps_optional_cache_paths() -> None:
    source = TrainBatch(
        torch.randint(0, 256, (2, 3, 4, 4), dtype=torch.uint8),
        torch.tensor([1, 2]),
        torch.tensor([3, 4]),
        torch.randn(2, 8, 2, 2, dtype=torch.bfloat16),
        torch.randn(2, 4, 8, dtype=torch.bfloat16),
    )
    batch = prepare_train_batch(source, "cpu")
    assert batch.pixels is not None and batch.pixels.dtype == torch.float32
    assert batch.cached_latents is not None and batch.cached_latents.dtype == torch.float32
    assert batch.cached_repa is not None and batch.cached_repa.dtype == torch.float32


def test_rng_resume_covers_only_active_streams() -> None:
    streams = RNGStreams.create(7)
    state = streams.state_dict()
    assert set(state) == {"shuffle", "initialization", "timesteps", "noise", "label_dropout"}
    RNGStreams.create(7).load_state_dict(state)


def test_flops_include_k_selector_and_repa_prefix() -> None:
    baseline = ExperimentConfig()
    final = ExperimentConfig()
    final.exploration.k_full = 4
    early = ExperimentConfig()
    early.repa.enabled = True
    early.exploration.k_early = 4
    early.exploration.early_selector = "repa"
    hybrid = ExperimentConfig()
    hybrid.early_prediction.enabled = True
    hybrid.exploration.k_early = 4
    hybrid.exploration.k_full = 2
    hybrid.exploration.early_selector = "early_prediction"
    assert model_flops_per_step(final) > model_flops_per_step(early)
    assert model_flops_per_step(early) > model_flops_per_step(baseline)
    assert model_flops_per_step(hybrid) > model_flops_per_step(final)


def test_local_batch_is_data_matched_not_k_scaled() -> None:
    assert local_batch_size(1024, 8) == 128


def test_one_step_hybrid_training_and_schema_v12_checkpoint(tmp_path) -> None:
    config = ExperimentConfig()
    config.model.image_size = 4
    config.model.patch_size = 2
    config.model.hidden_size = 16
    config.model.depth = 1
    config.model.num_heads = 4
    config.model.num_classes = 2
    config.train.epochs = 1
    config.train.max_steps = 1
    config.train.global_batch_size = 2
    config.train.optimizer.kind = "adamw"
    config.data.workers = 0
    config.validation.interval = 0
    config.validation.periodic_samples = 2
    config.validation.final_samples = 2
    config.checkpoint.interval = 0
    config.checkpoint.directory = str(tmp_path / "checkpoints")
    config.validation.plan_path = str(tmp_path / "plan.pt")
    config.compile.enabled = False
    config.early_prediction.enabled = True
    config.early_prediction.block = 1
    config.early_prediction.mlp_hidden_size = 8
    config.exploration.k_early = 2
    config.exploration.k_full = 2
    config.exploration.early_selector = "early_prediction"
    config.validate()
    dataset = [
        (torch.zeros(3, 4, 4, dtype=torch.uint8), 0, 0),
        (torch.full((3, 4, 4), 255, dtype=torch.uint8), 1, 1),
    ]
    streams = RNGStreams.create(3)
    loader, sampler = make_train_loader(dataset, config, streams)
    model = SiT(config.model, config.repa, config.early_prediction)
    ema = EMA(model, config.train.ema_decay)
    plan = create_validation_plan(2, 2, 2, 10)
    plan.save(config.validation.plan_path)
    state = train(
        canonical_model=model,
        execution_model=model,
        loader=loader,
        sampler=sampler,
        optimizer=build_optimizer(model, config.train.optimizer),
        ema=ema,
        streams=streams,
        config=config,
        checkpoint_manager=CheckpointManager(config.checkpoint.directory),
        data_fingerprint="test-data",
        validation_plan_hash=plan.plan_hash,
        validation_plan_path=config.validation.plan_path,
        logger=RunLogger(config.wandb, config.as_dict(), True),
    )
    assert state.global_step == 1 and state.epoch == 1
    checkpoint = tmp_path / "checkpoints" / "last"
    assert (checkpoint / "config.json").is_file()
    assert json.loads((checkpoint / "config.json").read_text())["schema_version"] == 12
    assert json.loads((checkpoint / "metadata.json").read_text())["schema_version"] == 12
