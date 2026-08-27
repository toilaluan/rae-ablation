from __future__ import annotations

import os

import torch
from torch import nn

from cascade_dit.checkpoint import (
    EMA,
    TRAINING_STATE_SCHEMA_VERSION,
    CheckpointManager,
    ResumeState,
)
from cascade_dit.config import (
    DDTAttentionConfig,
    DDTPathConfig,
    ExperimentConfig,
    ModelConfig,
    PixelDecoderConfig,
    SemanticDecoderConfig,
)
from cascade_dit.data import PreparedJointBatch
from cascade_dit.model import DDT, build_model
from cascade_dit.optim import build_optimizer
from cascade_dit.rng import RNGStreams
from cascade_dit.runner import ExperimentRunner
from cascade_dit.training import ScheduledTrainingModel, compute_training_loss


class FastDetailer(nn.Module):
    def forward(self, coarse: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return coarse + 0.01 * context[:, :3, None, None]


class LPIPSStub(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).abs().mean((1, 2, 3), keepdim=True)


def _config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.training.epochs = 2
    config.model = ModelConfig(
        base_prediction_depth=1,
        encoder=DDTPathConfig(16, 1, 1),
        semantic_decoder=SemanticDecoderConfig(32, 1, 2),
        pixel_decoder=PixelDecoderConfig(32, 3, 2, [1], [1]),
        attention=DDTAttentionConfig(rope=False),
    )
    config.model.conditioning.num_classes = 4
    config.optimizer.global_batch_size = 1
    config.validation.periodic_samples = 4
    config.evaluation.num_samples = 4
    config.validate()
    return config


def _model(config: ExperimentConfig) -> DDT:
    model = build_model(config.model)
    model.patch_detailer = FastDetailer()
    return model


def _batch() -> PreparedJointBatch:
    return PreparedJointBatch(
        torch.randn(1, 64, 768),
        torch.rand(1, 3, 128, 128),
        torch.tensor([2]),
    )


def test_cpu_joint_training_smoke_updates_one_model_and_ema() -> None:
    config = _config()
    model = _model(config)
    optimizer = build_optimizer(model, config.optimizer)
    ema = EMA(model)
    optimizer.zero_grad(set_to_none=True)
    result = compute_training_loss(
        model,
        _batch(),
        RNGStreams.create(8),
        config,
        LPIPSStub(),
    )
    result.loss.total.backward()
    optimizer.step()
    ema.update(model)
    assert torch.isfinite(result.loss.total)
    assert result.output.rgb is not None
    assert result.output.rgb.shape == (1, 3, 128, 128)


def test_cpu_ddp_joint_training_has_no_unused_phase_parameters(tmp_path) -> None:
    if torch.distributed.is_initialized():
        return
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'ddp-init'}",
        rank=0,
        world_size=1,
    )
    try:
        config = _config()
        model = _model(config)
        execution = torch.nn.parallel.DistributedDataParallel(model)
        result = compute_training_loss(
            execution,
            _batch(),
            RNGStreams.create(9),
            config,
            LPIPSStub(),
        )
        result.loss.total.backward()
        assert model.full_head.projection.weight.grad is not None
        assert model.base_head.projection.weight.grad is not None
        assert model.coarse_patch_projection.weight.grad is not None
    finally:
        torch.distributed.destroy_process_group()


def test_sparse_decoder_schedule_reuses_features_for_inner_updates() -> None:
    config = _config()
    config.training.decoder.update_interval = 2
    config.training.decoder.updates_per_active_step = 3
    model = _model(config)
    runner = object.__new__(ExperimentRunner)
    runner.config = config
    runner.device = torch.device("cpu")
    runner.global_step = 0
    runner.model = model
    runner.optimizer = build_optimizer(model, config.optimizer)
    runner.ema = EMA(model)
    runner.rng = RNGStreams.create(10)
    runner.lpips_model = LPIPSStub()
    execution = ScheduledTrainingModel(model)

    optimizer_steps = 0
    original_step = runner.optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal optimizer_steps
        optimizer_steps += 1
        return original_step(*args, **kwargs)

    runner.optimizer.step = counted_step
    active = runner._run_scheduled_training_step(execution, _batch())
    assert active["decoder_active"] == 1
    assert active["decoder_updates"] == 3
    assert optimizer_steps == 3

    pixel_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name in model.pixel_parameter_names()
    }
    runner.global_step = 1
    inactive = runner._run_scheduled_training_step(execution, _batch())
    assert inactive["decoder_active"] == 0
    assert inactive["decoder_updates"] == 0
    assert optimizer_steps == 4
    assert all(
        torch.equal(pixel_before[name], parameter)
        for name, parameter in model.named_parameters()
        if name in pixel_before
    )


def test_cpu_ddp_sparse_decoder_modes_support_unused_parameters(tmp_path) -> None:
    if torch.distributed.is_initialized():
        return
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'scheduled-ddp-init'}",
        rank=0,
        world_size=1,
    )
    try:
        config = _config()
        config.training.decoder.update_interval = 2
        config.training.decoder.updates_per_active_step = 2
        model = _model(config)
        execution = torch.nn.parallel.DistributedDataParallel(
            ScheduledTrainingModel(model), find_unused_parameters=True
        )
        runner = object.__new__(ExperimentRunner)
        runner.config = config
        runner.device = torch.device("cpu")
        runner.model = model
        runner.optimizer = build_optimizer(model, config.optimizer)
        runner.ema = EMA(model)
        runner.rng = RNGStreams.create(11)
        runner.lpips_model = LPIPSStub()
        for step in range(3):
            runner.global_step = step
            metrics = runner._run_scheduled_training_step(execution, _batch())
            assert metrics["decoder_active"] == int(step % 2 == 0)
    finally:
        torch.distributed.destroy_process_group()


def test_checkpoint_resume_state_is_single_phase(tmp_path) -> None:
    config = _config()
    model = _model(config)
    manager = CheckpointManager(tmp_path)
    state = ResumeState(
        schema_version=TRAINING_STATE_SCHEMA_VERSION,
        completed_epoch=1,
        global_step=3,
        training_complete=False,
        images_seen=4,
        estimated_flops=5,
        wall_clock_seconds=1.0,
        optimizer=None,
        scheduler=None,
        rng_streams=RNGStreams.create(4).state_dict(),
        data_fingerprint="data",
        representation_fingerprint="representations",
    )
    saved = manager.save(model=model, ema=EMA(model), state=state, config=config)
    loaded = manager.load(saved, model, EMA(model))
    assert loaded.completed_epoch == 1
    assert not loaded.training_complete
    assert not hasattr(loaded, "phase")
