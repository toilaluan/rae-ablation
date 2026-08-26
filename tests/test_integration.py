from __future__ import annotations

import torch

from cascade_dit.checkpoint import EMA, CheckpointManager, ResumeState
from cascade_dit.config import (
    DDTAttentionConfig,
    DDTDecoderConfig,
    DDTPathConfig,
    ExperimentConfig,
    ModelConfig,
)
from cascade_dit.data import PreparedEndpointBatch
from cascade_dit.model import build_model
from cascade_dit.optim import build_optimizer, parameter_count
from cascade_dit.rng import RNGStreams
from cascade_dit.training import compute_training_loss


def _config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.training.semantic.epochs = 1
    config.training.pixel.epochs = 1
    config.model = ModelConfig(
        encoder=DDTPathConfig(16, 1, 1),
        decoder=DDTDecoderConfig(32, 2, 1, 2),
        attention=DDTAttentionConfig(rope=False),
    )
    config.model.conditioning.num_classes = 4
    config.optimizer.global_batch_size = 1
    config.validation.periodic_samples = 4
    config.evaluation.num_samples = 4
    config.validate()
    return config


def _batch(ratio: float) -> PreparedEndpointBatch:
    semantic = torch.randn(1, 64, 768)
    pixel = torch.randn(1, 64, 768)
    ratios = torch.tensor([ratio])
    clean = torch.lerp(semantic, pixel, ratios[:, None, None])
    return PreparedEndpointBatch(
        clean,
        semantic if ratio != 1 else None,
        pixel,
        ratios,
        torch.tensor([2]),
    )


def test_progressive_boundary_keeps_optimizer_and_architecture_matches_baseline() -> None:
    config = _config()
    model = build_model(config.model)
    optimizer = build_optimizer(model, config.optimizer)
    optimizer_identity = id(optimizer)
    ema = EMA(model)
    streams = RNGStreams.create(8)
    for phase, ratio in (("semantic", 0.0), ("pixel", 1.0)):
        optimizer.zero_grad(set_to_none=True)
        result = compute_training_loss(
            model,
            _batch(ratio),
            streams,
            config,
            phase=phase,  # type: ignore[arg-type]
        )
        result.loss.total.backward()
        optimizer.step()
        ema.update(model)
        assert torch.isfinite(result.loss.total)
    assert id(optimizer) == optimizer_identity

    baseline_config = _config()
    baseline_config.training.mode = "pixel_only"
    baseline = build_model(baseline_config.model)
    assert parameter_count(model) == parameter_count(baseline)
    baseline_result = compute_training_loss(
        baseline,
        _batch(1.0),
        RNGStreams.create(8),
        baseline_config,
        phase="pixel",
    )
    assert torch.isfinite(baseline_result.loss.total)


def test_cpu_ddp_pixel_mode_tolerates_unused_semantic_head(tmp_path) -> None:
    if torch.distributed.is_initialized():
        return
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'ddp-init'}",
        rank=0,
        world_size=1,
    )
    try:
        config = _config()
        model = build_model(config.model)
        execution = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True
        )
        result = compute_training_loss(
            execution,
            _batch(1.0),
            RNGStreams.create(9),
            config,
            phase="pixel",
        )
        result.loss.total.backward()
        assert model.semantic_head.projection.weight.grad is None
    finally:
        torch.distributed.destroy_process_group()


def test_checkpoint_resume_state_supports_each_fixed_phase(tmp_path) -> None:
    config = _config()
    for phase, completed, within in (("semantic", 0, 0), ("pixel", 1, 0)):
        model = build_model(config.model)
        manager = CheckpointManager(tmp_path / phase)
        state = ResumeState(
            schema_version=5,
            completed_epoch=completed,
            global_step=3,
            phase=phase,  # type: ignore[arg-type]
            epoch_within_phase=within,
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
        assert loaded.phase == phase
        assert loaded.epoch_within_phase == within
