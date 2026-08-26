from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import cascade_dit.runner as runner_module
from cascade_dit.checkpoint import EMA, CheckpointManager, ResumeState
from cascade_dit.config import (
    TOKEN_DIM,
    DDTAttentionConfig,
    DDTPathConfig,
    ExperimentConfig,
    ModelConfig,
    RepresentationSourceConfig,
    RepresentationStageConfig,
)
from cascade_dit.data import data_fingerprint
from cascade_dit.model import DDT, has_lora
from cascade_dit.representations import ChannelStatistics, RepresentationManifest
from cascade_dit.rng import RNGStreams
from cascade_dit.runner import ExperimentRunner
from cascade_dit.validation import create_validation_plan


class RGBOracle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights: list[torch.Tensor] = []

    def forward(
        self,
        values: torch.Tensor,
        _timestep: torch.Tensor,
        _labels: torch.Tensor,
        representation_weights: torch.Tensor,
    ) -> torch.Tensor:
        self.weights.append(representation_weights.detach().cpu())
        return torch.zeros_like(values)


def test_early_stage_pixel_evaluation_uses_rgb_condition_and_prunes_archive(
    tmp_path, monkeypatch
) -> None:
    config = ExperimentConfig()
    config.experiment.output_directory = str(tmp_path)
    config.model.conditioning.num_classes = 2
    config.validation.periodic_samples = 2
    config.evaluation.num_samples = 4
    config.sampling.steps = 1
    config.sampling.batch_size = 1
    config.compile.enabled = False

    runner = object.__new__(ExperimentRunner)
    runner.config = config
    runner.rank = 0
    runner.world_size = 1
    runner.device = torch.device("cpu")
    runner.validation_plan = create_validation_plan(4, 2, 2, config.sampling.seed)
    runner.rgb_statistics = ChannelStatistics(
        torch.zeros(TOKEN_DIM), torch.ones(TOKEN_DIM), 2
    )
    runner.ema = SimpleNamespace(state_dict=lambda: {})

    model = RGBOracle()
    archive, output, sample_count, cached = runner._generate_pixel_evaluation(
        model, completed_epoch=1, training_complete=False
    )
    assert archive is not None and archive.is_file()
    assert sample_count == 2
    assert cached is None
    assert model.weights
    for weights in model.weights:
        assert torch.all(weights[:, :-1] == 0)
        assert torch.all(weights[:, -1] == 1)

    monkeypatch.setattr(
        runner_module,
        "evaluate_npz",
        lambda *_args, **_kwargs: {"FID": 12.5, "IS": 3.0},
    )
    metrics = runner._evaluate_pixel_archive(archive, output, sample_count, cached)
    assert metrics == {"FID": 12.5, "IS": 3.0}
    assert not archive.exists()
    assert (output / "preview.png").is_file()
    evaluation = json.loads((output / "evaluation.json").read_text())
    assert evaluation["num_samples"] == 2
    assert evaluation["sampling_plan_hash"] == runner.validation_plan.plan_hash

    second_archive, _, _, second_metrics = runner._generate_pixel_evaluation(
        model, completed_epoch=1, training_complete=False
    )
    assert second_archive is None
    assert second_metrics == metrics


def test_pixel_lora_resume_uses_active_config_and_checkpoint_tensor_width(
    tmp_path, monkeypatch
) -> None:
    source = ExperimentConfig()
    source.representations.stages = [
        RepresentationStageConfig("deep", 1, RepresentationSourceConfig("dino", 12)),
        RepresentationStageConfig("shallow", 1, RepresentationSourceConfig("dino", 8)),
    ]
    source.pixel_training.epochs = 3
    source.model = ModelConfig(
        encoder=DDTPathConfig(16, 1, 1),
        decoder=DDTPathConfig(32, 1, 2),
        attention=DDTAttentionConfig(rope=False),
    )
    source.model.conditioning.num_classes = 4
    model = DDT(source.model, 3)
    manager = CheckpointManager(tmp_path / "source")
    state = ResumeState(
        schema_version=4,
        completed_epoch=2,
        global_step=11,
        stage_index=2,
        stage="rgb",
        epoch_within_stage=0,
        transition_state="after",
        training_complete=False,
        images_seen=32,
        estimated_flops=64,
        wall_clock_seconds=5.0,
        optimizer=None,
        scheduler=None,
        rng_streams=RNGStreams.create(4).state_dict(),
        data_fingerprint=data_fingerprint(source.data),
        representation_fingerprint="representations",
    )
    checkpoint = manager.save_boundary(
        target_stage_index=2,
        position="after",
        model=model,
        ema=EMA(model),
        state=state,
        config=source,
    )

    active = ExperimentConfig()
    active.pixel_training.use_representation_guiding = False
    active.pixel_training.lora.use_lora = True
    active.pixel_training.epochs = 3
    active.representations.stages = []
    active.model = source.model
    active.experiment.output_directory = str(tmp_path / "active")
    active.validation.periodic_samples = 4
    active.evaluation.num_samples = 4
    active.compile.enabled = False
    manifest = RepresentationManifest(
        dataset_size=1,
        dataset_fingerprint=data_fingerprint(active.data),
        encoder_fingerprint="ignored",
        preparation_fingerprint="ignored",
        stages=[{"name": "rgb", "source": {"kind": "rgb", "layer": None}}],
        statistics={
            "rgb": {
                "mean": torch.zeros(TOKEN_DIM).tolist(),
                "std": torch.ones(TOKEN_DIM).tolist(),
                "count": 2,
            }
        },
        cache_files={},
        completion_files={},
        complete=True,
    )
    monkeypatch.setattr(
        runner_module,
        "load_representation_manifest",
        lambda _config: (manifest, "representations"),
    )

    with pytest.warns(UserWarning, match="active training configuration overrides"):
        runner = ExperimentRunner(active, resume=checkpoint)

    assert runner.representation_levels == 3
    assert runner.completed_epoch == 2
    assert runner.pixel_completed_epoch == 0
    assert has_lora(runner.model)
    assert all(
        "adapter_" in name
        for name, parameter in runner.model.named_parameters()
        if parameter.requires_grad
    )
