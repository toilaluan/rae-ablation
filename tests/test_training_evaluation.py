from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from torch import nn

import cascade_dit.runner as runner_module
from cascade_dit.config import TOKEN_DIM, ExperimentConfig
from cascade_dit.representations import ChannelStatistics
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
