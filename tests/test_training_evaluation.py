from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

import cascade_dit.runner as runner_module
from cascade_dit.config import TOKEN_DIM, ExperimentConfig
from cascade_dit.model import DDTOutput
from cascade_dit.representations import ChannelStatistics
from cascade_dit.runner import ExperimentRunner
from cascade_dit.validation import create_validation_plan


class ModeOracle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        values: torch.Tensor,
        timestep: torch.Tensor,
        _labels: torch.Tensor,
        pixel_ratio: torch.Tensor,
        *,
        prediction_mode: str,
    ) -> DDTOutput:
        self.calls.append(
            (prediction_mode, timestep.detach().cpu(), pixel_ratio.detach().cpu())
        )
        prediction = torch.zeros_like(values)
        return DDTOutput(prediction)


def _evaluation_runner(tmp_path) -> ExperimentRunner:
    config = ExperimentConfig()
    config.experiment.output_directory = str(tmp_path)
    config.model.conditioning.num_classes = 2
    config.validation.periodic_samples = 2
    config.evaluation.num_samples = 4
    config.sampling.steps = 2
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
    return runner


def test_semantic_evaluation_uses_only_semantic_trajectory_then_one_pixel_decode(
    tmp_path,
) -> None:
    runner = _evaluation_runner(tmp_path)
    model = ModeOracle()
    archive, _output, count, cached = runner._generate_evaluation(
        model,
        completed_epoch=1,
        training_complete=False,
        phase="semantic",
    )
    assert archive is not None and archive.is_file()
    assert count == 2 and cached is None
    semantic = [call for call in model.calls if call[0] == "semantic"]
    pixel = [call for call in model.calls if call[0] == "pixel"]
    assert len(semantic) == 4
    assert len(pixel) == 2
    assert all(torch.all(call[2] == 0) for call in model.calls)
    assert all(torch.all(call[1] == 0) for call in pixel)


def test_pixel_evaluation_uses_ratio_one_pixel_trajectory(tmp_path) -> None:
    runner = _evaluation_runner(tmp_path)
    model = ModeOracle()
    runner._generate_evaluation(
        model,
        completed_epoch=2,
        training_complete=False,
        phase="pixel",
    )
    assert model.calls
    assert all(mode == "pixel" for mode, _time, _ratio in model.calls)
    assert all(torch.all(ratio == 1) for _mode, _time, ratio in model.calls)


def test_runner_reuses_loader_and_rebuilds_only_wrapper_at_pixel_boundary(
    monkeypatch,
) -> None:
    config = ExperimentConfig()
    config.training.semantic.epochs = 1
    config.training.pixel.epochs = 2
    config.validation.interval_epochs = 0
    config.checkpoint.interval_epochs = 0
    config.checkpoint.save_phase_boundary = False

    class OneBatchLoader:
        def __len__(self) -> int:
            return 1

        def __iter__(self):
            return iter([None])

    loader = OneBatchLoader()
    builds: list[str] = []
    advances: list[tuple[str, int]] = []
    monkeypatch.setattr(
        runner_module,
        "make_endpoint_loader",
        lambda *_args, **_kwargs: loader,
    )
    monkeypatch.setattr(
        runner_module,
        "set_endpoint_loader_epoch",
        lambda _loader, phase, epoch: advances.append((phase, epoch)),
    )
    monkeypatch.setattr(runner_module, "model_flops_per_step", lambda *_args: 0)

    runner = object.__new__(ExperimentRunner)
    runner.config = config
    runner.rank = 0
    runner.world_size = 1
    runner.device = torch.device("cpu")
    runner.completed_epoch = 0
    runner.global_step = 0
    runner.images_seen = 0
    runner.estimated_flops = 0
    runner.model = nn.Linear(1, 1, bias=False)
    runner.optimizer = torch.optim.SGD(runner.model.parameters(), lr=0.1)
    optimizer_identity = id(runner.optimizer)
    runner.scheduler = torch.optim.lr_scheduler.LambdaLR(
        runner.optimizer, lr_lambda=lambda _step: 1.0
    )
    runner.ema = SimpleNamespace(update=lambda _model: None)
    runner.rng = object()
    runner.log = lambda _metrics: None
    runner._write_run_metrics = lambda **_kwargs: None
    runner._build_execution = lambda _steps, phase: (
        builds.append(phase) or runner.model
    )
    runner.batch_preprocessor = lambda _batch: SimpleNamespace(
        clean=torch.ones(1, 1, 1),
        pixel=torch.ones(1, 1, 1),
        labels=torch.zeros(1, dtype=torch.int64),
        pixel_ratio=torch.ones(1),
    )

    def training_loss(execution, *_args, **_kwargs):
        total = execution.weight.square().sum()
        zero = total.detach() * 0
        return SimpleNamespace(
            loss=SimpleNamespace(
                total=total,
                velocity_mse=total.detach(),
                pixel_auxiliary_mse=zero,
                weighted_pixel_auxiliary_mse=zero,
                pure_semantic_count=0,
            )
        )

    monkeypatch.setattr(runner_module, "compute_training_loss", training_loss)
    runner.run(max_steps=3)

    assert builds == ["semantic", "pixel"]
    assert advances == [("pixel", 0), ("pixel", 1)]
    assert id(runner.optimizer) == optimizer_identity
