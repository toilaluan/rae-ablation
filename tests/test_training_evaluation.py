from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

import cascade_dit.runner as runner_module
from cascade_dit.config import ExperimentConfig
from cascade_dit.model import DDTOutput
from cascade_dit.runner import ExperimentRunner
from cascade_dit.validation import create_validation_plan


class JointOracle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, bool, int]] = []

    def forward(
        self,
        values: torch.Tensor,
        timestep: torch.Tensor,
        _labels: torch.Tensor,
        *,
        decode_pixels: bool,
    ) -> DDTOutput:
        self.calls.append((timestep.detach().cpu(), decode_pixels, values.shape[0]))
        prediction = torch.zeros_like(values)
        pixels = (
            torch.zeros(values.shape[0], 3, 128, 128)
            if decode_pixels
            else None
        )
        return DDTOutput(prediction, prediction, pixels)


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
    runner.ema = SimpleNamespace(state_dict=lambda: {})
    return runner


def test_evaluation_integrates_semantics_then_decodes_pixels_once(tmp_path) -> None:
    runner = _evaluation_runner(tmp_path)
    model = JointOracle()
    archive, _output, count, cached = runner._generate_evaluation(
        model,
        completed_epoch=1,
        training_complete=False,
    )
    assert archive is not None and count == 2 and cached is None
    semantic_calls = [call for call in model.calls if not call[1]]
    pixel_calls = [call for call in model.calls if call[1]]
    assert len(semantic_calls) == count * runner.config.sampling.steps
    assert len(pixel_calls) == count
    assert all(batch == 1 for _time, _decode, batch in model.calls)
    assert all(torch.equal(time, torch.zeros(1)) for time, _decode, _batch in pixel_calls)


def test_runner_reuses_one_loader_and_one_execution_wrapper(monkeypatch) -> None:
    config = ExperimentConfig()
    config.training.epochs = 3
    config.validation.interval_epochs = 0
    config.checkpoint.interval_epochs = 0

    class OneBatchLoader:
        def __len__(self) -> int:
            return 1

        def __iter__(self):
            return iter([None])

    loader = OneBatchLoader()
    builds: list[int] = []
    advances: list[int] = []
    monkeypatch.setattr(runner_module, "make_joint_loader", lambda *_args, **_kwargs: loader)
    monkeypatch.setattr(
        runner_module,
        "set_joint_loader_epoch",
        lambda _loader, epoch: advances.append(epoch),
    )
    monkeypatch.setattr(
        runner_module, "model_flops_per_step", lambda *_args, **_kwargs: 0
    )

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
    runner.scheduler = torch.optim.lr_scheduler.LambdaLR(
        runner.optimizer, lr_lambda=lambda _step: 1.0
    )
    runner.ema = SimpleNamespace(update=lambda _model: None)
    runner.rng = object()
    runner.lpips_model = nn.Identity()
    runner.log = lambda _metrics: None
    runner._write_run_metrics = lambda **_kwargs: None
    runner._build_execution = lambda steps: (builds.append(steps) or runner.model)
    runner.batch_preprocessor = lambda _batch: SimpleNamespace(
        semantic=torch.ones(1, 1, 1),
        pixels=torch.ones(1, 3, 1, 1),
        labels=torch.zeros(1, dtype=torch.int64),
    )

    def training_loss(execution, *_args, **_kwargs):
        total = execution.weight.square().sum()
        zero = total.detach() * 0
        return SimpleNamespace(
            loss=SimpleNamespace(
                total=total,
                full_velocity_mse=total.detach(),
                base_velocity_mse=zero,
                pixel_l1=zero,
                pixel_lpips=zero,
            )
        )

    monkeypatch.setattr(runner_module, "compute_training_loss", training_loss)
    runner.run(max_steps=3)

    assert builds == [1]
    assert advances == [0, 1, 2]
