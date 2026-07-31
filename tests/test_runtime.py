from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from self_flow.checkpoint import EMA, CheckpointManager
from self_flow.config import ExperimentConfig, ModelConfig, ObjectiveConfig
from self_flow.latent_cache import latent_cache_fingerprint, latent_cache_spec
from self_flow.model import SiT, SiTOutput
from self_flow.rng import RNGStreams
from self_flow.sampling import ode_sample
from self_flow.validation import create_validation_plan


class ZeroModel(nn.Module):
    config = ModelConfig(
        image_size=32, patch_size=1, hidden_size=32, depth=1, num_heads=4, num_classes=2
    )

    def forward(self, x: torch.Tensor, _t: torch.Tensor, _labels: torch.Tensor) -> SiTOutput:
        return SiTOutput(torch.zeros_like(x), ())


def test_euler_and_heun_sampling_shapes() -> None:
    noise = torch.randn(2, 128, 2, 2)
    labels = torch.tensor([0, 1])
    for solver in ("euler", "heun"):
        output = ode_sample(ZeroModel(), noise, labels, steps=3, solver=solver, num_classes=2)
        assert output.shape == noise.shape
        assert torch.isfinite(output).all()


def test_latent_cache_identity_is_independent_of_objective() -> None:
    vanilla = ExperimentConfig()
    sigreg = ExperimentConfig(objective=ObjectiveConfig(kind="sigreg", align_layers=[4, 8]))
    assert latent_cache_fingerprint(latent_cache_spec(vanilla, 100)) == latent_cache_fingerprint(
        latent_cache_spec(sigreg, 100)
    )


def test_checkpoint_round_trip_with_callable_ema(tmp_path: Path) -> None:
    config = ExperimentConfig(
        model=ModelConfig(
            image_size=32, patch_size=1, hidden_size=32, depth=1, num_heads=4, num_classes=2
        ),
        objective=ObjectiveConfig(kind="ema", align_layers=[1], projector_hidden_size=48),
    )
    config.validation.periodic_samples = 2
    config.validation.final_samples = 2
    config.sampling.num_samples = 2
    model = SiT(config.model, align_layers=[1], projector_hidden_size=48)
    ema = EMA(model)
    plan_path = tmp_path / "validation-plan.pt"
    plan = create_validation_plan(2, 2, 2, 7)
    plan.save(plan_path)
    manager = CheckpointManager(tmp_path / "checkpoints")
    streams = RNGStreams.create(3)
    saved = manager.save(
        completed_epoch=1,
        global_step=2,
        model=model,
        ema=ema,
        optimizer_state={"optimizers": [], "implementations": []},
        rng_streams=streams.state_dict(),
        data_fingerprint="data",
        validation_plan_hash=plan.plan_hash,
        validation_plan_path=plan_path,
        last_completed_validation_epoch=None,
        wandb_run_id=None,
        resolved_config=config.as_dict(),
    )
    restored_model = SiT(config.model, align_layers=[1], projector_hidden_size=48)
    restored_ema = EMA(restored_model)
    state = manager.load(saved, restored_model, restored_ema, "data", plan.plan_hash)
    assert state.epoch == 1 and state.global_step == 2
    x = torch.randn(2, 128, 2, 2)
    output = restored_ema.model(x, torch.rand(2), torch.tensor([0, 1]))
    assert output.x0 is not None
