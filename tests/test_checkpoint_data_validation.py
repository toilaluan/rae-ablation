from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import TensorDataset

from jit_reg.checkpoint import EMA, CheckpointManager, should_checkpoint
from jit_reg.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    REGConfig,
    WandbConfig,
)
from jit_reg.data import ImageNetDataset, data_fingerprint, random_horizontal_flip
from jit_reg.model.sit import SiT
from jit_reg.optim import build_optimizer
from jit_reg.reg import REGTargets
from jit_reg.rng import RNGStreams, preserve_global_rng
from jit_reg.sampling import class_balanced_labels
from jit_reg.training import make_train_loader, train
from jit_reg.validation import (
    ValidationPlan,
    create_validation_plan,
    noise_for_samples,
    should_validate,
)
from jit_reg.wandb_utils import RunLogger, WandbLogger


def _tiny_model() -> SiT:
    return SiT(
        ModelConfig(
            image_size=16,
            patch_size=8,
            hidden_size=16,
            depth=2,
            num_heads=4,
            mlp_ratio=2,
            num_classes=10,
        )
    )


def test_pinned_dataset_adapter_fingerprint_and_device_flip() -> None:
    rows = [
        {"image": Image.fromarray(np.full((8, 8, 3), 128, np.uint8)), "label": 7}
    ]
    dataset = ImageNetDataset(rows, 16)
    image, label, index = dataset[0]
    assert image.shape == (3, 16, 16)
    assert label.item() == 7 and index.item() == 0
    config = DataConfig()
    first = data_fingerprint(config, 256)
    assert first == data_fingerprint(config, 256)
    assert first != data_fingerprint(config, 128)
    values = torch.arange(12).reshape(1, 1, 3, 4)
    assert torch.equal(
        random_horizontal_flip(values, torch.Generator().manual_seed(1), 1),
        values.flip(-1),
    )


def test_validation_plan_noise_and_rng_stream_resume(tmp_path: Path) -> None:
    first = create_validation_plan(20, 10, 2, seed=9)
    second = create_validation_plan(20, 10, 2, seed=9)
    assert first.plan_hash == second.plan_hash
    assert set(first.__dict__) == {"labels", "sample_ids", "noise_seeds", "plan_hash"}
    path = tmp_path / "plan.pt"
    first.save(path)
    assert ValidationPlan.load(path).plan_hash == first.plan_hash
    whole = noise_for_samples(first.noise_seeds[:4], (1, 2, 2), "cpu", torch.float32)
    split = torch.cat(
        [
            noise_for_samples(first.noise_seeds[:1], (1, 2, 2), "cpu", torch.float32),
            noise_for_samples(first.noise_seeds[1:4], (1, 2, 2), "cpu", torch.float32),
        ]
    )
    assert torch.equal(whole, split)
    streams = RNGStreams.create(12)
    saved = streams.state_dict()
    torch.rand(3, generator=streams.augmentations)
    streams.load_state_dict(saved)
    assert torch.equal(streams.augmentations.get_state(), saved["augmentations"])
    state = torch.random.get_rng_state()
    with preserve_global_rng():
        torch.randn(100)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_class_balance_and_epoch_cadence() -> None:
    labels = class_balanced_labels(50_000, 1_000)
    assert torch.equal(torch.bincount(labels), torch.full((1_000,), 50))
    assert not should_validate(1, 10, 0)
    assert should_validate(10, 10, 0)
    assert not should_checkpoint(3, 10, 0)
    assert should_checkpoint(10, 10, 0)


def test_checkpoint_resume_retention_best_and_no_scheduler(tmp_path: Path) -> None:
    model = _tiny_model()
    ema = EMA(model)
    manager = CheckpointManager(tmp_path, keep_last_n=2)
    streams = RNGStreams.create(1)
    kwargs = dict(
        global_step=1,
        model=model,
        ema=ema,
        optimizer_state={"optimizers": []},
        rng_streams=streams.state_dict(),
        data_fingerprint="data",
        validation_plan_hash="plan",
        last_completed_validation_epoch=None,
        wandb_run_id="run",
        resolved_config={"model": {"variant": "SiT-Tiny"}},
    )
    manager.save(completed_epoch=1, fid=10.0, **kwargs)
    manager.save(completed_epoch=2, fid=12.0, **kwargs)
    manager.save(completed_epoch=3, fid=8.0, **kwargs)
    assert not (tmp_path / "epoch-0001").exists()
    assert (tmp_path / "last").resolve().name == "epoch-0003"
    assert (tmp_path / "best-fid").resolve().name == "epoch-0003"
    weights = load_file(str(tmp_path / "epoch-0003" / "model.safetensors"))
    assert not any("_orig_mod" in key for key in weights)
    training = torch.load(
        tmp_path / "epoch-0003" / "training.pt", weights_only=False
    )
    assert "scheduler" not in training and "data_fingerprint" in training
    restored = _tiny_model()
    resumed = manager.load(
        tmp_path / "epoch-0003", restored, EMA(restored), "data", "plan"
    )
    assert resumed.epoch == 3 and resumed.wandb_run_id == "run"
    with pytest.raises(ValueError, match="fingerprint"):
        manager.load(tmp_path / "epoch-0003", restored, EMA(restored), "other", "plan")
    assert json.loads((tmp_path / "epoch-0003" / "config.json").read_text())


class _TinyTarget(nn.Module):
    def targets(self, images: torch.Tensor, target_grid: int) -> REGTargets:
        cls = images.mean(dim=(2, 3)).repeat(1, 3)[:, :8]
        spatial = torch.ones(images.shape[0], target_grid**2, 8)
        spatial = torch.nn.functional.normalize(spatial, dim=-1)
        return REGTargets(cls, spatial, torch.cat((cls[:, None], spatial), dim=1))


@pytest.mark.parametrize("reg_enabled", [False, True])
def test_one_step_pixel_training_smoke(tmp_path: Path, reg_enabled: bool) -> None:
    config = ExperimentConfig()
    config.model = ModelConfig(
        image_size=16,
        patch_size=8,
        hidden_size=16,
        depth=1,
        num_heads=4,
        mlp_ratio=2,
        num_classes=2,
        reg=REGConfig(
            enabled=reg_enabled,
            feature_dim=8,
            repa_block=1,
            repa_hidden_size=16,
        ),
    )
    config.train.optimizer.kind = "adamw"
    config.train.epochs = 1
    config.train.global_batch_size = 2
    config.train.micro_batch_size = 2
    config.train.max_steps = 1
    config.data.workers = 0
    config.validation.interval = 0
    config.checkpoint.directory = str(tmp_path / "checkpoints")
    pixels = torch.rand(2, 3, 16, 16)
    dataset = TensorDataset(pixels, torch.tensor([0, 1]), torch.arange(2))
    streams = RNGStreams.create(3)
    loader, sampler = make_train_loader(dataset, config, streams)
    model = SiT(config.model)
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
        validation_plan_hash="plan",
        logger=RunLogger(WandbConfig(), {}, True, log_interval=1),
        reg_encoder=_TinyTarget() if reg_enabled else None,  # type: ignore[arg-type]
    )
    assert state.global_step == 1
    assert (tmp_path / "checkpoints" / "epoch-0001" / "training.pt").is_file()


def test_wandb_is_rank_zero_only_and_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    class Run:
        id = "existing"

        def log(self, values: dict[str, object], step: int | None = None) -> None:
            calls.append({"values": values, "step": step})

        def finish(self) -> None:
            calls.append({"finished": True})

        def log_artifact(self, artifact: object) -> None:
            calls.append({"artifact": artifact})

    fake = SimpleNamespace(
        init=lambda **kwargs: (calls.append(kwargs), Run())[1],
        Image=lambda path, caption: (path, caption),
        Artifact=lambda **kwargs: SimpleNamespace(add_dir=lambda path: None, **kwargs),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    config = WandbConfig(enabled=True)
    assert WandbLogger(config, {}, False, "existing").run is None
    logger = WandbLogger(config, {"reg": False}, True, "existing")
    assert calls[0]["id"] == "existing" and calls[0]["resume"] == "allow"
    logger.log_training(
        {
            "train/epoch": 1.0,
            "train/loss": 2.0,
            "train/image_velocity_loss": 1.0,
            "train/cls_velocity_loss": 0.0,
            "train/repa_loss": 0.0,
            "train/lr": 1e-4,
            "train/gradient_norm": 0.5,
            "throughput/images_per_second": 10.0,
            "throughput/step_seconds": 0.1,
            "system/peak_memory_gib": 1.0,
        },
        step=1,
    )
    logger.log_grid(str(tmp_path / "grid.png"), "classes", 1)
    assert any("values" in call for call in calls)
