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
from jit_reg.checkpoint_bundle import load_checkpoint_bundle
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
    rows = [{"image": Image.fromarray(np.full((8, 8, 3), 128, np.uint8)), "label": 7}]
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
    legacy = {**saved, "cls_noise": saved["reg_noise"]}
    del legacy["reg_noise"]
    streams.load_state_dict(legacy)
    assert torch.equal(streams.reg_noise.get_state(), saved["reg_noise"])
    legacy["reg_timesteps"] = torch.Generator().manual_seed(99).get_state()
    streams.load_state_dict(legacy)
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
    plan = create_validation_plan(20, 10, 2, seed=9)
    plan.save(tmp_path / "validation-plan.pt")
    kwargs = dict(
        global_step=1,
        model=model,
        ema=ema,
        optimizer_state={"optimizers": []},
        rng_streams=streams.state_dict(),
        data_fingerprint="data",
        validation_plan_hash=plan.plan_hash,
        validation_plan_path=tmp_path / "validation-plan.pt",
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
    training = torch.load(tmp_path / "epoch-0003" / "training.pt", weights_only=False)
    assert "scheduler" not in training and "data_fingerprint" in training
    restored = _tiny_model()
    resumed = manager.load(
        tmp_path / "epoch-0003", restored, EMA(restored), "data", plan.plan_hash
    )
    assert resumed.epoch == 3 and resumed.wandb_run_id == "run"
    with pytest.raises(ValueError, match="fingerprint"):
        manager.load(
            tmp_path / "epoch-0003", restored, EMA(restored), "other", plan.plan_hash
        )
    assert json.loads((tmp_path / "epoch-0003" / "config.json").read_text())
    assert (tmp_path / "epoch-0003" / "validation-plan.pt").is_file()
    assert json.loads((tmp_path / "epoch-0003" / "metadata.json").read_text())


def test_checkpoint_bundle_is_self_contained_and_supports_legacy_layout(tmp_path: Path) -> None:
    model = _tiny_model()
    config = ExperimentConfig()
    config.model = model.config
    config.validation.final_samples = 20
    config.validation.periodic_samples = 10
    config.validation.plan_path = "validation-plan.pt"
    config.compile.enabled = False
    config.validate()
    plan = create_validation_plan(20, 10, 10, seed=config.train.seed + 7)
    plan_path = tmp_path / "validation-plan.pt"
    plan.save(plan_path)
    manager = CheckpointManager(tmp_path / "checkpoints")
    checkpoint = manager.save(
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
    bundle = load_checkpoint_bundle(tmp_path / "checkpoints")
    assert bundle.directory == checkpoint.resolve()
    assert bundle.weights.name == "ema.safetensors"
    assert bundle.plan.plan_hash == plan.plan_hash

    (checkpoint / "metadata.json").unlink()
    (checkpoint / "validation-plan.pt").unlink()
    saved_config = json.loads((checkpoint / "config.json").read_text())
    saved_config["train"]["reg_relative_timestep_shift_min"] = 0.5
    saved_config["train"]["reg_relative_timestep_shift_max"] = 8.0
    (checkpoint / "config.json").write_text(json.dumps(saved_config))
    legacy = load_checkpoint_bundle(checkpoint)
    assert legacy.plan.plan_hash == plan.plan_hash


def test_checkpoint_retention_protects_sampled_epochs(tmp_path: Path) -> None:
    model = _tiny_model()
    manager = CheckpointManager(tmp_path / "checkpoints", keep_last_n=1)
    plan = create_validation_plan(20, 10, 10, seed=7)
    plan_path = tmp_path / "plan.pt"
    plan.save(plan_path)
    kwargs = {
        "global_step": 1,
        "model": model,
        "ema": EMA(model),
        "optimizer_state": {"optimizers": []},
        "rng_streams": RNGStreams.create(1).state_dict(),
        "data_fingerprint": "data",
        "validation_plan_hash": plan.plan_hash,
        "validation_plan_path": plan_path,
        "last_completed_validation_epoch": None,
        "wandb_run_id": None,
        "resolved_config": ExperimentConfig().as_dict(),
    }
    first = manager.save(completed_epoch=1, **kwargs)
    (first / "samples-euler-test").mkdir()
    manager.save(completed_epoch=2, **kwargs)
    manager.save(completed_epoch=3, **kwargs)
    assert first.is_dir()
    assert not (manager.directory / "epoch-0002").exists()
    assert (manager.directory / "epoch-0003").is_dir()


class _TinyTarget(nn.Module):
    def __init__(self, token_mode: str) -> None:
        super().__init__()
        self.token_mode = token_mode

    def targets(self, images: torch.Tensor, target_grid: int) -> REGTargets:
        cls = images.mean(dim=(2, 3)).repeat(1, 3)[:, :8]
        spatial = torch.ones(images.shape[0], target_grid**2, 8)
        spatial = torch.nn.functional.normalize(spatial, dim=-1)
        reg_tokens = cls[:, None]
        if self.token_mode == "cls_spatial":
            reg_tokens = torch.cat((reg_tokens, torch.zeros_like(spatial)), dim=1)
        return REGTargets(reg_tokens, spatial)


class _TinyCache:
    fingerprint = "tiny-cache"

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return torch.zeros(indices.numel(), 1, 8, dtype=torch.bfloat16)


@pytest.mark.parametrize(
    ("reg_variant", "token_mode", "cached"),
    [
        ("disabled", "cls", False),
        ("standard", "cls", False),
        ("standard", "cls_spatial", False),
        ("wide_intermediate", "cls_spatial", False),
        ("standard", "cls", True),
    ],
)
def test_one_step_pixel_training_smoke(
    tmp_path: Path,
    reg_variant: str,
    token_mode: str,
    cached: bool,
) -> None:
    reg_enabled = reg_variant != "disabled"
    wide = reg_variant == "wide_intermediate"
    config = ExperimentConfig()
    config.model = ModelConfig(
        image_size=16,
        patch_size=8,
        hidden_size=16,
        depth=2 if wide else 1,
        num_heads=4,
        mlp_ratio=2,
        num_classes=2,
        reg=REGConfig(
            enabled=reg_enabled,
            repa_enabled=not cached,
            feature_dim=8,
            token_mode=token_mode,
            pool_size=2,
            repa_block=1,
            repa_hidden_size=16,
            cls_predictor=reg_variant if reg_enabled else "standard",
            cls_head_block=1,
            cls_head_hidden_size=32,
            cls_head_depth=2,
            cls_head_mlp_ratio=2,
        ),
    )
    config.train.optimizer.kind = "adamw"
    config.train.epochs = 1
    config.train.global_batch_size = 2
    config.train.micro_batch_size = 2
    config.train.max_steps = 1
    config.train.horizontal_flip_probability = 0 if cached else 0.5
    config.reg_cache.enabled = cached
    config.data.workers = 0
    config.validation.interval = 0
    config.checkpoint.directory = str(tmp_path / "checkpoints")
    pixels = torch.rand(2, 3, 16, 16)
    dataset = TensorDataset(pixels, torch.tensor([0, 1]), torch.arange(2))
    streams = RNGStreams.create(3)
    loader, sampler = make_train_loader(dataset, config, streams)
    model = SiT(config.model)
    plan = create_validation_plan(4, 2, 2, seed=5)
    plan.save(tmp_path / "plan.pt")
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
        validation_plan_path=tmp_path / "plan.pt",
        logger=RunLogger(WandbConfig(), {}, True, log_interval=1),
        reg_encoder=(
            _TinyTarget(token_mode) if reg_enabled and not cached else None
        ),  # type: ignore[arg-type]
        reg_cache=_TinyCache() if cached else None,  # type: ignore[arg-type]
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
            "train/reg_velocity_loss": 0.0,
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
