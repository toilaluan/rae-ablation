from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from rae_ablation.cache import (
    AsyncShardWriter,
    PendingShard,
    compute_cache_statistics,
    expected_manifest,
    initialize_manifest,
    verify_cache,
)
from rae_ablation.checkpoint import EMA, CheckpointManager, should_checkpoint
from rae_ablation.config import CacheConfig, DepthAttentionConfig, ModelConfig, WandbConfig
from rae_ablation.latents import LatentStatistics
from rae_ablation.model.dit import DiffusionTransformer
from rae_ablation.rng import preserve_global_rng
from rae_ablation.validation import (
    ValidationPlan,
    create_validation_plan,
    noise_for_samples,
    should_validate,
)
from rae_ablation.wandb_utils import WandbLogger


def _tiny_model() -> DiffusionTransformer:
    return DiffusionTransformer(
        ModelConfig(
            in_channels=4,
            hidden_size=16,
            depth=2,
            num_heads=4,
            mlp_size=32,
            base_grid_size=2,
            early_exit_block=1,
            gradient_checkpointing=False,
            depth_attention=DepthAttentionConfig(True, 1),
        )
    )


def test_manifest_resume_rejection_and_exact_fp16_statistics(tmp_path: Path) -> None:
    config = CacheConfig(codec="rae_compact", directory=str(tmp_path), shard_size=2, batch_size=1)
    manifest = expected_manifest(config, 3)
    # Use a tiny shape in this format-only unit test.
    manifest.codecs["rae_compact"]["shape"] = [2, 1, 1]
    initialize_manifest(tmp_path, manifest)
    save_file(
        {
            "rae_compact": torch.tensor([[[[1]], [[2]]], [[[3]], [[4]]]], dtype=torch.float16),
            "labels": torch.tensor([0, 1]),
            "indices": torch.tensor([0, 1]),
        },
        str(tmp_path / "shard-000000.safetensors"),
    )
    save_file(
        {
            "rae_compact": torch.tensor([[[[5]], [[6]]]], dtype=torch.float16),
            "labels": torch.tensor([2]),
            "indices": torch.tensor([2]),
        },
        str(tmp_path / "shard-000001.safetensors"),
    )
    assert verify_cache(tmp_path) == [0, 1]
    manifest.completed_shards = [0, 1]
    compute_cache_statistics(tmp_path, manifest)
    stats = LatentStatistics.load(tmp_path / "stats-rae_compact.safetensors")
    assert torch.equal(stats.mean.flatten(), torch.tensor([3.0, 4.0]))
    expected_std = torch.tensor([8 / 3, 8 / 3]).sqrt()
    assert torch.allclose(stats.std.flatten(), expected_std)
    different = expected_manifest(config, 4)
    with pytest.raises(ValueError, match="manifest"):
        initialize_manifest(tmp_path, different)


def test_background_writer_is_atomic_and_deterministic(tmp_path: Path) -> None:
    writer = AsyncShardWriter(tmp_path, max_pending=1)
    writer.submit(
        PendingShard(
            0,
            {
                "rae": [torch.tensor([[[[1.0]]]], dtype=torch.float16)],
                "labels": [torch.tensor([7])],
                "indices": [torch.tensor([3])],
            },
            [],
        )
    )
    writer.close()
    assert not list(tmp_path.glob("*.partial"))
    stored = load_file(str(tmp_path / "shard-000000.safetensors"))
    assert stored["indices"].item() == 3


def test_validation_plan_subset_noise_and_rng_preservation(tmp_path: Path) -> None:
    first = create_validation_plan(20, 10, 2, seed=9, channels=4)
    second = create_validation_plan(20, 10, 2, seed=9, channels=4)
    assert first.plan_hash == second.plan_hash
    assert torch.equal(first.periodic()[0], first.labels[:10])
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
    torch.manual_seed(12)
    state = torch.random.get_rng_state()
    with preserve_global_rng():
        torch.randn(100)
        random.random()
        np.random.rand()
    assert torch.equal(torch.random.get_rng_state(), state)


def test_epoch_cadence_and_no_duplicate_validation() -> None:
    assert not should_validate(1, 10, 5)
    assert should_validate(5, 10, 5)
    assert should_validate(10, 10, 0)
    assert not should_validate(5, 10, 5, last_completed_validation_epoch=5)
    assert should_validate(5, 10, 5, last_completed_validation_epoch=5, validate_now=True)
    assert not should_checkpoint(3, 10, 0)
    assert should_checkpoint(10, 10, 0)


def test_atomic_checkpoint_clean_keys_resume_retention_and_best(tmp_path: Path) -> None:
    model = _tiny_model()
    ema = EMA(model)
    manager = CheckpointManager(tmp_path, keep_last_n=2)
    kwargs = dict(
        global_step=1,
        model=model,
        ema=ema,
        optimizer_state={"optimizers": []},
        scheduler_state={"epoch": 1},
        rng_streams={"noise": torch.get_rng_state()},
        cache_fingerprint="cache",
        validation_plan_hash="plan",
        last_completed_validation_epoch=None,
        wandb_run_id="run",
        resolved_config={"codec": "rae"},
    )
    manager.save(completed_epoch=1, fid=10.0, **kwargs)
    manager.save(completed_epoch=2, fid=12.0, **kwargs)
    manager.save(completed_epoch=3, fid=8.0, **kwargs)
    assert not (tmp_path / "epoch-0001").exists()
    assert (tmp_path / "epoch-0002").exists() and (tmp_path / "epoch-0003").exists()
    assert (tmp_path / "last").resolve().name == "epoch-0003"
    assert (tmp_path / "best-fid").resolve().name == "epoch-0003"
    state = load_file(str(tmp_path / "epoch-0003" / "model.safetensors"))
    assert not any("_orig_mod" in key for key in state)
    restored = _tiny_model()
    restored_ema = EMA(restored)
    resumed = manager.load(tmp_path / "epoch-0003", restored, restored_ema, "cache", "plan")
    assert resumed.epoch == 3 and resumed.wandb_run_id == "run"


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

    def init(**kwargs: object) -> Run:
        calls.append(kwargs)
        return Run()

    fake = SimpleNamespace(
        init=init,
        Image=lambda path, caption: (path, caption),
        Artifact=lambda **kwargs: SimpleNamespace(add_dir=lambda path: None, **kwargs),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)
    config = WandbConfig(enabled=True)
    non_main = WandbLogger(config, {}, False, "existing")
    assert non_main.run is None and not calls
    logger = WandbLogger(config, {"codec": "rae"}, True, "existing")
    assert calls[0]["id"] == "existing" and calls[0]["resume"] == "allow"
    logger.log({"train/loss_velocity_full": 1.0}, step=2)
    logger.log_grid(str(tmp_path / "grid.png"), "classes", 2)
    assert any("values" in call for call in calls)
