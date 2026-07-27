from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from jit_reg.config import ExperimentConfig, load_checkpoint_config, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_required_presets_load() -> None:
    baseline = load_config(ROOT / "configs/baseline-p16.yaml")
    paper = load_config(ROOT / "configs/reg-repa-p16.yaml")
    early = load_config(ROOT / "configs/early-reg-p16.yaml")
    assert baseline.schema_version == paper.schema_version == early.schema_version == 4
    assert (
        baseline.train.optimizer.kind
        == paper.train.optimizer.kind
        == early.train.optimizer.kind
        == "muon"
    )
    assert baseline.model.patch_size == paper.model.patch_size == early.model.patch_size == 16
    assert not baseline.model.features.enabled
    assert paper.model.features.reg.num_tokens == 16
    assert paper.model.features.reg.target_std == pytest.approx(0.53)
    assert paper.model.features.repa.token_scope == "image_and_reg"
    assert paper.model.features.reg.early_prediction.block is None
    assert early.model.features.reg.spatial_rope
    assert early.model.features.reg.head.kind == "wide_mlp"
    assert early.model.features.reg.early_prediction.block == 4
    assert early.model.features.reg.early_prediction.reinjection_mode == "add"
    assert early.train.reg_timestep_loc == pytest.approx(-0.8)
    assert early.train.reg_timestep_scale == pytest.approx(0.8)
    assert early.sampling.img_denoising_interval == [0.3, 1.0]
    assert early.sampling.reg_denoising_interval == [0.0, 0.7]
    assert early.sampling.reg_timestep_shifts == [1.0]


@pytest.mark.parametrize("kind", ["adamw", "muon", "fused_muon"])
def test_all_optimizer_kinds_validate(kind: str) -> None:
    config = ExperimentConfig()
    config.train.optimizer.kind = kind
    config.validate()


def test_unknown_optimizer_kind_is_rejected() -> None:
    config = ExperimentConfig()
    config.train.optimizer.kind = "automatic"
    with pytest.raises(ValueError, match="adamw, muon, or fused_muon"):
        config.validate()


def test_retired_micro_batch_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="micro_batch_size"):
        load_config(
            ROOT / "configs/baseline-p16.yaml",
            ["train.micro_batch_size=8"],
        )


@pytest.mark.parametrize(
    ("reg", "repa", "scope"),
    [
        (False, False, "image"),
        (False, True, "image"),
        (True, False, "image"),
        (True, True, "image"),
        (True, True, "image_and_reg"),
    ],
)
def test_feature_combination_matrix(reg: bool, repa: bool, scope: str) -> None:
    config = ExperimentConfig()
    config.model.features.reg.enabled = reg
    config.model.features.repa.enabled = repa
    config.model.features.repa.token_scope = scope
    config.validate()


@pytest.mark.parametrize("head", ["standard", "wide_mlp"])
@pytest.mark.parametrize("block", [None, 1])
def test_reg_configuration_matrix(head: str, block: int | None) -> None:
    config = ExperimentConfig()
    config.model.depth = 2
    reg = config.model.features.reg
    reg.enabled = True
    reg.head.kind = head
    reg.early_prediction.block = block
    config.validate()


@pytest.mark.parametrize(("pool_size", "num_tokens"), [(1, 1), (4, 16), (16, 256)])
def test_reg_token_layout_properties(pool_size: int, num_tokens: int) -> None:
    reg = ExperimentConfig().model.features.reg
    reg.pool_size = pool_size
    assert reg.num_tokens == num_tokens


def test_inactive_feature_sections_are_not_validated() -> None:
    config = ExperimentConfig()
    config.model.features.encoder.model_name = ""
    config.model.features.encoder.feature_dim = -1
    config.model.features.reg.head.kind = "retired"
    config.model.features.reg.target_std = -1
    config.train.reg_timestep_scale = -1
    config.model.features.repa.block = -1
    config.model.features.repa.token_scope = "retired"
    config.feature_cache.enabled = True
    config.feature_cache.directory = ""
    config.feature_cache.batch_size = -1
    config.validate()


def test_active_sections_validate_only_their_own_fields() -> None:
    repa_only = ExperimentConfig()
    repa_only.model.features.repa.enabled = True
    repa_only.model.features.reg.head.kind = "irrelevant"
    repa_only.model.features.reg.target_std = -1
    repa_only.train.reg_timestep_scale = -1
    repa_only.validate()

    reg_only = ExperimentConfig()
    reg_only.model.features.reg.enabled = True
    reg_only.model.features.repa.block = -1
    reg_only.model.features.repa.token_scope = "irrelevant"
    reg_only.validate()


def test_repa_full_sequence_requires_reg() -> None:
    config = ExperimentConfig()
    config.model.features.repa.enabled = True
    config.model.features.repa.token_scope = "image_and_reg"
    with pytest.raises(ValueError, match="requires REG"):
        config.validate()


def test_clean_break_rejects_old_and_unknown_configs(tmp_path: Path) -> None:
    old = tmp_path / "old.yaml"
    old.write_text("model:\n  patch_size: 16\n")
    with pytest.raises(ValueError, match="clean break"):
        load_config(old)

    old_checkpoint = tmp_path / "config.json"
    old_checkpoint.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ValueError, match="cannot be migrated"):
        load_checkpoint_config(old_checkpoint)

    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("schema_version: 4\nretired_experiment: true\n")
    with pytest.raises(ValueError, match="invalid schema-v4"):
        load_config(unknown)


def test_sampling_intervals_and_reg_shift_candidates_validate() -> None:
    config = ExperimentConfig()
    config.sampling.img_denoising_interval = [0.3, 1.0]
    config.sampling.reg_denoising_interval = [0.0, 0.7]
    config.sampling.reg_timestep_shifts = [0.7, 1.0, 1.3]
    config.validate()

    config.sampling.reg_timestep_shifts = [1.0, 1.0]
    with pytest.raises(ValueError, match="unique"):
        config.validate()

    config.sampling.reg_timestep_shifts = [1.0]
    config.sampling.reg_denoising_interval = [0.7, 0.7]
    with pytest.raises(ValueError, match="start < end"):
        config.validate()

    config.sampling.reg_denoising_interval = [0.01, 0.02]
    config.sampling.sampling_steps = 10
    with pytest.raises(ValueError, match="collapses"):
        config.validate()


@pytest.mark.parametrize("target_std", [0.0, -1.0, float("inf"), float("nan")])
def test_active_reg_target_std_must_be_finite_positive(target_std: float) -> None:
    config = ExperimentConfig()
    config.model.features.reg.enabled = True
    config.model.features.reg.target_std = target_std
    with pytest.raises(ValueError, match="target_std"):
        config.validate()


def test_structured_override_is_the_feature_selection_interface() -> None:
    config = load_config(
        ROOT / "configs/baseline-p16.yaml",
        [
            "model.features.reg.enabled=true",
            "model.features.reg.head.kind=wide_mlp",
            "model.features.repa.enabled=true",
            "model.features.repa.token_scope=image_and_reg",
        ],
    )
    assert config.model.features.reg.enabled
    assert config.model.features.reg.head.kind == "wide_mlp"
    assert config.model.features.repa.enabled


def test_train_cli_has_no_retired_feature_or_evaluation_flags() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/train.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "baseline-p16.yaml" in result.stdout
    for retired in (
        "--reg ",
        "--reg-widehead",
        "--no-repa",
        "--reference-npz",
        "--eval-batch-size",
        "--eval-solver",
    ):
        assert retired not in result.stdout
