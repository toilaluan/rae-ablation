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
    latent = load_config(ROOT / "configs/latent-baseline-p16.yaml")
    assert baseline.schema_version == paper.schema_version == latent.schema_version == 8
    assert (
        baseline.train.optimizer.kind
        == paper.train.optimizer.kind
        == "muon"
    )
    assert baseline.model.patch_size == paper.model.patch_size == 16
    assert (
        baseline.model.num_img_t_tokens
        == paper.model.num_img_t_tokens
        == 1
    )
    assert baseline.model.num_reg_t_tokens == 0
    assert paper.model.num_reg_t_tokens == 1
    assert (
        baseline.model.num_c_tokens
        == paper.model.num_c_tokens
        == 1
    )
    assert baseline.model.variant == paper.model.variant == "DDT-B"
    assert baseline.model.hidden_size == paper.model.hidden_size == 768
    assert baseline.model.depth == paper.model.depth == 12
    assert baseline.model.num_heads == paper.model.num_heads == 12
    assert baseline.model.decoder.hidden_size == 2048
    assert baseline.model.decoder.depth == 2
    assert baseline.model.decoder.num_heads == 16
    assert not baseline.model.features.enabled
    assert paper.model.features.reg.num_tokens == 16
    assert paper.model.features.reg.target_std == pytest.approx(0.53)
    assert paper.model.features.reg.internal_guidance_block == 4
    assert not paper.model.fused_mlp
    assert not paper.model.features.reg.independent_attention
    assert not paper.model.features.reg.lora.enabled
    assert paper.model.features.reg.lora.rank == 16
    assert paper.model.features.reg.lora.alpha == pytest.approx(16)
    assert not paper.model.features.reg.lora.mlp
    assert paper.sampling.reg_guidance_scale == pytest.approx(1.78)
    assert paper.model.features.repa.token_scope == "image_and_reg"
    assert latent.model.input_space == "flux2_latents"
    assert latent.model.in_channels == 128
    assert latent.model.input_shape == (128, 16, 16)
    assert latent.model.grid_size == 16
    assert latent.model.patch_size == 1


def test_flux2_model_patch_size_is_configurable_in_latent_units() -> None:
    latent = load_config(
        ROOT / "configs/latent-baseline-p16.yaml",
        ["model.patch_size=2"],
    )
    assert latent.model.input_shape == (128, 16, 16)
    assert latent.model.patch_size == 2
    assert latent.model.grid_size == 8
    assert latent.model.model_variant == "DDT-B/2-flux2_latents"


@pytest.mark.parametrize(
    ("input_space", "in_channels", "message"),
    [
        ("pixels", 128, "must be 3"),
        ("flux2_latents", 3, "must be 128"),
    ],
)
def test_model_channels_must_match_input_space(
    input_space: str,
    in_channels: int,
    message: str,
) -> None:
    config = ExperimentConfig()
    config.model.input_space = input_space
    config.model.in_channels = in_channels
    config.model.patch_size = 1 if input_space == "flux2_latents" else 16
    with pytest.raises(ValueError, match=message):
        config.validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_img_t_tokens", -1),
        ("num_reg_t_tokens", -1),
        ("num_c_tokens", -1),
        ("num_c_tokens", 1.5),
        ("num_img_t_tokens", True),
    ],
)
def test_conditioning_token_counts_must_be_non_negative_integers(
    field: str,
    value: object,
) -> None:
    config = ExperimentConfig()
    setattr(config.model, field, value)
    with pytest.raises(ValueError, match=field):
        config.validate()


def test_reg_timestep_tokens_require_reg() -> None:
    config = ExperimentConfig()
    config.model.num_reg_t_tokens = 1
    with pytest.raises(ValueError, match="requires"):
        config.validate()


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


def test_removed_horizontal_flip_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="horizontal_flip_probability"):
        load_config(
            ROOT / "configs/baseline-p16.yaml",
            ["train.horizontal_flip_probability=0.5"],
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


def test_reg_configuration_exposes_internal_guidance_block() -> None:
    config = ExperimentConfig()
    config.model.features.reg.enabled = True
    config.validate()
    assert not hasattr(config.model.features.reg, "head")
    assert config.model.features.reg.internal_guidance_block == 4


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rank", 0, "rank"),
        ("rank", True, "rank"),
        ("alpha", 0.0, "alpha"),
        ("alpha", float("inf"), "alpha"),
    ],
)
def test_active_reg_lora_validates_rank_and_alpha(
    field: str,
    value: object,
    message: str,
) -> None:
    config = ExperimentConfig()
    config.model.features.reg.enabled = True
    config.model.features.reg.lora.enabled = True
    setattr(config.model.features.reg.lora, field, value)
    with pytest.raises(ValueError, match=message):
        config.validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hidden_size", 0, "decoder dimensions"),
        ("depth", 0, "decoder dimensions"),
        ("num_heads", 0, "decoder dimensions"),
        ("mlp_ratio", 0, "decoder dimensions"),
        ("hidden_size", 30, "divisible"),
    ],
)
def test_decoder_dimensions_validate(
    field: str,
    value: int,
    message: str,
) -> None:
    config = ExperimentConfig()
    setattr(config.model.decoder, field, value)
    with pytest.raises(ValueError, match=message):
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
    config.model.features.reg.target_std = -1
    config.model.features.reg.internal_guidance_block = -1
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

    schema_v7 = tmp_path / "schema-v7.yaml"
    schema_v7.write_text("schema_version: 7\n")
    with pytest.raises(ValueError, match="schema_version: 8"):
        load_config(schema_v7)

    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("schema_version: 8\nretired_experiment: true\n")
    with pytest.raises(ValueError, match="invalid schema-v8"):
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

    config.sampling.reg_denoising_interval = [0.0, 1.0]
    config.sampling.reg_guidance_scale = float("nan")
    with pytest.raises(ValueError, match="reg_guidance_scale"):
        config.validate()


@pytest.mark.parametrize("block", [0, 13, True, 1.5])
def test_internal_guidance_block_must_be_in_active_encoder(
    block: object,
) -> None:
    config = ExperimentConfig()
    config.model.features.reg.enabled = True
    config.model.features.reg.internal_guidance_block = block  # type: ignore[assignment]
    with pytest.raises(ValueError, match="internal_guidance_block"):
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
            "model.features.repa.enabled=true",
            "model.features.repa.token_scope=image_and_reg",
            "model.decoder.depth=3",
        ],
    )
    assert config.model.features.reg.enabled
    assert config.model.features.repa.enabled
    assert config.model.decoder.depth == 3


@pytest.mark.parametrize(
    "override",
    [
        "model.features.reg.head.kind=wide_mlp",
        "model.features.reg.early_prediction.block=4",
    ],
)
def test_removed_reg_head_and_early_overrides_are_rejected(
    override: str,
) -> None:
    with pytest.raises(ValueError):
        load_config(
            ROOT / "configs/reg-repa-p16.yaml",
            [override],
        )


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
