from __future__ import annotations

import json
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from jit_reg.config import ExperimentConfig, WandbConfig
from jit_reg.evaluation import (
    NpzImageReader,
    evaluate_npz,
    fid_from_statistics,
    validate_reference_statistics,
)
from jit_reg.sample_runtime import SamplingParameters, sample_to_archive
from jit_reg.validation import create_validation_plan
from jit_reg.validation_runtime import validate_ema
from jit_reg.wandb_utils import RunLogger


def _generated(positions: torch.Tensor) -> torch.Tensor:
    values = positions.float().view(-1, 1, 1, 1) / 255
    return values.expand(-1, 3, 256, 256).clone()


def _sample(
    root: Path,
    generate: object,
    *,
    checkpoint_hash: str = "abc",
    parameters: SamplingParameters | None = None,
) -> Path:
    return sample_to_archive(
        output_dir=root,
        plan=create_validation_plan(4, 2, 2, seed=3),
        sample_count=4,
        num_classes=2,
        image_size=256,
        patch_size=32,
        model_variant="SiT-B/32",
        reg_enabled=False,
        checkpoint_hash=checkpoint_hash,
        parameters=parameters or SamplingParameters(batch_size=2, sampling_steps=3),
        generate_batch=generate,  # type: ignore[arg-type]
    )


def test_sample_archive_is_atomic_balanced_resumable_and_records_jit_metadata(
    tmp_path: Path,
) -> None:
    calls = 0

    def interrupted(
        positions: torch.Tensor, _labels: torch.Tensor, _seeds: torch.Tensor
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return _generated(positions)

    parameters = SamplingParameters(batch_size=2, sampling_steps=3, log_interval=1)
    with pytest.raises(RuntimeError, match="interrupted"):
        _sample(tmp_path, interrupted, parameters=parameters)
    resumed_positions: list[int] = []

    def resumed(
        positions: torch.Tensor, _labels: torch.Tensor, _seeds: torch.Tensor
    ) -> torch.Tensor:
        resumed_positions.extend(positions.tolist())
        return _generated(positions)

    archive = _sample(tmp_path, resumed, parameters=parameters)
    assert resumed_positions == [2, 3]
    with np.load(archive, allow_pickle=False) as values:
        assert values["arr_0"].shape == (4, 256, 256, 3)
        assert values["arr_1"].tolist() == [0, 1, 0, 1]
    with zipfile.ZipFile(archive) as values:
        assert values.getinfo("arr_0.npy").compress_type == zipfile.ZIP_STORED
    manifest = json.loads((tmp_path / "samples.json").read_text())
    assert manifest["complete"] and manifest["schema_version"] == 4
    assert manifest["patch_size"] == 32 and manifest["jit_grid_size"] == 8
    assert manifest["model_variant"] == "SiT-B/32" and not manifest["reg_enabled"]
    assert manifest["sampling"]["cfg_interval"] == [0.0, 1.0]
    preview = np.asarray(Image.open(tmp_path / "preview.png"))
    assert preview[0, [0, 256, 512, 768], 0].tolist() == [0, 1, 2, 3]
    assert not list(tmp_path.glob(".samples-*.npy"))
    with pytest.raises(ValueError, match="does not match"):
        _sample(tmp_path, resumed, checkpoint_hash="different", parameters=parameters)


@pytest.mark.parametrize("compressed", [False, True])
def test_npz_reader_streams_stored_and_compressed(tmp_path: Path, compressed: bool) -> None:
    images = np.arange(3 * 256 * 256 * 3, dtype=np.uint8).reshape(3, 256, 256, 3)
    path = tmp_path / "images.npz"
    (np.savez_compressed if compressed else np.savez)(path, images)
    reader = NpzImageReader(path)
    assert np.array_equal(np.concatenate(list(reader.batches(2))), images)
    assert reader.compressed is compressed


def test_reference_validation_and_known_fid(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.npz"
    np.savez(reference_path, mu=np.zeros(2), sigma=np.eye(2))
    reference = validate_reference_statistics(reference_path, expected_features=2)
    generated = {"mu": np.array([1.0, 2.0]), "sigma": np.eye(2)}
    assert fid_from_statistics(generated, reference) == pytest.approx(5.0)


class _TinyFeatures(nn.Module):
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor]:
        means = images.float().mean(dim=(1, 2, 3))
        return (torch.stack((means, means.square()), dim=1),)


def test_evaluate_npz_with_streamed_features(tmp_path: Path) -> None:
    images = np.stack(
        [np.full((256, 256, 3), value, dtype=np.uint8) for value in (1, 2, 3, 4)]
    )
    samples = tmp_path / "samples.npz"
    np.savez_compressed(samples, images)
    features = np.stack(
        (np.array([1, 2, 3, 4]), np.array([1, 4, 9, 16])), axis=1
    )
    reference = tmp_path / "reference.npz"
    np.savez(reference, mu=features.mean(0), sigma=np.cov(features, rowvar=False))
    metrics = evaluate_npz(
        samples,
        reference,
        batch_size=2,
        device="cpu",
        feature_extractor=_TinyFeatures(),
    )
    assert metrics["FID"] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("script", ["train.py", "sample.py", "eval.py", "compare.py"])
def test_standalone_script_help(script: str) -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_distribution_and_console_logger(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert project["project"]["name"] == "jit-reg"
    logger = RunLogger(WandbConfig(), {}, True, log_interval=1)
    logger.log_training(
        {
            "train/epoch": 1.5,
            "train/loss": 3.0,
            "train/image_velocity_loss": 2.0,
            "train/cls_velocity_loss": 0.0,
            "train/repa_loss": 0.0,
            "train/lr": 1e-4,
            "train/gradient_norm": 0.5,
            "throughput/images_per_second": 10.0,
            "throughput/step_seconds": 0.2,
            "system/peak_memory_gib": 1.0,
        },
        step=1,
    )
    assert "image=" in capsys.readouterr().out


def test_online_validation_writes_shared_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import jit_reg.validation_runtime as runtime

    class TinyModel(nn.Module):
        def __init__(self, config: object) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(runtime, "SiT", TinyModel)
    monkeypatch.setattr(runtime, "compile_model_regions", lambda model, config: model)
    monkeypatch.setattr(runtime, "euler_sample", lambda model, noise, labels, **kwargs: noise)
    monkeypatch.setattr(runtime, "evaluate_npz", lambda *args, **kwargs: {"FID": 12.5})
    config = ExperimentConfig()
    config.model.num_classes = 2
    config.validation.periodic_samples = 2
    config.validation.final_samples = 4
    config.compile.enabled = False
    plan = create_validation_plan(4, 2, 2, seed=5)
    ema = SimpleNamespace(state_dict=lambda: {"weight": torch.zeros(())})
    metrics, output = validate_ema(
        ema=ema,
        config=config,
        plan=plan,
        completed_epoch=1,
        final=False,
        output_root=tmp_path,
        reference_npz=tmp_path / "reference.npz",
        sampling=SamplingParameters(batch_size=1, sampling_steps=1, log_interval=1),
        eval_batch_size=1,
        eval_log_interval=1,
    )
    assert metrics["validation/FID-5K"] == 12.5
    assert (output / "samples.npz").exists()
    evaluation = json.loads((output / "evaluation.json").read_text())
    assert evaluation["validation_plan_hash"] == plan.plan_hash
