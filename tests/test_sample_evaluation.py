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

from rae_ablation.config import ExperimentConfig, WandbConfig
from rae_ablation.evaluation import (
    NpzImageReader,
    evaluate_npz,
    fid_from_statistics,
    validate_reference_statistics,
)
from rae_ablation.sample_runtime import SamplingParameters, sample_to_archive
from rae_ablation.validation import create_validation_plan
from rae_ablation.validation_runtime import validate_ema
from rae_ablation.wandb_utils import RunLogger


def _decoded(positions: torch.Tensor) -> torch.Tensor:
    values = positions.to(torch.float32).view(-1, 1, 1, 1) / 255
    return values.expand(-1, 3, 256, 256).clone()


def test_sample_archive_is_atomic_balanced_and_resumable(tmp_path: Path) -> None:
    plan = create_validation_plan(4, 2, 2, seed=3, channels=1)
    calls = 0

    def interrupted(
        positions: torch.Tensor, _labels: torch.Tensor, _seeds: torch.Tensor
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return _decoded(positions)

    parameters = SamplingParameters(batch_size=2, sampling_steps=3, log_interval=1)
    with pytest.raises(RuntimeError, match="interrupted"):
        sample_to_archive(
            output_dir=tmp_path,
            plan=plan,
            sample_count=4,
            num_classes=2,
            codec="rae",
            checkpoint_hash="abc",
            parameters=parameters,
            generate_batch=interrupted,
        )
    partial_manifest = json.loads((tmp_path / "samples.json").read_text())
    assert not partial_manifest["complete"]
    assert (tmp_path / ".samples-completion.npy").exists()
    resumed_positions: list[int] = []

    def resumed(
        positions: torch.Tensor, _labels: torch.Tensor, _seeds: torch.Tensor
    ) -> torch.Tensor:
        resumed_positions.extend(positions.tolist())
        return _decoded(positions)

    archive = sample_to_archive(
        output_dir=tmp_path,
        plan=plan,
        sample_count=4,
        num_classes=2,
        codec="rae",
        checkpoint_hash="abc",
        parameters=parameters,
        generate_batch=resumed,
    )
    assert resumed_positions == [2, 3]
    with np.load(archive, allow_pickle=False) as values:
        assert values["arr_0"].shape == (4, 256, 256, 3)
        assert values["arr_0"].dtype == np.uint8
        assert values["arr_0"][:, 0, 0, 0].tolist() == [0, 1, 2, 3]
        assert values["arr_1"].tolist() == [0, 1, 0, 1]
    with zipfile.ZipFile(archive) as values:
        info = values.getinfo("arr_0.npy")
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.extract_version >= 45
    manifest = json.loads((tmp_path / "samples.json").read_text())
    assert manifest["complete"] and manifest["class_counts"] == [2, 2]
    assert (tmp_path / "preview.png").exists()
    preview = np.asarray(Image.open(tmp_path / "preview.png"))
    assert preview[0, [0, 256, 512, 768], 0].tolist() == [0, 1, 2, 3]
    assert not list(tmp_path.glob(".samples-*.npy"))
    assert not list(tmp_path.glob("*.partial-*"))
    with pytest.raises(ValueError, match="does not match"):
        sample_to_archive(
            output_dir=tmp_path,
            plan=plan,
            sample_count=4,
            num_classes=2,
            codec="rae",
            checkpoint_hash="different",
            parameters=parameters,
            generate_batch=resumed,
        )


@pytest.mark.parametrize("compressed", [False, True])
def test_npz_reader_streams_stored_and_compressed(tmp_path: Path, compressed: bool) -> None:
    images = np.arange(3 * 256 * 256 * 3, dtype=np.uint8).reshape(3, 256, 256, 3)
    path = tmp_path / "images.npz"
    save = np.savez_compressed if compressed else np.savez
    save(path, images)
    reader = NpzImageReader(path)
    actual = np.concatenate(list(reader.batches(2)))
    assert np.array_equal(actual, images)
    assert reader.compressed is compressed


def test_reference_validation_and_known_fid(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.npz"
    np.savez(reference_path, mu=np.zeros(2), sigma=np.eye(2))
    reference = validate_reference_statistics(reference_path, expected_features=2)
    generated = {"mu": np.array([1.0, 2.0]), "sigma": np.eye(2)}
    assert fid_from_statistics(generated, reference) == pytest.approx(5.0)
    np.savez(reference_path, mu=np.zeros(2), sigma=np.ones((3, 3)))
    with pytest.raises(ValueError, match="incompatible"):
        validate_reference_statistics(reference_path, expected_features=2)


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
    features = np.stack((np.array([1, 2, 3, 4]), np.array([1, 4, 9, 16])), axis=1)
    reference = tmp_path / "reference.npz"
    np.savez(reference, mu=features.mean(0), sigma=np.cov(features, rowvar=False))
    messages: list[str] = []
    metrics = evaluate_npz(
        samples,
        reference,
        batch_size=2,
        device="cpu",
        log_interval=1,
        log=messages.append,
        feature_extractor=_TinyFeatures(),
    )
    assert metrics["FID"] == pytest.approx(0.0, abs=1e-8)
    assert any(message.startswith("feature extraction:") for message in messages)


@pytest.mark.parametrize("script", ["train.py", "sample.py", "eval.py"])
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


def test_combined_entry_points_are_removed() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert project["project"]["scripts"].keys() == {"rae-cache", "rae-compare"}


def test_console_training_throttle_and_non_main_silence(capsys: pytest.CaptureFixture[str]) -> None:
    metrics = {
        "train/epoch": 1.5,
        "train/loss_velocity_combined": 3.0,
        "train/loss_velocity_full": 1.0,
        "train/loss_velocity_early": 2.0,
        "train/lr": 1e-4,
        "train/gradient_norm": 0.5,
        "throughput/images_per_second": 10.0,
        "throughput/step_seconds": 0.2,
        "system/peak_memory_gib": 1.0,
    }
    logger = RunLogger(WandbConfig(), {}, True, log_interval=2)
    logger.log_training(metrics, step=1)
    assert capsys.readouterr().out == ""
    logger.log_training(metrics, step=2)
    assert "step=2" in capsys.readouterr().out
    non_main = RunLogger(WandbConfig(), {}, False, log_interval=1)
    non_main.log_training(metrics, step=1)
    non_main.event("shutdown")
    assert capsys.readouterr().out == ""


def test_online_validation_writes_shared_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import rae_ablation.validation_runtime as runtime

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

    class TinyCodec:
        def decode(self, latents: torch.Tensor) -> torch.Tensor:
            return torch.zeros((latents.shape[0], 3, 256, 256), device=latents.device)

    monkeypatch.setattr(runtime, "build_diffusion_model", lambda _config: TinyModel())
    monkeypatch.setattr(runtime, "compile_model_regions", lambda model, _config: model)
    monkeypatch.setattr(runtime, "load_codecs", lambda _codec, _device: TinyCodec())
    monkeypatch.setattr(runtime, "euler_sample", lambda _model, noise, *_args: noise)
    monkeypatch.setattr(runtime, "evaluate_npz", lambda *_args, **_kwargs: {"FID": 12.5})
    config = ExperimentConfig()
    config.model.num_classes = 2
    config.validation.periodic_samples = 2
    config.validation.final_samples = 4
    config.compile.enabled = False
    plan = create_validation_plan(4, 2, 2, seed=5, channels=1)
    ema = SimpleNamespace(state_dict=lambda: {"weight": torch.zeros(())})
    statistics = SimpleNamespace(denormalize=lambda tensor: tensor)
    metrics, output = validate_ema(
        ema=ema,
        config=config,
        statistics=statistics,
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
    assert (output / "samples.json").exists()
    assert (output / "preview.png").exists()
    evaluation = json.loads((output / "evaluation.json").read_text())
    assert evaluation["validation_plan_hash"] == plan.plan_hash
