from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from jit_reg.config import ExperimentConfig
from jit_reg.evaluation import (
    evaluate_npz,
    fid_from_statistics,
    validate_reference_statistics,
)
from jit_reg.sample_runtime import (
    SamplingParameters,
    sample_to_archive,
    sampling_output_name,
)
from jit_reg.validation import create_validation_plan
from jit_reg.validation_runtime import validate_ema

ROOT = Path(__file__).resolve().parents[1]
FEATURE_SCHEMA = {
    "schema_version": 2,
    "encoder": {},
    "reg": {"enabled": False},
    "repa": {"enabled": False},
}


def test_sampling_output_name_includes_relative_reg_schedule() -> None:
    name = sampling_output_name(
        10,
        SamplingParameters(
            sampling_steps=3,
            timestep_shift=2,
            reg_timestep_ratio=0.5,
        ),
    )
    assert "-shift2p0-regshift0p5-" in name


def test_sampling_manifest_has_v2_features_and_rejects_old_schema(
    tmp_path: Path,
) -> None:
    plan = create_validation_plan(4, 2, 2, seed=4)

    def generate(
        positions: torch.Tensor,
        labels: torch.Tensor,
        seeds: torch.Tensor,
    ) -> torch.Tensor:
        del labels, seeds
        return torch.zeros(positions.numel(), 3, 8, 8)

    parameters = SamplingParameters(
        batch_size=2,
        sampling_steps=2,
        reg_timestep_ratio=0.5,
        log_interval=1,
    )
    output = tmp_path / "samples"
    archive = sample_to_archive(
        output_dir=output,
        plan=plan,
        sample_count=4,
        num_classes=2,
        image_size=8,
        patch_size=4,
        model_variant="SiT-Tiny/4",
        feature_schema=FEATURE_SCHEMA,
        checkpoint_hash="abc",
        parameters=parameters,
        generate_batch=generate,
    )
    assert archive.is_file()
    manifest = json.loads((output / "samples.json").read_text())
    assert manifest["schema_version"] == 10
    assert manifest["feature_schema"] == FEATURE_SCHEMA
    assert manifest["sampling"]["reg_timestep_ratio"] == 0.5

    old = tmp_path / "old"
    old.mkdir()
    (old / "samples.json").write_text(json.dumps({"schema_version": 9}))
    with pytest.raises(ValueError, match="old manifests"):
        sample_to_archive(
            output_dir=old,
            plan=plan,
            sample_count=4,
            num_classes=2,
            image_size=8,
            patch_size=4,
            model_variant="SiT-Tiny/4",
            feature_schema=FEATURE_SCHEMA,
            checkpoint_hash="abc",
            parameters=parameters,
            generate_batch=generate,
        )


def test_reference_validation_and_known_fid(tmp_path: Path) -> None:
    reference_path = tmp_path / "reference.npz"
    np.savez(reference_path, mu=np.zeros(2), sigma=np.eye(2))
    reference = validate_reference_statistics(reference_path, expected_features=2)
    generated = {"mu": np.array([1.0, 2.0]), "sigma": np.eye(2)}
    assert fid_from_statistics(generated, reference) == pytest.approx(5.0)


class TinyFeatures(nn.Module):
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor]:
        means = images.float().mean(dim=(1, 2, 3))
        return (torch.stack((means, means.square()), dim=1),)


def test_evaluate_npz_streams_features(tmp_path: Path) -> None:
    images = np.stack(
        [
            np.full((256, 256, 3), value, dtype=np.uint8)
            for value in (1, 2, 3, 4)
        ]
    )
    samples = tmp_path / "samples.npz"
    np.savez(samples, images)
    features = np.stack(
        (np.array([1, 2, 3, 4]), np.array([1, 4, 9, 16])), axis=1
    )
    reference = tmp_path / "reference.npz"
    np.savez(
        reference,
        mu=features.mean(0),
        sigma=np.cov(features, rowvar=False),
    )
    metrics = evaluate_npz(
        samples,
        reference,
        batch_size=2,
        device="cpu",
        feature_extractor=TinyFeatures(),
    )
    assert metrics["FID"] == pytest.approx(0.0, abs=1e-8)


def test_validation_fid_label_uses_actual_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import jit_reg.validation_runtime as runtime

    class TinyModel(nn.Module):
        def __init__(self, config: object) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

    monkeypatch.setattr(runtime, "SiT", TinyModel)
    monkeypatch.setattr(runtime, "compile_model_regions", lambda model, config: model)
    monkeypatch.setattr(runtime, "ode_sample", lambda model, noise, labels, **kwargs: noise)
    monkeypatch.setattr(runtime, "evaluate_npz", lambda *args, **kwargs: {"FID": 12.5})
    config = ExperimentConfig()
    config.model.image_size = 8
    config.model.patch_size = 4
    config.model.num_classes = 2
    config.validation.periodic_samples = 2
    config.validation.final_samples = 4
    config.compile.enabled = False
    plan = create_validation_plan(4, 2, 2, seed=5)
    ema = SimpleNamespace(state_dict=lambda: {"weight": torch.zeros(())})
    metrics, output = validate_ema(
        ema=ema,  # type: ignore[arg-type]
        config=config,
        plan=plan,
        completed_epoch=1,
        final=False,
        output_root=tmp_path,
        sampling=SamplingParameters(
            batch_size=1,
            sampling_steps=2,
            solver="heun",
            log_interval=1,
        ),
    )
    assert metrics["validation/FID-2"] == 12.5
    assert (output / "evaluation.json").is_file()


def test_standalone_sampling_inherits_defaults_and_evaluate_writes_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = ROOT / "scripts/sample.py"
    spec = importlib.util.spec_from_file_location("sample_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = ExperimentConfig()
    config.model.image_size = 8
    config.model.patch_size = 4
    config.model.hidden_size = 16
    config.model.depth = 1
    config.model.num_heads = 4
    config.model.num_classes = 2
    config.compile.enabled = False
    config.sampling.num_samples = 4
    config.sampling.batch_size = 3
    config.sampling.sampling_steps = 7
    config.evaluation.reference_npz = str(tmp_path / "reference.npz")
    plan = create_validation_plan(4, 2, 2, seed=3)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "ema.safetensors"
    weights.touch()
    bundle = SimpleNamespace(
        config=config,
        plan=plan,
        directory=checkpoint,
        weights=weights,
    )
    captured: dict[str, object] = {}

    def fake_archive(**kwargs: object) -> Path:
        captured.update(kwargs)
        output = Path(kwargs["output_dir"])  # type: ignore[arg-type]
        output.mkdir(parents=True)
        archive = output / "samples.npz"
        np.savez(archive, np.zeros((4, 8, 8, 3), dtype=np.uint8))
        return archive

    class EmptyModel(nn.Module):
        def __init__(self, config: object) -> None:
            super().__init__()

    monkeypatch.setattr(module, "load_checkpoint_bundle", lambda path: bundle)
    monkeypatch.setattr(module, "checkpoint_sha256", lambda path: "abc")
    monkeypatch.setattr(
        module, "initialize_distributed", lambda: (0, 1, torch.device("cpu"))
    )
    monkeypatch.setattr(module, "broadcast_string", lambda value, rank: value)
    monkeypatch.setattr(module, "load_file", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "SiT", EmptyModel)
    monkeypatch.setattr(module, "compile_model_regions", lambda model, config: model)
    monkeypatch.setattr(module, "sample_to_archive", fake_archive)
    monkeypatch.setattr(module, "evaluate_npz", lambda *args, **kwargs: {"FID": 1.0})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample.py",
            "--checkpoint-dir",
            str(checkpoint),
            "--evaluate",
        ],
    )
    module.main()
    parameters = captured["parameters"]
    assert isinstance(parameters, SamplingParameters)
    assert parameters.batch_size == 3
    assert parameters.sampling_steps == 7
    output = Path(captured["output_dir"])  # type: ignore[arg-type]
    assert (output / "evaluation.json").is_file()


@pytest.mark.parametrize(
    "script",
    ["train.py", "sample.py", "cache_features.py", "eval.py", "compare.py"],
)
def test_standalone_script_help(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
