from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from rae_ablation.reconstruction import (
    FeatureMoments,
    ReconstructionMetricAccumulator,
    ReconstructionParameters,
    build_reconstruction_report,
    rank_indices,
    save_reconstruction_report,
)


@pytest.mark.parametrize("sample_count,world_size", [(10, 3), (2, 4), (17, 8)])
def test_rank_indices_cover_samples_once(sample_count: int, world_size: int) -> None:
    partitions = [list(rank_indices(sample_count, rank, world_size)) for rank in range(world_size)]
    combined = [index for partition in partitions for index in partition]
    assert sorted(combined) == list(range(sample_count))
    assert len(combined) == len(set(combined))


def test_feature_moments_match_direct_statistics() -> None:
    features = torch.tensor([[1.0, 2.0], [3.0, 7.0], [8.0, 4.0], [5.0, 9.0]])
    moments = FeatureMoments(2, torch.device("cpu"))
    moments.update(features[:1])
    moments.update(features[1:])
    statistics = moments.statistics()
    assert statistics["mu"] == pytest.approx(features.numpy().mean(axis=0))
    assert statistics["sigma"] == pytest.approx(np.cov(features.numpy(), rowvar=False))


class _PixelLPIPS(nn.Module):
    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return (reconstructed - original).abs().mean(dim=(1, 2, 3))

    def reset(self) -> None:
        pass


class _MeanFeatures(nn.Module):
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor]:
        return (images.float().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1),)


def test_metric_accumulator_uses_per_image_means_and_clipped_uint8_rfid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rae_ablation.reconstruction as reconstruction

    def mean_distance(generated: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> float:
        return float(np.square(generated["mu"] - reference["mu"]).sum())

    monkeypatch.setattr(reconstruction, "fid_from_statistics", mean_distance)
    original = torch.stack(
        (
            torch.full((3, 256, 256), 0.2),
            torch.full((3, 256, 256), 0.4),
        )
    )
    accumulator = ReconstructionMetricAccumulator(
        torch.device("cpu"),
        lpips_metric=_PixelLPIPS(),
        feature_extractor=_MeanFeatures(),
        feature_dimensions=1,
    )
    accumulator.update(
        original,
        {
            "rae": original + 0.1,
            "rae_compact": original + 0.2,
        },
    )
    metrics = accumulator.compute()
    assert metrics["rae"]["LPIPS"] == pytest.approx(0.1)
    assert metrics["rae_compact"]["LPIPS"] == pytest.approx(0.2)
    assert metrics["rae"]["PSNR"] == pytest.approx(20.0, abs=1e-4)
    assert metrics["rae_compact"]["PSNR"] == pytest.approx(13.9794, abs=1e-4)
    assert metrics["rae"]["SSIM"] > metrics["rae_compact"]["SSIM"]
    assert metrics["rae"]["rFID"] < metrics["rae_compact"]["rFID"]


def test_reconstruction_report_contains_provenance_deltas_and_is_atomic(tmp_path: Path) -> None:
    metrics = {
        "rae": {"LPIPS": 0.1, "PSNR": 25.0, "SSIM": 0.9, "rFID": 0.5},
        "rae_compact": {"LPIPS": 0.2, "PSNR": 20.0, "SSIM": 0.8, "rFID": 1.5},
    }
    parameters = ReconstructionParameters(batch_size=4, workers=0, max_samples=8)
    report = build_reconstruction_report(
        metrics,
        sample_count=8,
        parameters=parameters,
        world_size=2,
        device=torch.device("cpu"),
    )
    assert report["dataset"]["split"] == "validation"
    assert report["codecs"]["rae"]["latent_shape"] == [1024, 16, 16]
    assert report["codecs"]["rae_compact"]["base_revision"]
    assert report["deltas"]["rae_compact_minus_rae"] == pytest.approx(
        {"LPIPS": 0.1, "PSNR": -5.0, "SSIM": -0.1, "rFID": 1.0}
    )
    output = tmp_path / "nested" / "metrics.json"
    save_reconstruction_report(output, report)
    assert json.loads(output.read_text()) == report
    assert not list(output.parent.glob("*.partial-*"))


@pytest.mark.parametrize(
    "parameters,error",
    [
        (ReconstructionParameters(batch_size=0), "batch size"),
        (ReconstructionParameters(workers=-1), "workers"),
        (ReconstructionParameters(max_samples=1), "at least two"),
        (ReconstructionParameters(log_interval=0), "log interval"),
    ],
)
def test_reconstruction_parameters_reject_invalid_values(
    parameters: ReconstructionParameters, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        parameters.validate()
