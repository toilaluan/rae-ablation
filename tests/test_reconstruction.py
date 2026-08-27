from __future__ import annotations

import json

import pytest
import torch
from PIL import Image
from torch import nn

import cascade_dit.preparation as preparation_module
from cascade_dit.config import ExperimentConfig
from cascade_dit.model import DDTOutput
from cascade_dit.preparation import prepare_reconstruction_representations
from cascade_dit.reconstruction import (
    ReconstructionBatch,
    ReconstructionBatchPreprocessor,
    load_reconstruction_manifest,
    select_class_balanced_indices,
)
from cascade_dit.representations import ChannelStatistics
from cascade_dit.validation import (
    save_reconstruction_evaluation,
    save_reconstruction_preview,
    validate_reconstruction,
)


class FakeImageNet:
    def __init__(self, labels: list[int]) -> None:
        self.labels = [{"label": label} for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def get_item(self, index: int) -> tuple[torch.Tensor, int, int]:
        label = int(self.labels[index]["label"])
        pixels = torch.full((3, 128, 128), index, dtype=torch.uint8)
        return pixels, label, index


class EncoderStub(nn.Module):
    def forward(self, pixels: torch.Tensor) -> dict[int, torch.Tensor]:
        values = pixels[:, :1, :8, :8].flatten(2).transpose(1, 2).float()
        return {12: values.expand(-1, -1, 768)}


class LPIPSStub(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).abs().mean((1, 2, 3), keepdim=True)


class ReconstructionOracle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        semantic: torch.Tensor,
        timestep: torch.Tensor,
        labels: torch.Tensor,
        *,
        decode_pixels: bool,
        force_class_drop: torch.Tensor,
    ) -> DDTOutput:
        assert decode_pixels
        self.calls.append(
            (
                semantic.detach().clone(),
                timestep.detach().clone(),
                force_class_drop.detach().clone(),
            )
        )
        pixels = torch.full(
            (semantic.shape[0], 3, 128, 128),
            0.5,
            device=semantic.device,
        )
        return DDTOutput(semantic, semantic, pixels)


def _config(tmp_path) -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.conditioning.num_classes = 2
    config.data.workers = 0
    config.validation.reconstruction.cache_directory = str(tmp_path)
    config.validation.reconstruction.preview_samples = 2
    config.compile.enabled = False
    return config


def test_class_balanced_selection_is_class_major_and_uses_first_examples() -> None:
    indices, labels = select_class_balanced_indices(
        FakeImageNet([1, 0, 1, 0, 1, 0]),  # type: ignore[arg-type]
        num_classes=2,
        samples_per_class=2,
    )
    assert indices == [1, 3, 0, 2]
    assert labels == [0, 0, 1, 1]


def test_reconstruction_cache_preparation_is_resumable_and_provenanced(
    tmp_path, monkeypatch
) -> None:
    config = _config(tmp_path)
    dataset = FakeImageNet([1, 0, 1, 0])
    loads = 0

    def load_encoder(*_args, **_kwargs):
        nonlocal loads
        loads += 1
        return EncoderStub()

    monkeypatch.setattr(preparation_module, "load_imagenet", lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr(preparation_module, "load_dino_encoder", load_encoder)

    first = prepare_reconstruction_representations(
        config, rank=0, world_size=1, device=torch.device("cpu")
    )
    second = prepare_reconstruction_representations(
        config, rank=0, world_size=1, device=torch.device("cpu")
    )

    assert first == second == load_reconstruction_manifest(config)
    assert first.complete
    assert first.source_indices == [1, 0]
    assert first.labels == [0, 1]
    assert loads == 1


def test_reconstruction_metrics_use_clean_tokens_fixed_time_and_paired_preview(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    config.sampling.final_decode_timestep = 0.5
    batch = ReconstructionBatch(
        semantic=torch.stack((torch.zeros(64, 768), torch.ones(64, 768))),
        pixels=torch.stack(
            (
                torch.full((3, 128, 128), 64, dtype=torch.uint8),
                torch.full((3, 128, 128), 191, dtype=torch.uint8),
            )
        ),
        labels=torch.tensor([0, 1]),
        positions=torch.tensor([0, 1]),
    )
    statistics = ChannelStatistics(torch.zeros(768), torch.ones(768), 128)
    model = ReconstructionOracle()
    result = validate_reconstruction(
        model,
        [batch],
        ReconstructionBatchPreprocessor(statistics, "cpu"),
        config,
        LPIPSStub(),
        device=torch.device("cpu"),
    )

    assert len(model.calls) == 1
    semantic, timestep, force_drop = model.calls[0]
    assert torch.equal(semantic, batch.semantic)
    assert torch.equal(timestep, torch.full((2,), 0.5))
    assert not force_drop.any()
    expected_l1 = (abs(0.5 - 64 / 255) + abs(0.5 - 191 / 255)) / 2
    assert result.metrics["l1"] == pytest.approx(expected_l1)
    assert result.metrics["lpips"] == pytest.approx(expected_l1 * 2)
    assert result.metrics["psnr"] == pytest.approx(
        -10 * torch.log10(torch.tensor(expected_l1**2)).item(), rel=1e-4
    )
    assert result.sample_count == 2

    preview = tmp_path / "preview.png"
    save_reconstruction_preview(result, preview)
    assert Image.open(preview).size == (256, 312)
    metrics = tmp_path / "reconstruction.json"
    save_reconstruction_evaluation(
        metrics,
        result,
        timestep=0.5,
        cache_fingerprint="cache",
        ema_sha256="ema",
    )
    payload = json.loads(metrics.read_text())
    assert payload["final_decode_timestep"] == 0.5
    assert payload["reconstruction_cache_fingerprint"] == "cache"
