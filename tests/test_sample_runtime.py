from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from cascade_dit.sample_runtime import (
    SamplingParameters,
    prune_sample_archive,
    sample_to_archive,
)

LABELS = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int64)
SEEDS = torch.arange(1, 7, dtype=torch.int64)


def _images(_labels: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    values = seeds.float().div(10).reshape(-1, 1, 1, 1)
    return values.expand(-1, 3, 2, 2).contiguous()


def _sample(output: Path, generate=_images) -> Path:
    return sample_to_archive(
        output_directory=output,
        num_samples=6,
        num_classes=2,
        image_size=2,
        checkpoint_hash="checkpoint",
        parameters=SamplingParameters(steps=1, batch_size=2),
        generate_batch=generate,
        labels=LABELS,
        seeds=SEEDS,
        plan_hash="plan",
    )


def _distributed_worker(rank: int, world_size: int, initialization: str, output: str) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{initialization}",
        rank=rank,
        world_size=world_size,
    )
    try:
        _sample(Path(output))
    finally:
        torch.distributed.destroy_process_group()


def test_sampling_resumes_and_prunes_only_after_evaluation(tmp_path) -> None:
    calls = 0

    def interrupted(labels: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return _images(labels, seeds)

    output = tmp_path / "resumable"
    with pytest.raises(RuntimeError, match="interrupted"):
        _sample(output, interrupted)
    completion = np.load(output / ".completion.npy")
    assert completion.tolist() == [1, 1, 0, 0, 0, 0]

    archive = _sample(output)
    with np.load(archive) as values:
        assert values["arr_1"].tolist() == LABELS.tolist()
        assert values["arr_0"][:, 0, 0, 0].tolist() == [26, 51, 76, 102, 128, 153]
    with pytest.raises(ValueError, match="after evaluation"):
        prune_sample_archive(output)
    (output / "evaluation.json").write_text("{}")
    prune_sample_archive(output)
    assert not archive.exists()
    assert (output / "preview.png").is_file()
    manifest = json.loads((output / "samples.json").read_text())
    assert manifest["complete"] is True
    assert manifest["archive_retained"] is False


def test_two_process_sampling_matches_single_process(tmp_path) -> None:
    distributed_output = tmp_path / "distributed"
    initialization = tmp_path / "gloo-init"
    torch.multiprocessing.spawn(
        _distributed_worker,
        args=(2, str(initialization), str(distributed_output)),
        nprocs=2,
        join=True,
    )
    single_archive = _sample(tmp_path / "single")
    with np.load(distributed_output / "samples.npz") as distributed_values, np.load(
        single_archive
    ) as single_values:
        assert np.array_equal(distributed_values["arr_0"], single_values["arr_0"])
        assert np.array_equal(distributed_values["arr_1"], single_values["arr_1"])
    manifest = json.loads((distributed_output / "samples.json").read_text())
    assert manifest["completed_samples"] == 6
    assert manifest["sampling_plan_hash"] == "plan"
