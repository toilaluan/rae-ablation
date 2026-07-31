from __future__ import annotations

from pathlib import Path

import pytest

from self_flow.config import ExperimentConfig, ObjectiveConfig, SIGRegConfig, load_config


@pytest.mark.parametrize(
    "name,kind,clean_forward",
    [
        ("vanilla.yaml", "vanilla", "truncated"),
        ("dual-timestep.yaml", "dual_timestep", "truncated"),
        ("self-flow-ema.yaml", "ema", "truncated"),
        ("sigreg-truncated.yaml", "sigreg", "truncated"),
        ("sigreg-full.yaml", "sigreg", "full"),
    ],
)
def test_canonical_configs(name: str, kind: str, clean_forward: str) -> None:
    config = load_config(Path("configs") / name)
    assert config.schema_version == 11
    assert config.objective.kind == kind
    assert config.objective.sigreg.clean_forward == clean_forward
    assert config.model.variant == "SiT-B/2"
    assert config.model.input_shape == (128, 16, 16)
    assert config.model.grid_size == 8


def test_alignment_layers_are_conditional_and_validated() -> None:
    config = ExperimentConfig(objective=ObjectiveConfig(kind="ema", align_layers=[4, 8]))
    config.validate()
    config.objective.align_layers = [8, 4]
    with pytest.raises(ValueError, match="sorted and unique"):
        config.validate()
    config.objective = ObjectiveConfig(kind="vanilla", align_layers=[4])
    with pytest.raises(ValueError, match="only valid"):
        config.validate()
    config.objective = ObjectiveConfig(
        kind="ema",
        align_layers=[4, 8],
        sigreg=SIGRegConfig(clean_forward="full"),
    )
    with pytest.raises(ValueError, match="only valid for kind=sigreg"):
        config.validate()


def test_old_schema_and_fused_muon_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "old.yaml"
    path.write_text("schema_version: 10\n")
    with pytest.raises(ValueError, match="schema_version: 11"):
        load_config(path)
    config = ExperimentConfig()
    config.train.optimizer.kind = "fused_muon"
    with pytest.raises(ValueError, match="adamw or muon"):
        config.validate()
