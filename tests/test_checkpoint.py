from __future__ import annotations

import json

import torch
from torch import nn

from cascade_dit.checkpoint import EMA, CheckpointManager
from cascade_dit.config import ExperimentConfig, ModelConfig
from cascade_dit.model import build_model
from cascade_dit.rng import RNGStreams
from cascade_dit.validation import create_validation_plan


def test_ema_foreach_updates_floating_state_and_copies_integer_buffers() -> None:
    model = nn.BatchNorm1d(3)
    ema = EMA(model, decay=0.75)
    original = {key: value.clone() for key, value in ema.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(4)
        model.bias.add_(4)
        model.running_mean.add_(4)
        model.running_var.add_(4)
        model.num_batches_tracked.add_(1)
    ema.update(model)
    current = model.state_dict()
    for key, value in ema.state_dict().items():
        if value.is_floating_point():
            assert torch.allclose(value, original[key].lerp(current[key], 0.25))
        else:
            assert torch.equal(value, current[key])


def test_checkpoint_restores_architecture_provenance_and_every_rng_stream(tmp_path) -> None:
    config = ExperimentConfig(
        model=ModelConfig(
            architecture="latent",
            hidden_size=(32, 48),
            depth=(1, 1),
            num_heads=(4, 3),
            base_model_depth=1,
            num_classes=2,
        )
    )
    model = build_model(config.model)
    ema = EMA(model)
    streams = RNGStreams.create(7)
    plan = create_validation_plan(4, 2, 2, 11)
    plan_path = tmp_path / "plan.pt"
    plan.save(plan_path)
    representations = {
        "mage_vae": {
            "repository": "mage-flow-community/Mage-Flow",
            "revision": "revision",
            "weight_sha256": "weights",
            "posterior": "mean",
            "image_latent_shape": [128, 16, 16],
            "encode_preprocessing": "[-1,1]",
        }
    }
    caches = {"image_latents": "online", "semantic_tokens": "not-used"}
    manager = CheckpointManager(tmp_path / "checkpoints")
    checkpoint = manager.save(
        completed_epoch=1,
        global_step=3,
        model=model,
        ema=ema,
        optimizer_state={"state": "optimizer"},
        rng_streams=streams.state_dict(),
        architecture="latent",
        data_fingerprint="data",
        validation_plan_hash=plan.plan_hash,
        validation_plan_path=plan_path,
        last_completed_validation_epoch=None,
        wandb_run_id=None,
        resolved_config=config.as_dict(),
        representation_provenance=representations,
        cache_provenance=caches,
    )
    saved_config = json.loads((checkpoint / "config.json").read_text())
    saved_config["model"].pop("image_target")
    train_values = saved_config["train"]
    train_values["timestep_loc"] = train_values.pop("semantic_timestep_loc")
    train_values.pop("image_timestep_loc")
    train_values["timestep_scale"] = train_values.pop("semantic_timestep_scale")
    train_values.pop("image_timestep_scale")
    train_values["timestep_shift"] = train_values.pop("semantic_timestep_shift")
    train_values.pop("image_timestep_shift")
    sampling_values = saved_config["sampling"]
    sampling_values["timestep_shift"] = sampling_values.pop(
        "semantic_timestep_shift"
    )
    sampling_values.pop("image_timestep_shift")
    (checkpoint / "config.json").write_text(json.dumps(saved_config))
    restored = manager.load(
        checkpoint,
        model,
        ema,
        expected_architecture="latent",
        expected_representation_provenance=representations,
        expected_cache_provenance=caches,
    )
    assert restored.architecture == "latent"
    assert set(restored.rng_streams) == set(streams.state_dict())


def test_pixel_target_checkpoint_roundtrip_uses_pixel_provenance(tmp_path) -> None:
    config = ExperimentConfig(
        model=ModelConfig(
            architecture="latent",
            image_target="pixels",
            hidden_size=(32, 48),
            depth=(1, 1),
            num_heads=(4, 3),
            base_model_depth=1,
            num_classes=2,
        )
    )
    model = build_model(config.model)
    ema = EMA(model)
    streams = RNGStreams.create(9)
    plan = create_validation_plan(4, 2, 2, 13)
    plan_path = tmp_path / "pixel-plan.pt"
    plan.save(plan_path)
    representations = {
        "pixels": {
            "kind": "pixels",
            "shape": [3, 256, 256],
            "dtype": "float32",
            "range": [-1.0, 1.0],
            "data_fingerprint": "data",
            "preprocessing": "uint8-to-[-1,1]",
        }
    }
    caches = {"image_latents": "not-used", "semantic_tokens": "not-used"}
    manager = CheckpointManager(tmp_path / "pixel-checkpoints")
    checkpoint = manager.save(
        completed_epoch=1,
        global_step=2,
        model=model,
        ema=ema,
        optimizer_state={},
        rng_streams=streams.state_dict(),
        architecture="latent",
        data_fingerprint="data",
        validation_plan_hash=plan.plan_hash,
        validation_plan_path=plan_path,
        last_completed_validation_epoch=None,
        wandb_run_id=None,
        resolved_config=config.as_dict(),
        representation_provenance=representations,
        cache_provenance=caches,
    )
    restored = manager.load(
        checkpoint,
        model,
        ema,
        expected_architecture="latent",
        expected_representation_provenance=representations,
        expected_cache_provenance=caches,
    )
    assert restored.representation_provenance == representations
