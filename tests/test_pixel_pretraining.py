from __future__ import annotations

import pytest
import torch
from PIL import Image
from torch import nn

import cascade_dit.pixel_training as pixel_training_module
import cascade_dit.runner as runner_module
from cascade_dit.checkpoint import EMA
from cascade_dit.config import (
    DDTAttentionConfig,
    DDTPathConfig,
    ExperimentConfig,
    ModelConfig,
    PixelDecoderConfig,
    SemanticDecoderConfig,
)
from cascade_dit.data import UnlabeledImageDataset
from cascade_dit.model import SemanticPixelDecoder, build_model
from cascade_dit.pixel_artifact import (
    PIXEL_DECODER_TRAINING_SCHEMA_VERSION,
    PixelDecoderCheckpointManager,
    PixelDecoderResumeState,
    load_pixel_decoder_artifact,
)
from cascade_dit.pixel_data import (
    PixelBatchPreprocessor,
    PixelPretrainingBatch,
    PixelSemanticStatistics,
)
from cascade_dit.pixel_training import PixelPretrainingRunner
from cascade_dit.representations import ChannelStatistics
from cascade_dit.runner import ExperimentRunner


class FastDetailer(nn.Module):
    def forward(self, coarse: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return coarse + 0.01 * context[:, :3, None, None]


class OnlineEncoder(nn.Module):
    def forward(self, pixels: torch.Tensor) -> dict[int, torch.Tensor]:
        batch = pixels.shape[0]
        return {12: torch.full((batch, 64, 768), 3.0, device=pixels.device)}


class LPIPSStub(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).abs().mean((1, 2, 3), keepdim=True)


def _config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.training.epochs = 2
    config.training.decoder.frozen_epochs = 1
    config.pixel_pretraining.epochs = 2
    config.pixel_pretraining.optimizer.scheduler.warmup_epochs = 0
    config.model = ModelConfig(
        base_prediction_depth=1,
        encoder=DDTPathConfig(16, 1, 1),
        semantic_decoder=SemanticDecoderConfig(32, 1, 2),
        pixel_decoder=PixelDecoderConfig(32, 1, 2, [1], [1]),
        attention=DDTAttentionConfig(rope=False),
    )
    return config


def _statistics() -> ChannelStatistics:
    return ChannelStatistics(torch.zeros(768), torch.ones(768), 128)


def test_manual_pixel_decoder_checkpoint_roundtrip(tmp_path) -> None:
    config = _config()
    decoder = SemanticPixelDecoder(config.model)
    ema = EMA(decoder)
    state = PixelDecoderResumeState(
        schema_version=PIXEL_DECODER_TRAINING_SCHEMA_VERSION,
        completed_epoch=1,
        global_step=7,
        training_complete=False,
        optimizer=None,
        scheduler=None,
        rng_state={"cpu": torch.get_rng_state(), "cuda": None},
        wall_clock_seconds=2.0,
        data_fingerprint="data",
        encoder_fingerprint="encoder",
    )
    saved = PixelDecoderCheckpointManager(tmp_path).save(
        model=decoder,
        ema=ema,
        state=state,
        config=config,
        statistics=_statistics(),
    )
    artifact = load_pixel_decoder_artifact(
        saved,
        expected_config=config.model,
        weights_kind="ema",
    )
    assert artifact.directory == saved
    assert artifact.weights_kind == "ema"
    assert len(artifact.weights_sha256) == 64
    assert artifact.statistics.count == 128
    assert all(
        torch.equal(artifact.model.state_dict()[name], value)
        for name, value in ema.state_dict().items()
    )

    incompatible = _config().model
    incompatible.pixel_decoder.width = 64
    incompatible.pixel_decoder.heads = 4
    with pytest.raises(ValueError, match="architecture mismatch"):
        load_pixel_decoder_artifact(saved, expected_config=incompatible)


def test_pixel_normalization_and_zero_flow_slice_reproduce_pretraining() -> None:
    config = _config()
    decoder = SemanticPixelDecoder(config.model)
    model = build_model(config.model)
    flow_mean = torch.randn(768)
    flow_std = torch.rand(768).add(0.5)
    decoder_mean = torch.randn(768)
    decoder_std = torch.rand(768).add(0.5)
    model.initialize_pretrained_pixel_decoder(
        decoder,
        flow_mean=flow_mean,
        flow_std=flow_std,
        decoder_mean=decoder_mean,
        decoder_std=decoder_std,
    )
    assert torch.equal(model.pixel_semantic_scale, flow_std / decoder_std)
    assert torch.equal(
        model.pixel_semantic_bias, (flow_mean - decoder_mean) / decoder_std
    )
    assert torch.count_nonzero(
        model.pixel_input_projection.weight[:, config.model.token_dim :]
    ) == 0


def test_warmup_freezes_backbone_but_rgb_hidden_path_updates_flow() -> None:
    config = _config()
    decoder = SemanticPixelDecoder(config.model)
    model = build_model(config.model)
    model.initialize_pretrained_pixel_decoder(
        decoder,
        flow_mean=torch.zeros(768),
        flow_std=torch.ones(768),
        decoder_mean=torch.zeros(768),
        decoder_std=torch.ones(768),
    )
    model.set_pixel_detailer(FastDetailer())
    model.set_pixel_decoder_frozen(True)
    with torch.no_grad():
        model.pixel_input_projection.weight[:, 768:].normal_(std=0.02)
        model.pixel_detail_context_projection.weight[:, 768:].normal_(std=0.02)
    output = model(
        torch.randn(1, 64, 768),
        torch.tensor([0.2]),
        torch.tensor([0]),
        decode_pixels=True,
    )
    assert output.rgb is not None
    output.rgb.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.encoder_blocks.parameters())
    assert all(parameter.grad is None for parameter in model.pixel_decoder.parameters())
    assert model.pixel_input_projection.weight.grad is not None
    assert model.full_head.projection.weight.grad is None

    model.zero_grad(set_to_none=True)
    model.set_pixel_decoder_frozen(False)
    output = model(
        torch.randn(1, 64, 768),
        torch.tensor([0.2]),
        torch.tensor([0]),
        decode_pixels=True,
    )
    assert output.rgb is not None
    output.rgb.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.pixel_decoder.parameters())


def test_cached_and_online_pixel_batch_preprocessing() -> None:
    statistics = ChannelStatistics(torch.ones(768), torch.full((768,), 2.0), 64)
    pixels = torch.full((1, 3, 128, 128), 255, dtype=torch.uint8)
    cached = PixelBatchPreprocessor(statistics, "cpu")
    semantic, prepared = cached(
        PixelPretrainingBatch(pixels, torch.full((1, 64, 768), 3.0))
    )
    assert torch.equal(semantic, torch.ones_like(semantic))
    assert torch.equal(prepared, torch.ones_like(prepared))

    online = PixelBatchPreprocessor(statistics, "cpu", OnlineEncoder(), 12)
    online_semantic, _pixels = online(PixelPretrainingBatch(pixels, None))
    assert torch.equal(online_semantic, torch.ones_like(online_semantic))


def test_online_statistics_artifact_roundtrip(tmp_path) -> None:
    artifact = PixelSemanticStatistics(
        dataset_fingerprint="dataset",
        encoder_fingerprint="encoder",
        representation_name="dino12",
        layer=12,
        statistics=_statistics(),
    )
    artifact.save(tmp_path)
    loaded = PixelSemanticStatistics.load(tmp_path)
    assert loaded.dataset_fingerprint == "dataset"
    assert torch.equal(loaded.statistics.mean, artifact.statistics.mean)
    assert torch.equal(loaded.statistics.std, artifact.statistics.std)


def test_unlabeled_dataset_requires_only_an_image_column() -> None:
    source = [{"picture": Image.new("L", (17, 23), color=127)}]
    dataset = UnlabeledImageDataset(source, "picture", 128)
    pixels, index = dataset[0]
    assert index == 0
    assert pixels.shape == (3, 128, 128)
    assert pixels.dtype == torch.uint8

    missing = UnlabeledImageDataset([{"other": Image.new("RGB", (1, 1))}], "picture", 128)
    with pytest.raises(ValueError, match="image column"):
        missing[0]


def test_standalone_pretraining_runner_writes_manually_selectable_epochs(
    tmp_path, monkeypatch
) -> None:
    config = _config()
    config.experiment.output_directory = str(tmp_path)
    config.compile.enabled = False
    config.pixel_pretraining.optimizer.global_batch_size = 1
    config.pixel_pretraining.validation_batches = 1
    config.pixel_pretraining.validation_interval_epochs = 1
    config.pixel_pretraining.checkpoint_interval_epochs = 1
    raw = PixelPretrainingBatch(
        torch.randint(0, 256, (1, 3, 128, 128), dtype=torch.uint8),
        torch.randn(1, 64, 768),
    )

    class OneBatchLoader:
        def __len__(self) -> int:
            return 1

        def __iter__(self):
            return iter([raw])

    monkeypatch.setattr(
        pixel_training_module,
        "prepare_pixel_pretraining_inputs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pixel_training_module,
        "load_pixel_semantic_statistics",
        lambda _config: _statistics(),
    )
    monkeypatch.setattr(
        pixel_training_module,
        "make_pixel_pretraining_loader",
        lambda *_args, **_kwargs: OneBatchLoader(),
    )
    monkeypatch.setattr(
        pixel_training_module,
        "set_pixel_loader_epoch",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(pixel_training_module, "build_lpips", lambda _device: LPIPSStub())

    runner = PixelPretrainingRunner(config)
    runner.run()
    first = tmp_path / "checkpoints" / "epoch-0001"
    second = tmp_path / "checkpoints" / "epoch-0002"
    assert first.is_dir() and second.is_dir()
    assert (tmp_path / "checkpoints" / "last").resolve() == second.resolve()
    selected = load_pixel_decoder_artifact(
        first, expected_config=config.model, weights_kind="ema"
    )
    assert selected.metadata["completed_epoch"] == 1


def test_fresh_joint_runner_loads_artifact_and_freezes_decoder(
    tmp_path, monkeypatch
) -> None:
    config = _config()
    decoder = SemanticPixelDecoder(config.model)
    checkpoint = PixelDecoderCheckpointManager(tmp_path / "artifact").save(
        model=decoder,
        ema=EMA(decoder),
        state=PixelDecoderResumeState(
            schema_version=PIXEL_DECODER_TRAINING_SCHEMA_VERSION,
            completed_epoch=1,
            global_step=1,
            training_complete=False,
            optimizer=None,
            scheduler=None,
            rng_state={"cpu": torch.get_rng_state(), "cuda": None},
            wall_clock_seconds=1.0,
            data_fingerprint="data",
            encoder_fingerprint="encoder",
        ),
        config=config,
        statistics=_statistics(),
    )
    config.training.decoder.pretrained_checkpoint = str(checkpoint)
    config.experiment.output_directory = str(tmp_path / "joint")
    config.model.conditioning.num_classes = 2
    config.validation.periodic_samples = 2
    config.evaluation.num_samples = 2
    manifest = type(
        "Manifest",
        (),
        {
            "statistics": {
                "dino12": {
                    "mean": torch.full((768,), 2.0).tolist(),
                    "std": torch.full((768,), 4.0).tolist(),
                    "count": 128,
                }
            }
        },
    )()
    monkeypatch.setattr(
        runner_module,
        "load_representation_manifest",
        lambda _config: (manifest, "representations"),
    )
    monkeypatch.setattr(runner_module, "build_lpips", lambda _device: LPIPSStub())
    runner = ExperimentRunner(config)
    assert runner.model.pixel_decoder_is_frozen()
    assert len(runner.pixel_decoder_fingerprint) == 64
    assert torch.equal(runner.model.pixel_semantic_scale, torch.full((768,), 4.0))
    assert torch.equal(runner.model.pixel_semantic_bias, torch.full((768,), 2.0))

    joint_checkpoint = runner.manager.save(
        model=runner.model,
        ema=runner.ema,
        state=runner._resume_state(training_complete=False),
        config=config,
    )
    config.training.decoder.pretrained_checkpoint = str(tmp_path / "removed-artifact")
    resumed = ExperimentRunner(config, resume=joint_checkpoint)
    assert resumed.pixel_decoder_fingerprint == runner.pixel_decoder_fingerprint
    assert torch.equal(resumed.model.pixel_semantic_scale, runner.model.pixel_semantic_scale)
