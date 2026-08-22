from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cascade_dit.checkpoint import EMA, CheckpointManager
from cascade_dit.config import ExperimentConfig, load_config
from cascade_dit.data import data_fingerprint, load_imagenet, preprocessing_metadata
from cascade_dit.distributed import initialize_distributed
from cascade_dit.image_latent_cache import ImageLatentCache
from cascade_dit.mage import (
    file_sha256,
    load_mage_vae_codec,
    mage_vae_provenance,
)
from cascade_dit.model import build_model
from cascade_dit.model.compile import compile_model_regions
from cascade_dit.optim import build_optimizer
from cascade_dit.rng import RNGStreams
from cascade_dit.sample_runtime import SamplingParameters
from cascade_dit.semantic_statistics import (
    SemanticTokenNormalizer,
    load_semantic_statistics,
    semantic_statistics_sha256,
)
from cascade_dit.semantic_token_cache import SemanticTokenCache
from cascade_dit.training import (
    TrainState,
    make_train_loader,
    train,
    training_requires_pixels,
)
from cascade_dit.validation import ValidationPlan, create_validation_plan
from cascade_dit.validation_runtime import validate_ema
from cascade_dit.vision import (
    load_dino_semantic_encoder,
    semantic_encoder_fingerprint,
)
from cascade_dit.wandb_utils import RunLogger

torch._dynamo.config.cache_size_limit = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CascadeDiT on native Mage latents")
    parser.add_argument("--config", type=Path, default=Path("configs/cascade-shared.yaml"))
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def _sampling_parameters(config: ExperimentConfig) -> SamplingParameters:
    values = config.sampling
    return SamplingParameters(
        batch_size=values.batch_size,
        semantic_steps=(
            0 if config.model.architecture == "latent" else values.semantic_steps
        ),
        image_steps=values.image_steps,
        solver=values.solver,
        timestep_shift=values.timestep_shift,
        semantic_guidance_scale=values.semantic_guidance_scale,
        semantic_guidance_start=values.semantic_guidance_start,
        semantic_guidance_end=values.semantic_guidance_end,
        class_guidance_scale=values.class_guidance_scale,
        class_guidance_start=values.class_guidance_start,
        class_guidance_end=values.class_guidance_end,
        log_interval=values.log_interval,
    )


def _parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    semantic = getattr(model, "semantic_dit", None)
    image = getattr(model, "image_latent_dit", None)
    if semantic is not None and image is not None:
        counts["semantic_dit"] = sum(p.numel() for p in semantic.parameters())
        counts["conditional_latent_dit"] = sum(p.numel() for p in image.parameters())
    return counts


def main() -> None:
    from huggingface_hub import hf_hub_download

    args = build_parser().parse_args()
    config = load_config(args.config, args.override)
    if args.output:
        config.checkpoint.directory = str(args.output / "checkpoints")
        config.validation.plan_path = str(args.output / "validation-plan.pt")
    config.validate()
    rank, world_size, device = initialize_distributed()
    mage_path = hf_hub_download(
        repo_id=config.image_latents.repository,
        filename=config.image_latents.weight_file,
        revision=config.image_latents.revision,
    )
    mage_digest = file_sha256(mage_path)
    mage_provenance = mage_vae_provenance(config.image_latents, mage_digest)
    cascade = config.model.architecture != "latent"
    statistics = None
    statistics_digest = "not-used"
    semantic_provenance: dict[str, object] | None = None
    if cascade:
        statistics_path = Path(config.semantic_tokens.statistics_path).resolve()
        config.semantic_tokens.statistics_path = str(statistics_path)
        statistics = load_semantic_statistics(
            statistics_path,
            expected_channels=config.semantic_tokens.channels,
            expected_semantic_encoder_fingerprint=semantic_encoder_fingerprint(
                config.semantic_tokens
            ),
            expected_data_fingerprint=data_fingerprint(
                config.data, config.image_latents.image_size
            ),
        )
        statistics_digest = semantic_statistics_sha256(statistics_path)
        semantic_provenance = {
            "repository": config.semantic_tokens.repository,
            "revision": config.semantic_tokens.revision,
            "fingerprint": semantic_encoder_fingerprint(config.semantic_tokens),
            "statistics_sha256": statistics_digest,
            "shape": list(config.semantic_tokens.shape),
            "preprocessing": "256px-view-resized-to-128px-bicubic-imagenet-normalized",
        }
    needs_pixels = not config.image_latents.cache.enabled or (
        cascade and not config.semantic_tokens.cache.enabled
    )
    dataset = load_imagenet(
        config.data,
        config.image_latents.image_size,
        load_pixels=needs_pixels,
    )
    image_cache = (
        ImageLatentCache(config, len(dataset), mage_digest)
        if config.image_latents.cache.enabled
        else None
    )
    semantic_cache = (
        SemanticTokenCache(config, len(dataset), statistics_digest)
        if cascade and config.semantic_tokens.cache.enabled
        else None
    )
    if needs_pixels != training_requires_pixels(
        config.model.architecture, image_cache, semantic_cache
    ):
        raise RuntimeError("dataset pixel-loading policy is inconsistent with caches")
    mage_codec = (
        None
        if image_cache is not None
        else load_mage_vae_codec(config.image_latents, device)
    )
    semantic_encoder = (
        load_dino_semantic_encoder(
            config.semantic_tokens,
            device,
            normalizer=SemanticTokenNormalizer(statistics),
        )
        if cascade and semantic_cache is None and statistics is not None
        else None
    )
    streams = RNGStreams.create(config.train.seed, device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(streams.initialization.initial_seed())
        model = build_model(config.model).to(device)
    fingerprint = data_fingerprint(config.data, config.image_latents.image_size)
    representation_provenance: dict[str, object] = {"mage_vae": mage_provenance}
    if semantic_provenance is not None:
        representation_provenance["semantic_encoder"] = semantic_provenance
    cache_provenance = {
        "image_latents": image_cache.fingerprint if image_cache is not None else "online",
        "semantic_tokens": (
            "not-used"
            if not cascade
            else semantic_cache.fingerprint
            if semantic_cache is not None
            else "online"
        ),
    }
    plan_path = Path(config.validation.plan_path)
    if rank == 0 and not plan_path.exists():
        create_validation_plan(
            config.validation.final_samples,
            config.validation.periodic_samples,
            config.model.num_classes,
            config.train.seed + 7,
        ).save(plan_path)
    if world_size > 1:
        torch.distributed.barrier()
    plan = ValidationPlan.load(plan_path)
    ema = EMA(model, config.train.ema_decay)
    optimizer = build_optimizer(model, config.train.optimizer)
    manager = CheckpointManager(
        config.checkpoint.directory, config.checkpoint.keep_last_n
    )
    state = TrainState()
    resume_run_id: str | None = None
    if args.resume:
        resumed = manager.load(
            args.resume.resolve(),
            model,
            ema,
            expected_architecture=config.model.architecture,
            expected_data_fingerprint=fingerprint,
            expected_validation_plan_hash=plan.plan_hash,
            expected_representation_provenance=representation_provenance,
            expected_cache_provenance=cache_provenance,
        )
        optimizer.load_state_dict(resumed.optimizer)
        streams.load_state_dict(resumed.rng_streams)
        state = TrainState(
            resumed.epoch,
            resumed.global_step,
            resumed.last_completed_validation_epoch,
        )
        resume_run_id = resumed.wandb_run_id
    execution_model = compile_model_regions(model, config.compile)
    if world_size > 1:
        execution_model = torch.nn.parallel.DistributedDataParallel(
            execution_model,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
    loader, sampler = make_train_loader(
        dataset, config, streams, image_cache, semantic_cache
    )
    logged_config = config.as_dict()
    logged_config["provenance"] = {
        "dataset_and_preprocessing": preprocessing_metadata(
            config.data, config.image_latents.image_size
        ),
        "data_fingerprint": fingerprint,
        "validation_plan_hash": plan.plan_hash,
        "representations": representation_provenance,
        "caches": cache_provenance,
    }
    logged_config["model_parameters"] = _parameter_counts(model)
    logger = RunLogger(
        config.wandb, logged_config, rank == 0, resume_run_id, args.log_interval
    )
    logger.event(
        f"training initialized: architecture={config.model.architecture}, "
        f"image_latents={cache_provenance['image_latents']}, "
        f"semantic_tokens={cache_provenance['semantic_tokens']}, "
        f"parameters={logged_config['model_parameters']['total']:,}"
    )
    sampling = _sampling_parameters(config)
    output_root = Path(config.checkpoint.directory).parent / "validation"

    def validate_callback(epoch: int, final: bool) -> dict[str, float]:
        metrics, output = validate_ema(
            ema=ema,
            config=config,
            plan=plan,
            completed_epoch=epoch,
            final=final,
            output_root=output_root,
            sampling=sampling,
            representation_provenance=representation_provenance,
            cache_provenance=cache_provenance,
        )
        preview = output / "preview.png"
        if preview.is_file():
            logger.log_grid(str(preview), f"EMA epoch {epoch}", state.global_step)
        return metrics

    try:
        train(
            canonical_model=model,
            execution_model=execution_model,
            mage_codec=mage_codec,
            semantic_encoder=semantic_encoder,
            loader=loader,
            sampler=sampler,
            optimizer=optimizer,
            ema=ema,
            streams=streams,
            config=config,
            checkpoint_manager=manager,
            data_fingerprint=fingerprint,
            representation_provenance=representation_provenance,
            cache_provenance=cache_provenance,
            validation_plan_hash=plan.plan_hash,
            validation_plan_path=plan_path,
            logger=logger,
            state=state,
            validation_callback=validate_callback,
        )
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
