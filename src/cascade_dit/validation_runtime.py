from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .checkpoint import EMA
from .config import IMAGE_LATENT_SHAPE, SEMANTIC_TOKEN_SHAPE, ExperimentConfig
from .distributed import broadcast_string
from .evaluation import evaluate_npz, save_evaluation
from .mage import load_mage_vae_codec
from .model import build_model
from .model.compile import compile_model_regions
from .rng import preserving_global_rng
from .sample_runtime import (
    REPRESENTATION_SCHEMA_VERSION,
    SamplingParameters,
    sample_to_archive,
    sampling_output_name,
    state_dict_sha256,
)
from .sampling import sample_cascade, sample_latent
from .validation import ValidationPlan, noise_for_samples


def _distributed() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


@torch.inference_mode()
@preserving_global_rng
def validate_ema(
    *,
    ema: EMA,
    config: ExperimentConfig,
    plan: ValidationPlan,
    completed_epoch: int,
    final: bool,
    output_root: str | Path,
    sampling: SamplingParameters | Sequence[SamplingParameters],
    representation_provenance: dict[str, Any],
    cache_provenance: dict[str, str],
) -> tuple[dict[str, float], Path]:
    rank, _world_size = _distributed()
    device = torch.device(
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )
    if isinstance(sampling, SamplingParameters):
        parameters = sampling
    else:
        try:
            parameters = next(iter(sampling))
        except StopIteration as error:
            raise ValueError("validation requires sampling parameters") from error
    parameters.validate()
    sample_count = (
        config.validation.final_samples
        if final
        else config.validation.periodic_samples
    )
    run_kind = "final" if final else "periodic"
    root = Path(output_root) / f"epoch-{completed_epoch:04d}-{run_kind}"
    root.mkdir(parents=True, exist_ok=True)
    digest = state_dict_sha256(ema.state_dict()) if rank == 0 else None
    digest = broadcast_string(digest, rank)
    codec = load_mage_vae_codec(config.image_latents, device)
    if codec.provenance != representation_provenance.get("mage_vae"):
        raise ValueError("validation Mage-VAE provenance does not match training")
    model = build_model(config.model).to(device)
    model.load_state_dict(ema.state_dict())
    model.eval()
    execution_model = compile_model_regions(model, config.compile)
    output = root / sampling_output_name(sample_count, parameters)
    representation_schema = {
        "schema_version": REPRESENTATION_SCHEMA_VERSION,
        "architecture": config.model.architecture,
        "image_latents": representation_provenance["mage_vae"],
        "semantic_tokens": representation_provenance.get("semantic_encoder"),
        "training_caches": cache_provenance,
        "conditioning": (
            "class"
            if config.model.architecture == "latent"
            else "patchwise-semantic-adaln-and-class"
        ),
        "prediction": "clean-x0",
    }

    def generate_batch(
        _positions: torch.Tensor, labels: torch.Tensor, noise_seeds: torch.Tensor
    ) -> torch.Tensor:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        image_noise = noise_for_samples(
            noise_seeds, IMAGE_LATENT_SHAPE, device, dtype=dtype, seed_offset=0
        )
        batch_labels = labels.to(device)
        if config.model.architecture == "latent":
            generated = sample_latent(
                execution_model,
                image_noise,
                batch_labels,
                steps=parameters.image_steps,
                timestep_shift=parameters.timestep_shift,
                class_guidance_scale=parameters.class_guidance_scale,
                class_guidance_interval=(
                    parameters.class_guidance_start,
                    parameters.class_guidance_end,
                ),
                solver=parameters.solver,
                num_classes=config.model.num_classes,
                velocity_denominator_floor=config.train.velocity_denominator_floor,
            )
        else:
            semantic_noise = noise_for_samples(
                noise_seeds,
                SEMANTIC_TOKEN_SHAPE,
                device,
                dtype=dtype,
                seed_offset=1_000_003,
            )
            generated = sample_cascade(
                execution_model,
                semantic_noise,
                image_noise,
                batch_labels,
                semantic_steps=parameters.semantic_steps,
                image_steps=parameters.image_steps,
                timestep_shift=parameters.timestep_shift,
                semantic_guidance_scale=parameters.semantic_guidance_scale,
                semantic_guidance_interval=(
                    parameters.semantic_guidance_start,
                    parameters.semantic_guidance_end,
                ),
                class_guidance_scale=parameters.class_guidance_scale,
                class_guidance_interval=(
                    parameters.class_guidance_start,
                    parameters.class_guidance_end,
                ),
                solver=parameters.solver,
                num_classes=config.model.num_classes,
                velocity_denominator_floor=config.train.velocity_denominator_floor,
            )
        return codec.decode(generated.image_latents)

    started = time.perf_counter()
    archive = sample_to_archive(
        output_dir=output,
        plan=plan,
        sample_count=sample_count,
        num_classes=config.model.num_classes,
        image_size=config.image_latents.image_size,
        patch_size=config.model.image_latent_patch_size,
        grid_size=config.model.token_grid_size,
        architecture=config.model.architecture,
        representation_schema=representation_schema,
        checkpoint_hash=digest,
        parameters=parameters,
        generate_batch=generate_batch,
    )
    metrics: dict[str, float] = {
        "throughput/validation_sampling_seconds": time.perf_counter() - started
    }
    if rank == 0:
        raw = evaluate_npz(
            archive,
            config.evaluation.reference_npz,
            batch_size=config.evaluation.batch_size,
            device=(
                device
                if config.evaluation.device == "auto"
                else torch.device(config.evaluation.device)
            ),
            log_interval=config.evaluation.log_interval,
        )
        save_evaluation(output / "evaluation.json", raw, plan.plan_hash)
        suffix = (
            f"{sample_count // 1000}K"
            if sample_count % 1000 == 0
            else str(sample_count)
        )
        metrics[f"validation/FID-{suffix}"] = float(raw["FID"])
    return metrics, output
