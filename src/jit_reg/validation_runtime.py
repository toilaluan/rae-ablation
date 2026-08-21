from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import torch

from .checkpoint import EMA
from .config import ExperimentConfig
from .distributed import broadcast_string
from .evaluation import evaluate_npz, save_evaluation
from .latent_statistics import latent_statistics_sha256
from .model import DDT
from .model.compile import compile_model_regions
from .pixel_decoder import load_pixel_decoder
from .rng import preserving_global_rng
from .sample_runtime import (
    FEATURE_SCHEMA_VERSION,
    SamplingParameters,
    sample_to_archive,
    sampling_output_name,
    state_dict_sha256,
)
from .sampling import ode_sample
from .validation import ValidationPlan, noise_for_samples
from .vision import vision_fingerprint


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
    endpoint_source: str = "online",
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
    decoder, decoder_digest = load_pixel_decoder(config.decoder, device)
    statistics_digest = latent_statistics_sha256(config.vision.statistics_path)
    feature_schema = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "input_space": (
            f"dinov3-b16-normalized-block{config.vision.target_layer}-patch-tokens"
        ),
        "input_shape": list(config.model.input_shape),
        "vision_fingerprint": vision_fingerprint(config.vision),
        "decoder_sha256": decoder_digest,
        "latent_statistics_sha256": statistics_digest,
        "training_endpoint_source": endpoint_source,
        "conditioning": "timestep-waypoint-stage-class",
        "prediction": (
            "clean-x-full-and-base-with-repa"
            if config.repa.enabled
            else "clean-x-full-and-base"
        ),
        "waypoint": {
            "time": config.waypoint.time,
            "pool_size": config.waypoint.pool_size,
            "downsample_mode": config.waypoint.downsample_mode,
            "upsample_mode": config.waypoint.upsample_mode,
        },
    }
    model = DDT(config.model, config.repa).to(device)
    model.load_state_dict(ema.state_dict())
    model.eval()
    execution_model = compile_model_regions(model, config.compile)
    output = root / sampling_output_name(sample_count, parameters)

    def generate_batch(
        _positions: torch.Tensor, labels: torch.Tensor, noise_seeds: torch.Tensor
    ) -> torch.Tensor:
        noise = noise_for_samples(
            noise_seeds,
            config.model.input_shape,
            device,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        generated = ode_sample(
            execution_model,
            noise,
            labels.to(device),
            semantic_steps=parameters.semantic_steps,
            target_steps=parameters.target_steps,
            waypoint_time=parameters.waypoint_time,
            timestep_shift=parameters.timestep_shift,
            ig_scale=parameters.ig_scale,
            ig_interval=(parameters.ig_start, parameters.ig_end),
            cfg_scale=parameters.cfg_scale,
            cfg_interval=(parameters.cfg_start, parameters.cfg_end),
            solver=parameters.solver,
            num_classes=config.model.num_classes,
            velocity_denominator_floor=config.train.velocity_denominator_floor,
        )
        return decoder(generated.to(next(decoder.parameters()).dtype)).float().clamp(0, 1)

    started = time.perf_counter()
    archive = sample_to_archive(
        output_dir=output,
        plan=plan,
        sample_count=sample_count,
        num_classes=config.model.num_classes,
        image_size=config.vision.image_size,
        patch_size=config.vision.patch_size,
        grid_size=config.model.grid_size,
        model_variant=config.model.model_variant,
        feature_schema=feature_schema,
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
