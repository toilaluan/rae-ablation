from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from .checkpoint import EMA
from .config import ExperimentConfig
from .distributed import broadcast_string
from .evaluation import evaluate_npz, save_evaluation
from .model.compile import compile_model_regions
from .model.sit import SiT
from .rng import preserving_global_rng
from .sample_runtime import (
    FEATURE_SCHEMA_VERSION,
    SamplingParameters,
    sample_to_archive,
    sampling_output_name,
    state_dict_sha256,
)
from .sampling import ode_sample
from .vae import load_flux2_codec
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
        config.validation.final_samples if final else config.validation.periodic_samples
    )
    run_kind = "final" if final else "periodic"
    root = Path(output_root) / f"epoch-{completed_epoch:04d}-{run_kind}"
    root.mkdir(parents=True, exist_ok=True)
    digest = state_dict_sha256(ema.state_dict()) if rank == 0 else None
    digest = broadcast_string(digest, rank)
    feature_schema = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "input_space": config.model.input_space,
        "input_shape": list(config.model.input_shape),
        "conditioning": "timestep_plus_class_vector",
        "prediction": "x0",
        "repa": asdict(config.repa),
        "vae": (
            {
                "model_name": config.vae.model_name,
                "revision": config.vae.revision,
                "subfolder": config.vae.subfolder,
            }
            if config.model.input_space == "flux2_latents"
            else None
        ),
    }
    codec = (
        load_flux2_codec(config.vae, image_size=config.model.image_size, device=device)
        if config.model.input_space == "flux2_latents"
        else None
    )
    model = SiT(config.model, config.repa).to(device)
    model.load_state_dict(ema.state_dict())
    model.eval()
    execution_model = compile_model_regions(model, config.compile)
    output = root / sampling_output_name(sample_count, parameters)

    def generate_batch(
        _positions: torch.Tensor,
        labels: torch.Tensor,
        noise_seeds: torch.Tensor,
    ) -> torch.Tensor:
        noise = noise_for_samples(
            noise_seeds,
            config.model.input_shape,
            device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        with torch.autocast(
            device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            generated = ode_sample(
                execution_model,
                noise,
                labels.to(device),
                steps=parameters.sampling_steps,
                guidance_scale=parameters.guidance_scale,
                timestep_shift=parameters.timestep_shift,
                cfg_interval=(parameters.cfg_start, parameters.cfg_end),
                solver=parameters.solver,
                num_classes=config.model.num_classes,
                velocity_denominator_floor=config.train.velocity_denominator_floor,
            )
        if codec is not None:
            return codec.decode(generated)
        return generated.float().add(1).div(2)

    started = time.perf_counter()
    archive = sample_to_archive(
        output_dir=output,
        plan=plan,
        sample_count=sample_count,
        num_classes=config.model.num_classes,
        image_size=config.model.image_size,
        patch_size=config.model.patch_size,
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
        suffix = f"{sample_count // 1000}K" if sample_count % 1000 == 0 else str(sample_count)
        metrics[f"validation/FID-{suffix}"] = float(raw["FID"])
    return metrics, output
