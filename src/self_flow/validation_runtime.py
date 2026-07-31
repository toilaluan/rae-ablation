from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .checkpoint import EMA
from .config import ExperimentConfig
from .distributed import broadcast_string
from .evaluation import evaluate_npz, save_evaluation
from .model import SiT
from .model.compile import clean_state_dict, compile_model_regions
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
def validate_model(
    *,
    model: nn.Module,
    ema: EMA | None,
    config: ExperimentConfig,
    plan: ValidationPlan,
    completed_epoch: int,
    final: bool,
    output_root: str | Path,
    sampling: SamplingParameters,
) -> tuple[dict[str, float], Path]:
    sampling.validate()
    rank, _world_size = _distributed()
    device = next(model.parameters()).device
    state = ema.state_dict() if ema is not None else clean_state_dict(model)
    digest = state_dict_sha256(state) if rank == 0 else None
    digest = broadcast_string(digest, rank)
    run_kind = "final" if final else "periodic"
    root = Path(output_root) / f"epoch-{completed_epoch:04d}-{run_kind}"
    count = config.validation.final_samples if final else config.validation.periodic_samples

    evaluation_model = SiT(
        config.model,
        align_layers=config.objective.align_layers,
        projector_hidden_size=config.objective.projector_hidden_size,
    ).to(device)
    evaluation_model.load_state_dict(state)
    evaluation_model.eval()
    execution_model = compile_model_regions(evaluation_model, config.compile)
    codec = load_flux2_codec(config.vae, image_size=config.model.image_size, device=device)
    feature_schema = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "input_space": "flux2_latents",
        "input_shape": list(config.model.input_shape),
        "conditioning": "per-token-adaln-zero",
        "position_encoding": "factorized-2d-rope",
        "objective": asdict(config.objective),
        "vae": {
            "model_name": config.vae.model_name,
            "revision": config.vae.revision,
            "subfolder": config.vae.subfolder,
        },
    }
    output = root / sampling_output_name(count, sampling)

    def generate_batch(
        _positions: torch.Tensor, labels: torch.Tensor, noise_seeds: torch.Tensor
    ) -> torch.Tensor:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        noise = noise_for_samples(noise_seeds, config.model.input_shape, device, dtype=dtype)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            generated = ode_sample(
                execution_model,
                noise,
                labels.to(device),
                steps=sampling.sampling_steps,
                guidance_scale=sampling.guidance_scale,
                timestep_shift=sampling.timestep_shift,
                cfg_interval=(sampling.cfg_start, sampling.cfg_end),
                solver=sampling.solver,
                num_classes=config.model.num_classes,
            )
        return codec.decode(generated)

    archive = sample_to_archive(
        output_dir=output,
        plan=plan,
        sample_count=count,
        num_classes=config.model.num_classes,
        image_size=config.model.image_size,
        patch_size=config.model.patch_size,
        grid_size=config.model.grid_size,
        model_variant=config.model.model_variant,
        feature_schema=feature_schema,
        checkpoint_hash=digest,
        parameters=sampling,
        generate_batch=generate_batch,
    )
    metrics: dict[str, float] = {}
    if rank == 0:
        raw = evaluate_npz(
            archive,
            config.evaluation.reference_npz,
            batch_size=config.evaluation.batch_size,
            device=device if config.evaluation.device == "auto" else config.evaluation.device,
            log_interval=config.evaluation.log_interval,
        )
        save_evaluation(output / "evaluation.json", raw, plan.plan_hash)
        suffix = f"{count // 1000}K" if count % 1000 == 0 else str(count)
        metrics[f"validation/FID-{suffix}"] = float(raw["FID"])
    return metrics, output
