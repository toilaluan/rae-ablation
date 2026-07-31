from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file

from self_flow.checkpoint_bundle import load_checkpoint_bundle
from self_flow.distributed import broadcast_string, initialize_distributed
from self_flow.evaluation import evaluate_npz, save_evaluation
from self_flow.model import SiT
from self_flow.model.compile import compile_model_regions
from self_flow.sample_runtime import (
    FEATURE_SCHEMA_VERSION,
    SamplingParameters,
    checkpoint_sha256,
    sample_to_archive,
    sampling_output_name,
)
from self_flow.sampling import ode_sample
from self_flow.vae import load_flux2_codec
from self_flow.validation import noise_for_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample a Self-Flow checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--solver", choices=("euler", "heun"))
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--timestep-shift", type=float)
    parser.add_argument("--cfg-start", type=float)
    parser.add_argument("--cfg-end", type=float)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--reference-npz", type=Path)
    return parser


def _value(explicit: object | None, inherited: object) -> object:
    return inherited if explicit is None else explicit


def main() -> None:
    args = build_parser().parse_args()
    bundle = load_checkpoint_bundle(args.checkpoint_dir)
    config = bundle.config
    defaults = config.sampling
    sample_count = int(_value(args.num_samples, defaults.num_samples))
    parameters = SamplingParameters(
        batch_size=int(_value(args.batch_size, defaults.batch_size)),
        sampling_steps=int(_value(args.sampling_steps, defaults.sampling_steps)),
        solver=str(_value(args.solver, defaults.solver)),
        guidance_scale=float(_value(args.guidance_scale, defaults.guidance_scale)),
        timestep_shift=float(_value(args.timestep_shift, defaults.timestep_shift)),
        cfg_start=float(_value(args.cfg_start, defaults.cfg_start)),
        cfg_end=float(_value(args.cfg_end, defaults.cfg_end)),
        log_interval=int(_value(args.log_interval, defaults.log_interval)),
    )
    parameters.validate()
    rank, _world_size, device = initialize_distributed()
    digest = checkpoint_sha256(bundle.weights) if rank == 0 else None
    digest = broadcast_string(digest, rank)
    model = SiT(
        config.model,
        align_layers=config.objective.align_layers,
        projector_hidden_size=config.objective.projector_hidden_size,
    ).to(device)
    model.load_state_dict(load_file(str(bundle.weights), device=str(device)))
    model.eval()
    execution_model = compile_model_regions(model, config.compile)
    codec = load_flux2_codec(config.vae, image_size=config.model.image_size, device=device)

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
                steps=parameters.sampling_steps,
                guidance_scale=parameters.guidance_scale,
                timestep_shift=parameters.timestep_shift,
                cfg_interval=(parameters.cfg_start, parameters.cfg_end),
                solver=parameters.solver,
                num_classes=config.model.num_classes,
            )
        return codec.decode(generated)

    output_dir = bundle.directory / sampling_output_name(sample_count, parameters)
    archive = sample_to_archive(
        output_dir=output_dir,
        plan=bundle.plan,
        sample_count=sample_count,
        num_classes=config.model.num_classes,
        image_size=config.model.image_size,
        patch_size=config.model.patch_size,
        grid_size=config.model.grid_size,
        model_variant=config.model.model_variant,
        feature_schema={
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
        },
        checkpoint_hash=digest,
        parameters=parameters,
        generate_batch=generate_batch,
    )
    if args.evaluate and rank == 0:
        reference = args.reference_npz or Path(config.evaluation.reference_npz)
        metrics = evaluate_npz(
            archive,
            reference,
            batch_size=config.evaluation.batch_size,
            device=device if config.evaluation.device == "auto" else config.evaluation.device,
            log_interval=config.evaluation.log_interval,
        )
        save_evaluation(output_dir / "evaluation.json", metrics, bundle.plan.plan_hash)


if __name__ == "__main__":
    main()
