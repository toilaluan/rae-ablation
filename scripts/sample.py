from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from cascade_dit.checkpoint_bundle import load_checkpoint_bundle
from cascade_dit.config import IMAGE_LATENT_SHAPE, SEMANTIC_TOKEN_SHAPE
from cascade_dit.distributed import broadcast_string, initialize_distributed
from cascade_dit.evaluation import evaluate_npz, save_evaluation
from cascade_dit.mage import load_mage_vae_codec
from cascade_dit.model import build_model
from cascade_dit.model.compile import compile_model_regions
from cascade_dit.sample_runtime import (
    REPRESENTATION_SCHEMA_VERSION,
    SamplingParameters,
    checkpoint_sha256,
    sample_to_archive,
    sampling_output_name,
)
from cascade_dit.sampling import sample_cascade, sample_latent
from cascade_dit.validation import noise_for_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample a CascadeDiT checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--semantic-steps", type=int)
    parser.add_argument("--image-steps", type=int)
    parser.add_argument("--solver", choices=("euler", "heun"))
    parser.add_argument("--timestep-shift", type=float)
    parser.add_argument("--semantic-guidance-scale", type=float)
    parser.add_argument("--semantic-guidance-start", type=float)
    parser.add_argument("--semantic-guidance-end", type=float)
    parser.add_argument("--class-guidance-scale", type=float)
    parser.add_argument("--class-guidance-start", type=float)
    parser.add_argument("--class-guidance-end", type=float)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--reference-npz", type=Path)
    return parser


def _value(explicit: object | None, inherited: object) -> object:
    return inherited if explicit is None else explicit


def main() -> None:
    args = build_parser().parse_args()
    bundle = load_checkpoint_bundle(args.checkpoint_dir)
    config, defaults = bundle.config, bundle.config.sampling
    sample_count = int(_value(args.num_samples, defaults.num_samples))
    parameters = SamplingParameters(
        batch_size=int(_value(args.batch_size, defaults.batch_size)),
        semantic_steps=(
            0
            if config.model.architecture == "latent"
            else int(_value(args.semantic_steps, defaults.semantic_steps))
        ),
        image_steps=int(_value(args.image_steps, defaults.image_steps)),
        solver=str(_value(args.solver, defaults.solver)),
        timestep_shift=float(_value(args.timestep_shift, defaults.timestep_shift)),
        semantic_guidance_scale=float(
            _value(
                args.semantic_guidance_scale, defaults.semantic_guidance_scale
            )
        ),
        semantic_guidance_start=float(
            _value(
                args.semantic_guidance_start, defaults.semantic_guidance_start
            )
        ),
        semantic_guidance_end=float(
            _value(args.semantic_guidance_end, defaults.semantic_guidance_end)
        ),
        class_guidance_scale=float(
            _value(args.class_guidance_scale, defaults.class_guidance_scale)
        ),
        class_guidance_start=float(
            _value(args.class_guidance_start, defaults.class_guidance_start)
        ),
        class_guidance_end=float(
            _value(args.class_guidance_end, defaults.class_guidance_end)
        ),
        log_interval=int(_value(args.log_interval, defaults.log_interval)),
    )
    parameters.validate()
    rank, _world_size, device = initialize_distributed()
    digest = checkpoint_sha256(bundle.weights) if rank == 0 else None
    digest = broadcast_string(digest, rank)
    metadata = json.loads((bundle.directory / "metadata.json").read_text())
    if metadata.get("architecture") != config.model.architecture:
        raise ValueError("checkpoint architecture does not match its config")
    representations = metadata.get("representation_provenance")
    if not isinstance(representations, dict):
        raise ValueError("checkpoint representation provenance is missing")
    model = build_model(config.model).to(device)
    model.load_state_dict(load_file(str(bundle.weights), device=str(device)))
    model.eval()
    execution_model = compile_model_regions(model, config.compile)
    codec = load_mage_vae_codec(config.image_latents, device)
    if codec.provenance != representations.get("mage_vae"):
        raise ValueError("sampling Mage-VAE does not match checkpoint provenance")

    @torch.inference_mode()
    def generate_batch(
        _positions: torch.Tensor, labels: torch.Tensor, noise_seeds: torch.Tensor
    ) -> torch.Tensor:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        batch_labels = labels.to(device)
        image_noise = noise_for_samples(
            noise_seeds, IMAGE_LATENT_SHAPE, device, dtype=dtype, seed_offset=0
        )
        common = {
            "steps": parameters.image_steps,
            "timestep_shift": parameters.timestep_shift,
            "class_guidance_scale": parameters.class_guidance_scale,
            "class_guidance_interval": (
                parameters.class_guidance_start,
                parameters.class_guidance_end,
            ),
            "solver": parameters.solver,
            "num_classes": config.model.num_classes,
            "velocity_denominator_floor": config.train.velocity_denominator_floor,
        }
        if config.model.architecture == "latent":
            generated = sample_latent(
                execution_model, image_noise, batch_labels, **common
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

    representation_schema = {
        "schema_version": REPRESENTATION_SCHEMA_VERSION,
        "architecture": config.model.architecture,
        "image_latents": representations["mage_vae"],
        "semantic_tokens": representations.get("semantic_encoder"),
        "training_caches": metadata.get("cache_provenance"),
        "conditioning": (
            "class"
            if config.model.architecture == "latent"
            else "patchwise-semantic-adaln-and-class"
        ),
        "prediction": "clean-x0",
    }
    output_dir = bundle.directory / sampling_output_name(sample_count, parameters)
    archive = sample_to_archive(
        output_dir=output_dir,
        plan=bundle.plan,
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
    if args.evaluate and rank == 0:
        reference = args.reference_npz or Path(config.evaluation.reference_npz)
        evaluation_device = (
            device
            if config.evaluation.device == "auto"
            else torch.device(config.evaluation.device)
        )
        metrics = evaluate_npz(
            archive,
            reference,
            batch_size=config.evaluation.batch_size,
            device=evaluation_device,
            log_interval=config.evaluation.log_interval,
        )
        save_evaluation(output_dir / "evaluation.json", metrics, bundle.plan.plan_hash)


if __name__ == "__main__":
    main()
