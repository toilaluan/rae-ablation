from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from jit_reg.checkpoint_bundle import load_checkpoint_bundle
from jit_reg.distributed import broadcast_string, initialize_distributed
from jit_reg.evaluation import evaluate_npz, save_evaluation
from jit_reg.model import DDT
from jit_reg.model.compile import compile_model_regions
from jit_reg.pixel_decoder import decoder_latent_statistics_sha256, load_pixel_decoder
from jit_reg.sample_runtime import (
    FEATURE_SCHEMA_VERSION,
    SamplingParameters,
    checkpoint_sha256,
    sample_to_archive,
    sampling_output_name,
)
from jit_reg.sampling import ode_sample
from jit_reg.validation import noise_for_samples
from jit_reg.vision import vision_fingerprint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample a schema-v15 DINO waypoint checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--semantic-steps", type=int)
    parser.add_argument("--target-steps", type=int)
    parser.add_argument("--solver", choices=("euler", "heun"))
    parser.add_argument("--timestep-shift", type=float)
    parser.add_argument("--ig-scale", type=float)
    parser.add_argument("--ig-start", type=float)
    parser.add_argument("--ig-end", type=float)
    parser.add_argument("--cfg-scale", type=float)
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
    config, defaults = bundle.config, bundle.config.sampling
    sample_count = int(_value(args.num_samples, defaults.num_samples))
    parameters = SamplingParameters(
        batch_size=int(_value(args.batch_size, defaults.batch_size)),
        semantic_steps=int(_value(args.semantic_steps, defaults.semantic_steps)),
        target_steps=int(_value(args.target_steps, defaults.target_steps)),
        waypoint_time=config.waypoint.time,
        solver=str(_value(args.solver, defaults.solver)),
        timestep_shift=float(_value(args.timestep_shift, defaults.timestep_shift)),
        ig_scale=float(_value(args.ig_scale, defaults.ig_scale)),
        ig_start=float(_value(args.ig_start, defaults.ig_start)),
        ig_end=float(_value(args.ig_end, defaults.ig_end)),
        cfg_scale=float(_value(args.cfg_scale, defaults.cfg_scale)),
        cfg_start=float(_value(args.cfg_start, defaults.cfg_start)),
        cfg_end=float(_value(args.cfg_end, defaults.cfg_end)),
        log_interval=int(_value(args.log_interval, defaults.log_interval)),
    )
    parameters.validate()
    rank, _world_size, device = initialize_distributed()
    digest = checkpoint_sha256(bundle.weights) if rank == 0 else None
    digest = broadcast_string(digest, rank)
    metadata = json.loads((bundle.directory / "metadata.json").read_text())
    model = DDT(config.model, config.repa).to(device)
    model.load_state_dict(load_file(str(bundle.weights), device=str(device)))
    model.eval()
    execution_model = compile_model_regions(model, config.compile)
    decoder, decoder_digest = load_pixel_decoder(config.decoder, device)
    if metadata.get("decoder_sha256") != decoder_digest:
        raise ValueError("configured decoder does not match checkpoint provenance")
    statistics_digest = decoder_latent_statistics_sha256(config.decoder.checkpoint)
    if metadata.get("latent_statistics_sha256") != statistics_digest:
        raise ValueError("decoder and diffusion checkpoints use different latent statistics")
    if metadata.get("feature_fingerprint") != vision_fingerprint(config.vision):
        raise ValueError("checkpoint DINO feature provenance does not match its config")

    @torch.inference_mode()
    def generate_batch(
        _positions: torch.Tensor, labels: torch.Tensor, noise_seeds: torch.Tensor
    ) -> torch.Tensor:
        batch_labels = labels.to(device)
        noise = noise_for_samples(
            noise_seeds,
            config.model.input_shape,
            device,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        generated = ode_sample(
            execution_model,
            noise,
            batch_labels,
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

    output_dir = bundle.directory / sampling_output_name(sample_count, parameters)
    archive = sample_to_archive(
        output_dir=output_dir,
        plan=bundle.plan,
        sample_count=sample_count,
        num_classes=config.model.num_classes,
        image_size=config.vision.image_size,
        patch_size=config.vision.patch_size,
        grid_size=config.model.grid_size,
        model_variant=config.model.model_variant,
        feature_schema={
            "schema_version": FEATURE_SCHEMA_VERSION,
            "input_space": (
                f"dinov3-b16-normalized-block{config.vision.target_layer}-patch-tokens"
            ),
            "input_shape": list(config.model.input_shape),
            "vision_fingerprint": vision_fingerprint(config.vision),
            "decoder_sha256": decoder_digest,
            "latent_statistics_sha256": statistics_digest,
            "training_endpoint_source": metadata.get("endpoint_cache_fingerprint"),
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
        },
        checkpoint_hash=digest,
        parameters=parameters,
        generate_batch=generate_batch,
    )
    if args.evaluate and rank == 0:
        reference = args.reference_npz or Path(config.evaluation.reference_npz)
        eval_device = (
            device
            if config.evaluation.device == "auto"
            else torch.device(config.evaluation.device)
        )
        metrics = evaluate_npz(
            archive,
            reference,
            batch_size=config.evaluation.batch_size,
            device=eval_device,
            log_interval=config.evaluation.log_interval,
        )
        save_evaluation(output_dir / "evaluation.json", metrics, bundle.plan.plan_hash)


if __name__ == "__main__":
    main()
