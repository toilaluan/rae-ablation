from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from cascade_dit.checkpoint_bundle import load_checkpoint_bundle
from cascade_dit.config import IMAGE_SIZE, TOKEN_COUNT, TOKEN_DIM
from cascade_dit.distributed import broadcast_string, initialize_distributed
from cascade_dit.evaluation import evaluate_npz, save_evaluation
from cascade_dit.model import build_model
from cascade_dit.model.compile import compile_model_regions
from cascade_dit.sample_runtime import (
    SamplingParameters,
    checkpoint_sha256,
    noise_for_seeds,
    sample_to_archive,
    sampling_output_name,
)
from cascade_dit.sampling import sample_semantic_then_pixel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample a joint semantic-flow DDT checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--solver", choices=("euler", "heun"))
    parser.add_argument("--timestep-shift", type=float)
    parser.add_argument("--internal-guidance-scale", type=float)
    parser.add_argument("--internal-guidance-t-min", type=float)
    parser.add_argument("--internal-guidance-t-max", type=float)
    parser.add_argument("--final-decode-timestep", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--reference-npz", type=Path)
    return parser


def _pick(explicit: object | None, inherited: object) -> object:
    return inherited if explicit is None else explicit


def main() -> None:
    arguments = build_parser().parse_args()
    rank, _world_size, device = initialize_distributed()
    bundle = load_checkpoint_bundle(arguments.checkpoint_dir)
    config, defaults = bundle.config, bundle.config.sampling
    parameters = SamplingParameters(
        steps=int(_pick(arguments.steps, defaults.steps)),
        solver=str(_pick(arguments.solver, defaults.solver)),
        timestep_shift=float(_pick(arguments.timestep_shift, defaults.timestep_shift)),
        internal_guidance_scale=float(
            _pick(arguments.internal_guidance_scale, defaults.internal_guidance.scale)
        ),
        internal_guidance_t_min=float(
            _pick(arguments.internal_guidance_t_min, defaults.internal_guidance.t_min)
        ),
        internal_guidance_t_max=float(
            _pick(arguments.internal_guidance_t_max, defaults.internal_guidance.t_max)
        ),
        final_decode_timestep=float(
            _pick(arguments.final_decode_timestep, defaults.final_decode_timestep)
        ),
        batch_size=int(_pick(arguments.batch_size, defaults.batch_size)),
        seed=int(_pick(arguments.seed, defaults.seed)),
    )
    num_samples = int(_pick(arguments.num_samples, config.evaluation.num_samples))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = build_model(config.model).to(device)
    model.load_state_dict(load_file(bundle.weights, device=str(device)))
    model.eval()
    execution = compile_model_regions(model, config.compile)
    execution_holder = {"model": execution}
    @torch.inference_mode()
    def generate(labels: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
        noise = noise_for_seeds(
            seeds, (TOKEN_COUNT, TOKEN_DIM), device, dtype=dtype
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            pixels = sample_semantic_then_pixel(
                execution_holder["model"],
                noise,
                labels.to(device),
                steps=parameters.steps,
                solver=parameters.solver,
                timestep_shift=parameters.timestep_shift,
                internal_guidance_scale=parameters.internal_guidance_scale,
                internal_guidance_t_min=parameters.internal_guidance_t_min,
                internal_guidance_t_max=parameters.internal_guidance_t_max,
                final_decode_timestep=parameters.final_decode_timestep,
                denominator_floor=config.objective.denominator_floor,
            )
        return pixels.float().clamp(0, 1)

    output = bundle.directory / sampling_output_name(num_samples, parameters)
    checkpoint_hash = checkpoint_sha256(bundle.weights) if rank == 0 else None
    checkpoint_hash = broadcast_string(checkpoint_hash, rank)
    archive = sample_to_archive(
        output_directory=output,
        num_samples=num_samples,
        num_classes=config.model.num_classes,
        image_size=IMAGE_SIZE,
        checkpoint_hash=checkpoint_hash,
        parameters=parameters,
        generate_batch=generate,
    )
    if rank == 0:
        print(archive)
    execution_holder.clear()
    del execution, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if arguments.evaluate and rank == 0:
        reference = arguments.reference_npz or Path(config.evaluation.reference_npz)
        metrics = evaluate_npz(
            archive,
            reference,
            batch_size=config.evaluation.batch_size,
            device=config.evaluation.device,
        )
        sample_manifest = json.loads((output / "samples.json").read_text())
        save_evaluation(
            output / "evaluation.json",
            metrics,
            str(sample_manifest["sampling_plan_hash"]),
            num_samples=num_samples,
        )
        print(metrics)


if __name__ == "__main__":
    main()
