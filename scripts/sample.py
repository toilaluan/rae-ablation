from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from cascade_dit.checkpoint_bundle import load_checkpoint_bundle
from cascade_dit.config import IMAGE_SIZE, TOKEN_COUNT, TOKEN_DIM
from cascade_dit.evaluation import evaluate_npz, save_evaluation
from cascade_dit.model import build_model
from cascade_dit.model.compile import compile_model_regions
from cascade_dit.preparation import statistics_from_manifest
from cascade_dit.representations import RepresentationManifest, unpatchify_rgb
from cascade_dit.sample_runtime import (
    SamplingParameters,
    checkpoint_sha256,
    noise_for_seeds,
    sample_to_archive,
    sampling_output_name,
)
from cascade_dit.sampling import sample_tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample a completed RGB DDT checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--solver", choices=("euler", "heun"))
    parser.add_argument("--timestep-shift", type=float)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--guidance-start", type=float)
    parser.add_argument("--guidance-end", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--reference-npz", type=Path)
    return parser


def _pick(explicit: object | None, inherited: object) -> object:
    return inherited if explicit is None else explicit


def main() -> None:
    arguments = build_parser().parse_args()
    bundle = load_checkpoint_bundle(arguments.checkpoint_dir, require_rgb_complete=True)
    config, defaults = bundle.config, bundle.config.sampling
    parameters = SamplingParameters(
        steps=int(_pick(arguments.steps, defaults.steps)),
        solver=str(_pick(arguments.solver, defaults.solver)),
        timestep_shift=float(_pick(arguments.timestep_shift, defaults.timestep_shift)),
        guidance_scale=float(_pick(arguments.guidance_scale, defaults.guidance_scale)),
        guidance_start=float(_pick(arguments.guidance_start, defaults.guidance_start)),
        guidance_end=float(_pick(arguments.guidance_end, defaults.guidance_end)),
        batch_size=int(_pick(arguments.batch_size, defaults.batch_size)),
        seed=int(_pick(arguments.seed, defaults.seed)),
    )
    num_samples = int(_pick(arguments.num_samples, config.evaluation.num_samples))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = build_model(config.model, config.num_representation_levels).to(device)
    model.load_state_dict(load_file(bundle.weights, device=str(device)))
    model.eval()
    execution = compile_model_regions(model, config.compile)
    root = Path(config.representations.directory)
    manifest = RepresentationManifest.load(root / config.representations.manifest)
    rgb_statistics = statistics_from_manifest(manifest, config.stage_names[-1])

    @torch.inference_mode()
    def generate(labels: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
        noise = noise_for_seeds(
            seeds, (TOKEN_COUNT, TOKEN_DIM), device, dtype=dtype
        )
        weights = torch.zeros(
            labels.shape[0], config.num_representation_levels, device=device
        )
        weights[:, -1] = 1
        normalized = sample_tokens(
            execution,
            noise,
            labels.to(device),
            weights,
            steps=parameters.steps,
            solver=parameters.solver,
            timestep_shift=parameters.timestep_shift,
            guidance_scale=parameters.guidance_scale,
            guidance_start=parameters.guidance_start,
            guidance_end=parameters.guidance_end,
            denominator_floor=config.objective.denominator_floor,
            num_classes=config.model.num_classes,
        )
        tokens = rgb_statistics.denormalize(normalized.float())
        return unpatchify_rgb(tokens).clamp(0, 1)

    output = bundle.directory / sampling_output_name(num_samples, parameters)
    archive = sample_to_archive(
        output_directory=output,
        num_samples=num_samples,
        num_classes=config.model.num_classes,
        image_size=IMAGE_SIZE,
        checkpoint_hash=checkpoint_sha256(bundle.weights),
        parameters=parameters,
        generate_batch=generate,
    )
    print(archive)
    if arguments.evaluate:
        reference = arguments.reference_npz or Path(config.evaluation.reference_npz)
        metrics = evaluate_npz(
            archive,
            reference,
            batch_size=config.evaluation.batch_size,
            device=config.evaluation.device,
        )
        sample_manifest = json.loads((output / "samples.json").read_text())
        save_evaluation(
            output / "evaluation.json", metrics, str(sample_manifest["sampling_plan_hash"])
        )
        print(metrics)


if __name__ == "__main__":
    main()
