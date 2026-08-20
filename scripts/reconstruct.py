from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from jit_reg.config import IMAGE_SIZE
from jit_reg.pixel_codec import load_pixel_codec_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct one image with a trained pixel codec"
    )
    parser.add_argument("image", type=Path, help="input image")
    parser.add_argument(
        "--codec",
        type=Path,
        default=Path("artifacts/pixel-codec/best"),
        help="pixel codec artifact directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reconstruction.png"),
        help="reconstructed image path (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device, or 'auto' to use CUDA when available (default: %(default)s)",
    )
    return parser


def resolve_device(value: str) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_image(path: Path, device: torch.device) -> tuple[Tensor, tuple[int, int]]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        original_size = image.size
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.uint8).copy()
    pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return pixels.to(device=device, dtype=torch.float32).div_(255), original_size


def tensor_statistics(tensor: Tensor) -> dict[str, float]:
    values = tensor.detach().float().flatten()
    quantiles = torch.quantile(values, torch.tensor([0.01, 0.5, 0.99], device=values.device))
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "min": values.min().item(),
        "p01": quantiles[0].item(),
        "median": quantiles[1].item(),
        "p99": quantiles[2].item(),
        "max": values.max().item(),
        "abs_mean": values.abs().mean().item(),
        "rms": values.square().mean().sqrt().item(),
    }


def save_image(pixels: Tensor, path: Path) -> None:
    array = (
        pixels[0]
        .detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    artifact = load_pixel_codec_artifact(args.codec, device)
    pixels, original_size = load_image(args.image, device)

    with torch.inference_mode():
        latent = artifact.model.encode(pixels)
        normalized_latent = artifact.statistics.normalize(latent)
        reconstruction = artifact.model.decode(latent)

    save_image(reconstruction, args.output)
    difference = reconstruction.float() - pixels.float()
    mse = difference.square().mean().item()
    report: dict[str, Any] = {
        "input": str(args.image.resolve()),
        "input_size": list(original_size),
        "codec_input_size": [IMAGE_SIZE, IMAGE_SIZE],
        "output": str(args.output.resolve()),
        "codec": str(artifact.directory),
        "weights_sha256": artifact.weights_sha256,
        "device": str(device),
        "latent_shape": list(latent.shape),
        "latent": {
            "raw": tensor_statistics(latent),
            "normalized": tensor_statistics(normalized_latent),
        },
        "reconstruction": {
            "mae": difference.abs().mean().item(),
            "mse": mse,
            "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
            "pixels": tensor_statistics(reconstruction),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
