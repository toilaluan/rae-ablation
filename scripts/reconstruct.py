from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch import Tensor

from jit_reg.config import IMAGE_SIZE, PixelCodecConfig, load_config
from jit_reg.pixel_codec import PixelCodec, PixelTokenStatistics, load_pixel_codec_artifact


@dataclass(frozen=True)
class LoadedCodec:
    model: PixelCodec
    statistics: PixelTokenStatistics | None
    directory: Path
    weights_sha256: str
    source_kind: str


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
        "--config",
        type=Path,
        default=Path("configs/pixel-codec.yaml"),
        help="model config used when --codec is a training checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config override used when loading a training checkpoint",
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
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if value == "auto"
        else torch.device(value)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _weights_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_training_checkpoint(
    source: Path,
    device: torch.device,
    config: PixelCodecConfig,
) -> LoadedCodec:
    weights = source if source.is_file() else source / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(
            f"codec path is neither a portable artifact nor a training checkpoint: {source.resolve()}"
        )
    state = load_file(str(weights), device=str(device))
    prefix = "_orig_mod."
    if state and all(name.startswith(prefix) for name in state):
        state = {name.removeprefix(prefix): value for name, value in state.items()}
    model = PixelCodec(config).to(device)
    model.load_state_dict(state)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return LoadedCodec(
        model=model,
        statistics=None,
        directory=source.resolve().parent if source.is_file() else source.resolve(),
        weights_sha256=_weights_digest(weights),
        source_kind="training_checkpoint",
    )


def load_codec(
    source: Path,
    device: torch.device,
    config: PixelCodecConfig,
) -> LoadedCodec:
    if source.is_dir() and (source / "metadata.json").exists():
        artifact = load_pixel_codec_artifact(source, device)
        return LoadedCodec(
            model=artifact.model,
            statistics=artifact.statistics,
            directory=artifact.directory,
            weights_sha256=artifact.weights_sha256,
            source_kind="portable_artifact",
        )
    return _load_training_checkpoint(source, device, config)


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
    config = load_config(args.config, args.override)
    codec = load_codec(args.codec, device, config.pixel_codec)
    pixels, original_size = load_image(args.image, device)

    with torch.inference_mode():
        latent = codec.model.encode(pixels)
        normalized_latent = (
            codec.statistics.normalize(latent) if codec.statistics is not None else None
        )
        reconstruction = codec.model.decode(latent)

    save_image(reconstruction, args.output)
    difference = reconstruction.float() - pixels.float()
    mse = difference.square().mean().item()
    report: dict[str, Any] = {
        "input": str(args.image.resolve()),
        "input_size": list(original_size),
        "codec_input_size": [IMAGE_SIZE, IMAGE_SIZE],
        "output": str(args.output.resolve()),
        "codec": str(codec.directory),
        "codec_source": codec.source_kind,
        "weights_sha256": codec.weights_sha256,
        "device": str(device),
        "latent_shape": list(latent.shape),
        "latent": {
            "raw": tensor_statistics(latent),
            "normalized": (
                tensor_statistics(normalized_latent)
                if normalized_latent is not None
                else None
            ),
            "normalization_note": (
                None
                if normalized_latent is not None
                else "unavailable: training checkpoints do not contain dataset statistics"
            ),
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
