from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from rae_ablation.codecs import load_codecs

RUN_CODEC_INTEGRATION = os.environ.get("RAE_RUN_CODEC_INTEGRATION") == "1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_IMAGE = PROJECT_ROOT / "assets" / "angus.jpeg"
COMPARISON_IMAGE = PROJECT_ROOT / "assets" / "angus_decoded.jpeg"


def _image_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    resized = image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    storage = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
    return storage.reshape(256, 256, 3).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255


def _tensor_image(tensor: torch.Tensor) -> Image.Image:
    pixels = tensor.detach().squeeze(0).float().clamp(0, 1).mul(255).round().byte().cpu()
    return Image.fromarray(pixels.permute(1, 2, 0).numpy(), mode="RGB")


def _save_comparison(images: list[Image.Image], labels: list[str], path: Path) -> None:
    caption_height = 28
    panel_width, panel_height = images[0].size
    canvas = Image.new(
        "RGB", (panel_width * len(images), panel_height + caption_height), color="white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        left = index * panel_width
        canvas.paste(image, (left, caption_height))
        text_box = draw.textbbox((0, 0), label)
        text_width = text_box[2] - text_box[0]
        draw.text((left + (panel_width - text_width) / 2, 7), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=95, subsampling=0)


@pytest.mark.integration
@pytest.mark.skipif(
    not RUN_CODEC_INTEGRATION,
    reason="set RAE_RUN_CODEC_INTEGRATION=1 to download and run the pinned codecs",
)
def test_rae_and_compact_reconstruct_angus() -> None:
    """Encode once with RAE, decode both latent grids, and save a labeled comparison."""
    assert SOURCE_IMAGE.is_file()
    requested_device = os.environ.get(
        "RAE_CODEC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(requested_device)
    source = Image.open(SOURCE_IMAGE).convert("RGB")
    pixels = _image_tensor(source, device)
    rae, compact = load_codecs("both", str(device))

    with (
        torch.inference_mode(),
        torch.autocast(
            device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        rae_latent = rae.encode(pixels)
        compact_latent = compact.compress(rae_latent)
        rae_reconstruction = rae.decode(rae_latent)
        compact_reconstruction = compact.decode(compact_latent)

    assert rae_latent.shape == (1, 1024, 16, 16)
    assert compact_latent.shape == (1, 1024, 8, 8)
    assert rae_reconstruction.shape == compact_reconstruction.shape == (1, 3, 256, 256)
    _save_comparison(
        [
            _tensor_image(pixels),
            _tensor_image(rae_reconstruction),
            _tensor_image(compact_reconstruction),
        ],
        ["Original", "RAE 16x16", "Compact RAE 8x8"],
        COMPARISON_IMAGE,
    )
    assert COMPARISON_IMAGE.is_file()
