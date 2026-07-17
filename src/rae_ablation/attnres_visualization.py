from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .model.attnres import AttnResCapture


def _capture_arrays(
    capture: AttnResCapture, sampling_steps: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if sampling_steps < 1 or not capture.records:
        raise ValueError("AttnRes capture must contain at least one sampling step")
    if len(capture.records) % sampling_steps:
        raise ValueError("AttnRes record count does not divide into sampling steps")
    sites_per_step = len(capture.records) // sampling_steps
    first = capture.records[:sites_per_step]
    site_labels = np.asarray([record.site for record in first])
    max_sources = max(len(record.source_labels) for record in first)
    source_labels = np.full((sites_per_step, max_sources), "", dtype="<U64")
    valid_sources = np.zeros((sites_per_step, max_sources), dtype=np.bool_)
    spatial = np.full((sampling_steps, sites_per_step, max_sources), np.nan, dtype=np.float32)
    time = np.full_like(spatial, np.nan)
    classes = np.full_like(spatial, np.nan)

    for step in range(sampling_steps):
        records = capture.records[step * sites_per_step : (step + 1) * sites_per_step]
        if tuple(record.site for record in records) != tuple(site_labels.tolist()):
            raise ValueError("AttnRes routing sites changed across sampling steps")
        for site, record in enumerate(records):
            count = len(record.source_labels)
            if step == 0:
                source_labels[site, :count] = record.source_labels
                valid_sources[site, :count] = True
            elif tuple(source_labels[site, :count]) != record.source_labels:
                raise ValueError("AttnRes residual sources changed across sampling steps")
            spatial[step, site, :count] = record.spatial.detach().float().cpu().numpy()
            time[step, site, :count] = record.time.detach().float().cpu().numpy()
            classes[step, site, :count] = record.classes.detach().float().cpu().numpy()
    return (
        spatial,
        time,
        classes,
        site_labels,
        source_labels,
        valid_sources,
    )


def _colors(values: np.ndarray) -> np.ndarray:
    clipped = np.nan_to_num(values, nan=-1.0).clip(-1, 1)
    valid = clipped >= 0
    clipped = clipped.clip(0, 1)
    red = (255 * clipped).astype(np.uint8)
    green = (255 * np.sqrt(clipped)).astype(np.uint8)
    blue = (255 * (1 - clipped)).astype(np.uint8)
    rgb = np.stack((red, green, blue), axis=-1)
    rgb[~valid] = np.asarray([235, 235, 235], dtype=np.uint8)
    return rgb


def _save_heatmaps(
    spatial: np.ndarray,
    site_labels: np.ndarray,
    valid_sources: np.ndarray,
    target: Path,
) -> None:
    steps, sites, _max_sources = spatial.shape
    columns = min(4, sites)
    rows = math.ceil(sites / columns)
    panel_width = 220
    panel_height = 24 + max(steps * 4, 80) + 22
    canvas = Image.new("RGB", (columns * panel_width, rows * panel_height), "white")
    draw = ImageDraw.Draw(canvas)
    for site in range(sites):
        row, column = divmod(site, columns)
        x0 = column * panel_width
        y0 = row * panel_height
        count = int(valid_sources[site].sum())
        values = spatial[:, site, :count]
        heatmap = Image.fromarray(_colors(values), mode="RGB").resize(
            (max(count * 16, 32), max(steps * 4, 80)),
            resample=Image.Resampling.NEAREST,
        )
        draw.text((x0 + 4, y0 + 4), str(site_labels[site]), fill="black")
        image_x = x0 + 28
        image_y = y0 + 22
        canvas.paste(heatmap, (image_x, image_y))
        draw.text((x0 + 4, image_y), "0", fill="black")
        draw.text((x0 + 2, image_y + heatmap.height - 10), str(steps - 1), fill="black")
        draw.text(
            (image_x, image_y + heatmap.height + 4),
            f"sources 0..{count - 1}",
            fill="black",
        )
    temporary = target.with_suffix(target.suffix + f".partial-{os.getpid()}")
    canvas.save(temporary, format="PNG")
    os.replace(temporary, target)


def save_attnres_visualization(
    *,
    capture: AttnResCapture,
    sampling_timesteps: np.ndarray,
    labels: np.ndarray,
    plan_hash: str,
    block_size: int,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    (
        spatial,
        time,
        classes,
        site_labels,
        source_labels,
        valid_sources,
    ) = _capture_arrays(capture, len(sampling_timesteps))
    output_dir = Path(output_dir)
    npz_path = output_dir / "attnres-weights.npz"
    png_path = output_dir / "attnres-weights.png"
    temporary = npz_path.with_suffix(npz_path.suffix + f".partial-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            spatial_weights=spatial,
            time_token_weights=time,
            class_token_weights=classes,
            sampling_timesteps=np.asarray(sampling_timesteps, dtype=np.float32),
            site_labels=site_labels,
            source_labels=source_labels,
            valid_sources=valid_sources,
            visualization_labels=np.asarray(labels, dtype=np.int64),
            validation_plan_hash=np.asarray(plan_hash),
            attnres_block_size=np.asarray(block_size, dtype=np.int64),
        )
    os.replace(temporary, npz_path)
    _save_heatmaps(spatial, site_labels, valid_sources, png_path)
    return npz_path, png_path


__all__ = ["save_attnres_visualization"]
