from __future__ import annotations

import json
import math
import time
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from torch import Tensor, nn

LogFunction = Callable[[str], None]


def _read_npy_header(handle: BinaryIO) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        raise ValueError(f"unsupported NPY format version in sample archive: {version}")
    return tuple(shape), bool(fortran_order), np.dtype(dtype)


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            raise ValueError("NPY payload ended before its declared shape")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class NpzImageReader:
    """Validate and stream an NHWC uint8 arr_0 from a stored or compressed NPZ."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with zipfile.ZipFile(self.path) as archive:
            try:
                info = archive.getinfo("arr_0.npy")
            except KeyError as error:
                raise ValueError("sample NPZ does not contain arr_0") from error
            with archive.open(info) as handle:
                shape, fortran_order, dtype = _read_npy_header(handle)
        if fortran_order:
            raise ValueError("sample arr_0 must use C order")
        if (
            len(shape) != 4
            or shape[-1] != 3
            or shape[1] < 1
            or shape[1] != shape[2]
        ):
            raise ValueError(f"sample arr_0 must have shape NxSxSx3, received {shape}")
        if dtype != np.dtype(np.uint8):
            raise ValueError(f"sample arr_0 must use uint8, received {dtype}")
        if shape[0] < 2:
            raise ValueError("FID requires at least two generated samples")
        expected_payload = math.prod(shape) * dtype.itemsize
        with zipfile.ZipFile(self.path) as archive:
            info = archive.getinfo("arr_0.npy")
            with archive.open(info) as handle:
                _read_npy_header(handle)
                payload_start = handle.tell()
            if info.file_size - payload_start != expected_payload:
                raise ValueError("sample arr_0 payload size does not match its NPY header")
        self.shape = shape
        self.dtype = dtype
        self.compressed = info.compress_type != zipfile.ZIP_STORED

    def batches(self, batch_size: int) -> Iterator[np.ndarray]:
        if batch_size < 1:
            raise ValueError("evaluation batch size must be positive")
        row_shape = self.shape[1:]
        row_bytes = math.prod(row_shape) * self.dtype.itemsize
        with zipfile.ZipFile(self.path) as archive, archive.open("arr_0.npy") as handle:
            shape, fortran_order, dtype = _read_npy_header(handle)
            if shape != self.shape or fortran_order or dtype != self.dtype:
                raise ValueError("sample arr_0 changed after archive validation")
            for begin in range(0, self.shape[0], batch_size):
                count = min(batch_size, self.shape[0] - begin)
                payload = _read_exact(handle, count * row_bytes)
                yield np.frombuffer(payload, dtype=self.dtype).reshape((count, *row_shape))
            if handle.read(1):
                raise ValueError("sample arr_0 has trailing payload bytes")


def validate_reference_statistics(
    path: str | Path, expected_features: int | None = 2048
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if "mu" not in archive or "sigma" not in archive:
            raise ValueError("reference NPZ must contain mu and sigma")
        mu = np.asarray(archive["mu"])
        sigma = np.asarray(archive["sigma"])
    if mu.ndim != 1 or sigma.shape != (mu.size, mu.size):
        raise ValueError("reference mu and sigma have incompatible shapes")
    if expected_features is not None and mu.size != expected_features:
        raise ValueError(
            f"reference statistics must have {expected_features} features, received {mu.size}"
        )
    if not np.issubdtype(mu.dtype, np.number) or not np.issubdtype(sigma.dtype, np.number):
        raise ValueError("reference mu and sigma must be numeric")
    mu = mu.astype(np.float64, copy=False)
    sigma = sigma.astype(np.float64, copy=False)
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        raise ValueError("reference mu and sigma must be finite")
    if not np.allclose(sigma, sigma.T, rtol=1e-5, atol=1e-8):
        raise ValueError("reference sigma must be symmetric")
    return {"mu": mu, "sigma": sigma}


def fid_from_statistics(
    generated: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> float:
    from torch_fidelity.metric_fid import fid_statistics_to_metric

    left = {
        "mu": np.asarray(generated["mu"], dtype=np.float64),
        "sigma": np.asarray(generated["sigma"], dtype=np.float64),
    }
    right = {
        "mu": np.asarray(reference["mu"], dtype=np.float64),
        "sigma": np.asarray(reference["sigma"], dtype=np.float64),
    }
    if left["mu"].shape != right["mu"].shape:
        raise ValueError("generated and reference feature dimensions differ")
    metric = fid_statistics_to_metric(left, right, verbose=False)
    return float(metric["frechet_inception_distance"])


def _default_feature_extractor(device: torch.device) -> nn.Module:
    from torch_fidelity.utils import create_feature_extractor

    return create_feature_extractor(
        "inception-v3-compat",
        ["2048", "logits_unbiased"],
        cuda=device.type == "cuda",
        verbose=False,
    )


@torch.inference_mode()
def extract_feature_statistics(
    reader: NpzImageReader,
    *,
    batch_size: int,
    device: torch.device,
    log_interval: int = 10,
    log: LogFunction = print,
    feature_extractor: nn.Module | None = None,
) -> dict[str, np.ndarray]:
    if log_interval < 1:
        raise ValueError("evaluation log interval must be positive")
    extractor = feature_extractor or _default_feature_extractor(device)
    extractor = extractor.to(device).eval()
    feature_sum: Tensor | None = None
    feature_outer: Tensor | None = None
    probability_batches: list[Tensor] = []
    count = 0
    started = time.perf_counter()
    total_batches = math.ceil(reader.shape[0] / batch_size)
    for batch_index, images in enumerate(reader.batches(batch_size), start=1):
        writable = np.array(images, copy=True)
        tensor = torch.from_numpy(writable).permute(0, 3, 1, 2).to(device)
        output = extractor(tensor)
        features = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(output, (tuple, list)) and len(output) > 1:
            probability_batches.append(output[1].detach().float().softmax(dim=-1).cpu())
        features = features.detach().to(device="cpu", dtype=torch.float64)
        if features.ndim != 2:
            raise ValueError("Inception feature extractor did not return a matrix")
        if feature_sum is None:
            feature_sum = torch.zeros(features.shape[1], dtype=torch.float64)
            feature_outer = torch.zeros(
                (features.shape[1], features.shape[1]), dtype=torch.float64
            )
        assert feature_outer is not None
        feature_sum.add_(features.sum(dim=0))
        feature_outer.addmm_(features.T, features)
        count += features.shape[0]
        if batch_index % log_interval == 0 or batch_index == total_batches:
            elapsed = max(time.perf_counter() - started, 1e-9)
            log(
                f"feature extraction: {count}/{reader.shape[0]} images, "
                f"{count / elapsed:.2f} images/s"
            )
    if feature_sum is None or feature_outer is None or count < 2:
        raise ValueError("not enough generated features for FID")
    mu = feature_sum / count
    sigma = (feature_outer - count * torch.outer(mu, mu)) / (count - 1)
    result = {"mu": mu.numpy(), "sigma": sigma.numpy()}
    if probability_batches:
        result["probabilities"] = torch.cat(probability_batches).numpy()
    return result


def inception_score(probabilities: np.ndarray, splits: int = 10) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < splits or splits < 1:
        raise ValueError("Inception probabilities must be [N,C] with at least one row per split")
    epsilon = np.finfo(np.float64).eps
    scores = []
    for subset in np.array_split(values, splits):
        marginal = subset.mean(axis=0, keepdims=True)
        divergence = subset * (
            np.log(subset.clip(min=epsilon)) - np.log(marginal.clip(min=epsilon))
        )
        scores.append(float(np.exp(divergence.sum(axis=1).mean())))
    return float(np.mean(scores))


def evaluate_npz(
    samples: str | Path,
    reference: str | Path,
    *,
    batch_size: int = 64,
    device: str | torch.device = "auto",
    log_interval: int = 10,
    log: LogFunction = print,
    feature_extractor: nn.Module | None = None,
) -> dict[str, float]:
    resolved_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(device) == "auto"
        else torch.device(device)
    )
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation was requested but CUDA is unavailable")
    reader = NpzImageReader(samples)
    compression = "compressed" if reader.compressed else "stored"
    log(f"sample archive valid: shape={reader.shape}, dtype=uint8, compression={compression}")
    expected_features = None if feature_extractor is not None else 2048
    reference_stats = validate_reference_statistics(reference, expected_features)
    log(
        f"reference statistics valid: features={reference_stats['mu'].size}, "
        f"device={resolved_device}"
    )
    generated_stats = extract_feature_statistics(
        reader,
        batch_size=batch_size,
        device=resolved_device,
        log_interval=log_interval,
        log=log,
        feature_extractor=feature_extractor,
    )
    fid = fid_from_statistics(generated_stats, reference_stats)
    log(f"FID: {fid:.6f}")
    metrics = {"FID": fid}
    if "probabilities" in generated_stats:
        score = inception_score(generated_stats["probabilities"])
        metrics["IS"] = score
        log(f"IS: {score:.6f}")
    return metrics


def save_evaluation(
    path: str | Path,
    metrics: dict[str, Any],
    plan_hash: str,
    *,
    num_samples: int | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    values: dict[str, Any] = {"sampling_plan_hash": plan_hash, "metrics": metrics}
    if num_samples is not None:
        values["num_samples"] = num_samples
    temporary.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
