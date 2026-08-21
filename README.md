# DINOv3 semantic-waypoint diffusion

This repository trains an ImageNet-128 diffusion transformer in frozen DINOv3-B/16
feature space. It uses one schema-v15 configuration, `configs/dino-waypoint.yaml`.

The frozen encoder extracts two independently normalized 768x8x8 endpoints in one traversal:

- target `T`: block-2 patch tokens, decoded to RGB by the trained pixel decoder;
- semantic: final block-12 patch tokens, also used unpooled by REPA;
- waypoint `S`: block-12 tokens area-pooled to 4x4 and bilinearly restored to 8x8.

## Piecewise flow

With waypoint time `tau=0.5` and full-resolution Gaussian noise `epsilon`:

```text
M = (1 - tau) * S + tau * epsilon
epsilon -> M -> T
```

Training samples semantic and target legs equally. A shifted logit-normal local time
`u` is mapped to the active global interval:

```text
semantic: t = tau + (1 - tau) * u
          xt = (1 - t) * S + t * epsilon
          clean prediction target = S

target:   t = tau * u
          xt = (1 - t/tau) * T + (t/tau) * M
          clean prediction target = T
```

The shallow base head exits after encoder block 4; the full head uses the complete
DDT. The full head is supervised on both trajectory legs, while the base-head loss is
applied only to semantic-leg examples. Predictions are converted to velocity with
`(xt - prediction) / max(t, 0.05)`. REPA is disabled for the waypoint preset. Setting
`repa.enabled=true` enables the baseline REPA projector and aligns encoder block 8
with the original unpooled block-12 features.

Sampling applies internal guidance in clean-prediction space only on the semantic leg.
The target leg bypasses the base head and applies ordinary CFG to the full prediction.
The default trajectory uses 10 semantic steps from `1 -> 0.5` and 40 target steps from
`0.5 -> 0`. Euler therefore uses 50 NFE; Heun uses 99 NFE.

## Setup and latent statistics

```bash
uv sync --extra dev
torchrun --standalone --nproc-per-node=8 scripts/generate_latent_statistics.py \
  --config configs/dino-waypoint.yaml
```

Statistics contain separate per-channel mean/std vectors for block-2 target tokens and
block-12 semantic tokens. Shared-normalization statistics and decoders trained against
them are incompatible; regenerate the statistics and retrain the pixel decoder.

Train a decoder if one is not already available:

```bash
torchrun --standalone --nproc-per-node=8 scripts/train_decoder.py \
  --config configs/dino-waypoint.yaml \
  --output runs/pixel-decoder
```

## Optional paired endpoint cache

The cache stores normalized BF16 target and original unpooled semantic tensors. A
full ImageNet cache requires about 235 GiB before filesystem overhead. Build it after
the latent statistics:

```bash
torchrun --standalone --nproc-per-node=8 scripts/cache_endpoints.py \
  --config configs/dino-waypoint.yaml
```

Enable cache-backed training with `vision.cache.enabled=true`. Training then loads
labels and stable indices only; it does not read RGB or instantiate DINO. Missing,
incomplete, or fingerprint-mismatched caches are rejected rather than falling back
online.

## Diffusion training and sampling

```bash
torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/dino-waypoint.yaml \
  --override decoder.checkpoint=runs/pixel-decoder/checkpoints/last \
  --override vision.cache.enabled=true \
  --output runs/semantic-waypoint-seed0

torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/semantic-waypoint-seed0/checkpoints/last \
  --semantic-steps 10 \
  --target-steps 40 \
  --evaluate
```

The single YAML explicitly lists every supported option, including online/cache
selection, pooling, waypoint timing, dual-head loss, IG, CFG, compilation, validation,
evaluation, checkpointing, and W&B settings.

## Verification

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check .
```
