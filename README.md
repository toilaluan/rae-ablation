# DINOv3 clean semantic-cascade diffusion

This repository trains an ImageNet-128 diffusion transformer in frozen DINOv3-B/16
feature space. It uses one schema-v16 configuration, `configs/dino-cascade.yaml`.

The frozen encoder extracts two independently normalized 768x8x8 endpoints in one traversal:

- target `T`: block-2 patch tokens, decoded to RGB by the trained pixel decoder;
- semantic: final block-12 patch tokens, also used unpooled by REPA;
- semantic prior `S`: block-12 tokens area-pooled to 4x4 and bilinearly restored to 8x8.

## Clean semantic cascade

The two stages use independent local timesteps in `[0,1]`:

```text
epsilon -> S -> T
```

Training samples semantic and target legs independently per example with equal
probability. Both stages use the same shifted logit-normal time `u`:

```text
semantic: xt = (1 - u) * S + u * epsilon
          clean prediction target = S

target:   xt = (1 - u) * T + u * S
          clean prediction target = T
```

The shallow base head exits after encoder block 4; the full head uses the complete
DDT. The full head is supervised on both trajectory legs, while the base-head loss is
applied only to semantic-leg examples. Predictions are converted to velocity with
`(xt - prediction) / max(u, 0.05)`. REPA is disabled for the cascade preset. Setting
`repa.enabled=true` enables the baseline REPA projector and aligns encoder block 8
with the original unpooled block-12 features.

Sampling first completes the semantic leg from `u=1 -> 0`, producing clean `S` with
internal guidance. It then resets local time to `u=1` and runs `S -> T` with ordinary
CFG on the full head. The default allocation is 10 semantic and 40 target steps.
Euler uses 50 NFE; Heun uses 98 because endpoint corrections at `u=0` are skipped.

## Setup and latent statistics

```bash
uv sync --extra dev
torchrun --standalone --nproc-per-node=8 scripts/generate_latent_statistics.py \
  --config configs/dino-cascade.yaml
```

Statistics contain separate per-channel mean/std vectors for block-2 target tokens and
block-12 semantic tokens. Shared-normalization statistics and decoders trained against
them are incompatible; regenerate the statistics and retrain the pixel decoder.

Train a decoder if one is not already available:

```bash
torchrun --standalone --nproc-per-node=8 scripts/train_decoder.py \
  --config configs/dino-cascade.yaml \
  --output runs/pixel-decoder
```

## Optional paired endpoint cache

The cache stores normalized BF16 target and original unpooled semantic tensors. A
full ImageNet cache requires about 235 GiB before filesystem overhead. Build it after
the latent statistics:

```bash
torchrun --standalone --nproc-per-node=8 scripts/cache_endpoints.py \
  --config configs/dino-cascade.yaml
```

Enable cache-backed training with `vision.cache.enabled=true`. Training then loads
labels and stable indices only; it does not read RGB or instantiate DINO. Missing,
incomplete, or fingerprint-mismatched caches are rejected rather than falling back
online.

## Diffusion training and sampling

```bash
torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/dino-cascade.yaml \
  --override decoder.checkpoint=runs/pixel-decoder/checkpoints/last \
  --override vision.cache.enabled=true \
  --output runs/semantic-cascade-seed0

torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/semantic-cascade-seed0/checkpoints/last \
  --semantic-steps 10 \
  --target-steps 40 \
  --evaluate
```

The single YAML explicitly lists every supported option, including online/cache
selection, semantic pooling, dual-head loss, IG, CFG, compilation, validation,
evaluation, checkpointing, and W&B settings.

## Verification

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check .
```
