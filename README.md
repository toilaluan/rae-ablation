# JiT with independent REG and REPA

`jit-reg` trains a class-conditional SiT-B/16 directly on ImageNet-256 pixels. REG
jointly denoises DINOv3 representation tokens; REPA aligns an intermediate SiT
sequence with frozen DINOv3 targets. Either feature can be enabled independently,
and disabling both leaves the baseline model and objective unchanged.

This repository uses configuration schema v3. Older configs, checkpoints, REG
feature caches, and sampling manifests are intentionally incompatible and must be
rebuilt. Independent REPA caches remain reusable.

## Presets

- `configs/baseline-p16.yaml`: feature-free SiT-B/16 baseline.
- `configs/reg-repa-p16.yaml`: pooled 4×4 spatial REG plus block-4,
  full-sequence REPA; equal image and REG time.
- `configs/early-reg-p16.yaml`: pooled 4×4 spatial REG with aligned 2D RoPE, a
  block-4 2048-wide depth-2 head, additive FC–GELU–FC reinjection, and equal
  image and REG time.

The v3 feature configuration lives under `model.features`:

```yaml
model:
  features:
    encoder:
      model_name: vit_base_patch16_dinov3.lvd1689m
      feature_dim: 768
      register_tokens: 4
    reg:
      enabled: true
      pool_size: 4
      target_std: 0.53
      spatial_rope: true
      loss_weight: 0.03
      head:
        kind: standard       # standard | wide_mlp
        hidden_size: 2048
        depth: 2
        mlp_ratio: 4.0
      timestep:
        train_ratio_min: 1.0
        train_ratio_max: 1.0
        sample_ratio: 1.0
      early_prediction:
        block: null          # null predicts after the final block
        projection_hidden_size: 2048
        reinjection_mode: replace  # replace | add
    repa:
      enabled: true
      block: 4
      projection_hidden_size: 2048
      loss_weight: 0.5
      token_scope: image_and_reg  # image | image_and_reg
```

Every transformer block uses the same adaLN weights for both streams while applying
image-time parameters to image tokens and REG-time parameters to REG tokens. REG
heads predict clean x₀. The REG velocity objective is derived from that x₀ using
REG time; the pixel objective uses image time.

## Environment

Python 3.12 is required.

```bash
uv sync --extra dev
uv run hf auth login
uv run wandb login
uv run pytest -q
uv run ruff check .
```

Download the ImageNet-256 ADM reference statistics to the path configured by
`evaluation.reference_npz`:

```bash
mkdir -p data
curl -L \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz \
  -o data/VIRTUAL_imagenet256_labeled.npz
```

## Training

All feature and online-evaluation settings come from YAML. Structured overrides are
the only command-line mechanism for changing them.

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/baseline-p16.yaml \
  --output runs/baseline-p16

uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/reg-repa-p16.yaml \
  --output runs/reg-repa-p16

uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/early-reg-p16.yaml \
  --output runs/early-reg-p16
```

Training logs separate timing for DataLoader wait, host batch preparation, GPU
input preparation, GPU forward/backward, optimizer/EMA, and logging. Validation
and checkpoint durations are logged when those operations run. Because logging
duration is only known after a step is emitted, `throughput/logging_seconds` on a
step describes the preceding step's logging call.

For example, a REPA-only run can be selected without feature-specific CLI flags:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/baseline-p16.yaml \
  --override model.features.repa.enabled=true \
  --output runs/repa-p16
```

## Independent feature caches

The component builder creates only missing active components. Canonical cached REG
tokens have per-token mean zero and standard deviation one; `target_std` is applied
only after online extraction or cache lookup during training. REG and REPA live
under separate fingerprints, so changing a head, loss, optimizer, REPA block, REG
target scale or timing, sampling, evaluation, or the other feature does not
invalidate a reusable component.

Cached training disables random flips:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/cache_features.py \
  --config configs/reg-repa-p16.yaml \
  --override train.horizontal_flip_probability=0

uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/reg-repa-p16.yaml \
  --override feature_cache.enabled=true \
  --override train.horizontal_flip_probability=0 \
  --override data.prefetch_factor=1 \
  --output runs/reg-repa-cached-p16
```

Cached targets are gathered in DataLoader workers and pinned with the image batch.
REPA targets are large, so start with `data.prefetch_factor=1` and increase it only
when host memory and profiling show that deeper prefetching is useful.

Cache manifests record environment versions diagnostically, but only pinned
dataset/preprocessing, encoder identity, component feature schema, and the
component’s target layout participate in its fingerprint.

Cache construction uses two pinned host buffers, a dedicated CUDA copy stream, and
one bounded disk-writer thread per rank. It drains queued writes before
finalization and logs data wait, GPU extraction, buffer backpressure, copy,
write/flush, and total throughput. A capacity preflight runs before allocation.
Full ImageNet needs approximately 29.3 GiB for 4×4 REG, 469.2 GiB for 16×16 REPA,
or 498.5 GiB for both.

## Sampling and evaluation

Standalone sampling inherits `sampling.*` from the checkpoint. Explicit sampling
flags override individual defaults. REG checkpoints jointly integrate distinct
image and REG schedules with Euler or Heun; CFG-window decisions use image time.

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/reg-repa-p16/checkpoints/last

uv run torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/early-reg-p16/checkpoints/last \
  --solver heun \
  --evaluate
```

`--evaluate` computes FID on rank zero using the checkpoint’s `evaluation.*`
settings and writes `evaluation.json` beside `samples.npz`. Output names and
manifests record the relative REG schedule and complete schema-v3 feature
description. Sampling is rank-partitioned and resumable only from a compatible
schema-v3 manifest.
