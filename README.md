# JiT + REG

`jit-reg` trains a class-conditional SiT directly on ImageNet-256 RGB pixels with the JiT
clean-image objective. The canonical baseline is SiT-B/32: 256×256 images are projected into an
8×8 patch grid, processed by a 12-block adaLN-Zero transformer, and predicted directly back into
pixels. Patch size is configurable whenever it divides the image size exactly.

Optional standard REG adds a jointly diffused DINOv3 feature state and optional block-4 REPA
alignment.
The default `model.reg.token_mode=cls` state contains one CLS token. The opt-in `cls_spatial` mode
also adaptively pools the normalized 16×16 patch map to a configurable grid (4×4 by default),
applies parameter-free LayerNorm to each pooled token, and gives those tokens learned absolute
positions inside SiT. The frozen `vit_base_patch16_dinov3.lvd1689m` target is loaded only for REG
training and its four register tokens are discarded. REPA aligns only SiT image-patch tokens with
DINO patch features resized to the JiT grid. Sampling a REG checkpoint does not load DINOv3.

The opt-in wide-intermediate REG ablation predicts the complete clean REG state x₀ after block 4
with two conditioned, 2048-wide SwiGLU residual layers. By default, the prediction is projected
back to the SiT hidden size and replaces the hidden REG prefix before blocks 5–12. Two additional
ablations can project predicted CLS x₀ into downstream conditioning and project the final wide CLS
latent into a configurable number of continuation REG tokens. A third optional branch uses
predicted x₀ or the wide CLS latent to gate a fresh projection of the raw noisy 32×32×3 patches
back into the image-token stream. Standard REG remains the default.

Set `model.reg.cls_head_block=-1` to move the wide REG predictor after the final SiT transformer
block. In this terminal mode the predictor is auxiliary: it predicts REG x₀ for the REG objective
but does not feed the image prediction. Clean REG reinjection, continuation tokens, x₀
conditioning, and patch reprojection are not allocated or applied, even if their configuration
values are present.

## Environment

Python 3.12 is required.

```bash
uv sync --extra dev
uv run hf auth login
uv run wandb login       # only for --wandb runs
uv run pytest -q
uv run ruff check .
```

The ImageNet repository and revision are pinned in `configs/canonical.yaml`. Training reads it
directly through Hugging Face Datasets; there is no feature-cache preparation step.

Download the ImageNet-256 ADM reference statistics once:

```bash
mkdir -p data
curl -L \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz \
  -o data/VIRTUAL_imagenet256_labeled.npz
```

## Train

Baseline:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/canonical.yaml \
  --output runs/baseline-p32 \
  --reference-npz data/VIRTUAL_imagenet256_labeled.npz \
  --wandb
```

REG + REPA:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/canonical.yaml \
  --reg \
  --output runs/reg-p32 \
  --reference-npz data/VIRTUAL_imagenet256_labeled.npz \
  --wandb
```

REG without REPA:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/canonical.yaml \
  --reg --no-repa \
  --output runs/reg-no-repa-p32 \
  --reference-npz data/VIRTUAL_imagenet256_labeled.npz \
  --wandb
```

REG with CLS plus a pooled 4×4 spatial feature map:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/canonical.yaml \
  --reg \
  --override model.reg.token_mode=cls_spatial \
  --override model.reg.pool_size=4 \
  --output runs/reg-spatial-p32 \
  --reference-npz data/VIRTUAL_imagenet256_labeled.npz \
  --wandb
```

Wide-intermediate REG + REPA (matched ablation):

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/canonical.yaml \
  --reg-widehead \
  --output runs/reg-widehead-p32 \
  --reference-npz data/VIRTUAL_imagenet256_labeled.npz \
  --wandb
```

Enable the two independent wide-head extensions with structured overrides:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/canonical.yaml \
  --reg-widehead --no-repa \
  --override model.reg.x0_conditioning=true \
  --override model.reg.continuation_tokens=4 \
  --output runs/reg-wide-extended-p32 \
  --reference-npz data/VIRTUAL_imagenet256_labeled.npz \
  --wandb
```

Enable conditional raw-patch reprojection with its default 2048-dimensional interaction:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/canonical.yaml \
  --reg-widehead --no-repa \
  --override model.reg.patch_reprojection.enabled=true \
  --override model.reg.patch_reprojection.source=x0_cls \
  --output runs/reg-patch-reprojection-p32 \
  --reference-npz data/VIRTUAL_imagenet256_labeled.npz \
  --wandb
```

At the intermediate REG head, this branch patchifies the original noisy image again, projects each
raw patch into `model.reg.patch_reprojection.intermediate_size`, gates it with either predicted CLS
x₀ or the final wide CLS latent (plus timestep conditioning by default), projects it to the SiT
hidden size, and adds it to the existing image tokens with a learned residual scale initialized to
`1e-3`. Set `source=wide_cls` to test the less constrained latent source.

Change patch size with a structured override, for example
`--override model.patch_size=16`. The canonical recipe is 80 epochs, global batch 1024, Muon,
learning rate `2e-4` for 25 epochs followed by a 25-epoch linear decay to `2e-5`, EMA `0.9995`,
10% classifier-free label dropout, and logit-normal image timesteps with location `-0.8` and
scale `0.8`.

REG and image states use the same sampled timestep for interpolation, velocity losses, transformer
conditioning, validation, and sampling.

Images are randomly flipped on the training device through a checkpointed RNG stream. The same
flipped `[0,1]` image is normalized for DINOv3 in REG runs and scaled to `[-1,1]` for JiT.

## Cache REG targets

REPA-disabled runs can read precomputed REG targets and avoid loading DINOv3 during training. The
cache stores BF16 CLS tokens for `token_mode=cls`, or CLS plus only the configured pooled tokens
for `token_mode=cls_spatial`. Native 16×16 patch features and REPA targets are never saved.

Cached runs deliberately disable horizontal flipping, so use the same zero flip probability when
building and training:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/cache_reg.py \
  --config configs/canonical.yaml \
  --override model.reg.token_mode=cls \
  --override train.horizontal_flip_probability=0

uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/canonical.yaml \
  --reg --no-repa \
  --override reg_cache.enabled=true \
  --override train.horizontal_flip_probability=0 \
  --output runs/reg-cached-p32 \
  --reference-npz data/VIRTUAL_imagenet256_labeled.npz \
  --wandb
```

Use matching `model.reg.token_mode` and `model.reg.pool_size` overrides in both commands. Caches
live under `data/reg-tokens/<fingerprint>/`, are resumable under `torchrun`, and are rejected when
their dataset, DINO model, token configuration, environment, shape, or completion state differs.
REPA-enabled runs continue to extract DINO features online because REPA requires spatial targets.

Resume standard REG with the same model overrides and REG variant:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/canonical.yaml \
  --reg \
  --output runs/reg-p32 \
  --resume runs/reg-p32/checkpoints/last \
  --reference-npz data/VIRTUAL_imagenet256_labeled.npz \
  --validate-now --wandb
```

Resume the wide-intermediate variant with `--reg-widehead`:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/canonical.yaml \
  --reg-widehead \
  --output runs/reg-widehead-p32 \
  --resume runs/reg-widehead-p32/checkpoints/last \
  --reference-npz data/VIRTUAL_imagenet256_labeled.npz \
  --validate-now --wandb
```

Checkpoints contain raw model and EMA weights, optimizer state, all explicit RNG streams, the
pinned dataset/preprocessing and REG-cache fingerprints, resolved configuration, validation plan,
lightweight metadata, and W&B identity. DINOv3 is outside model/EMA state. `last` and `best-fid`
are resumable pointers.

## Objective and sampling

Time runs from noise to data:

```text
x_t = t*x + (1-t)*noise
v_target = (x - x_t) / max(1-t, 0.05)
v_pred   = (x_pred - x_t) / max(1-t, 0.05)
```

The baseline minimizes image velocity MSE. REG adds `0.03 × REG-state velocity MSE` and
`0.5 × negative cosine REPA loss` over resized DINO patch targets. Both REG variants log
unweighted `train/reg_x0_mse` and `train/reg_x0_cosine`; these diagnostics do not change the
objective weights.

Sampling uses 50-step Euler integration and batched conditional/null-label CFG. Scale `1` selects
the conditional prediction. CFG defaults to the full trajectory and can be restricted with
`--cfg-start` and `--cfg-end`. REG sampling deterministically initializes and jointly integrates
the image and configured REG state with the same schedule. `--timestep-shift` controls that shared
schedule.

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/baseline-p32/checkpoints/last

uv run torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/reg-p32/checkpoints/last

uv run torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --timestep-shift 0.125 \
  --checkpoint-dir runs/reg-widehead-p32/checkpoints/last
```

Generation is rank-partitioned and resumable. It writes an ADM-compatible `samples.npz`, a
preview, and a manifest beneath the resolved epoch checkpoint. The child directory name records
sample count, Euler steps, CFG scale, timestep shift, and CFG interval, for example
`samples-euler-n50000-steps50-cfg1p78-shift1p0-window0p0-1p0`. Its manifest also records model and
REG settings, checkpoint hash, validation-plan hash, and sampler parameters. Resuming rejects
incompatible settings, and checkpoint retention protects epochs containing sampling outputs.

## Matched wide-head experiment

Use the standard and wide-intermediate commands above from scratch with the same data revision,
seed, optimizer, loss weights, validation plan, and sampler settings. The canonical 80-epoch run
performs periodic FID-5K and final FID-50K. Compare FID convergence and final FID alongside image
and REG velocity losses, REG x₀ MSE/cosine, REPA loss, gradient norm, parameter count, throughput,
and peak memory. Treat the ablation as successful only when matched final FID improves with stable
training; otherwise keep standard REG as the default and record the wide head as a negative result.

## Validate, evaluate, and compare

Periodic and final validation are configured under `validation.*` and use the EMA model plus the
same immutable labels, sample IDs, and noise seeds as standalone sampling. To evaluate an archive:

```bash
uv run python scripts/eval.py \
  --reference data/VIRTUAL_imagenet256_labeled.npz \
  --output runs/baseline-p32/evaluation.json \
  --samples runs/baseline-p32/checkpoints/last/samples-euler-n50000-steps50-cfg1p78-shift1p0-window0p0-1p0/samples.npz

uv run python scripts/eval.py \
  --samples runs/reg-p32/checkpoints/last/samples-euler-n50000-steps50-cfg1p78-shift1p0-window0p0-1p0/samples.npz \
  --reference data/VIRTUAL_imagenet256_labeled.npz \
  --output runs/reg-p32/evaluation.json

uv run python scripts/compare.py \
  runs/baseline-p32/evaluation.json \
  runs/reg-p32/evaluation.json \
  --output runs/comparison.json
```

Comparison is model-agnostic and requires matching validation-plan hashes.

## Smoke workflow

This checks the direct data path and a tiny baseline without compilation:

```bash
uv run python scripts/train.py \
  --config configs/canonical.yaml \
  --output /tmp/jit-reg-smoke \
  --reference-npz data/VIRTUAL_imagenet256_labeled.npz \
  --max-steps 2 \
  --override data.max_samples=32 \
  --override data.workers=0 \
  --override model.hidden_size=64 \
  --override model.depth=2 \
  --override model.num_heads=4 \
  --override compile.enabled=false \
  --override train.global_batch_size=8 \
  --override train.micro_batch_size=4 \
  --override validation.interval=0 \
  --override validation.periodic_samples=1000 \
  --override validation.final_samples=1000 \
  --eval-sampling-steps 2
```

Add `--reg --override model.reg.repa_block=2` to smoke-test standard REG with the two-block model.
For wide-intermediate REG, add `--reg-widehead`, set `model.reg.repa_block=1`, and set
`model.reg.cls_head_block=1`; an intermediate head must run before the final block. Use
`model.reg.cls_head_block=-1` for the auxiliary terminal-head mode.
