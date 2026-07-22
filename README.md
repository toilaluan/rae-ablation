# JiT + REG

`jit-reg` trains a class-conditional SiT directly on ImageNet-256 RGB pixels with the JiT
clean-image objective. The canonical baseline is SiT-B/32: 256×256 images are projected into an
8×8 patch grid, processed by a 12-block adaLN-Zero transformer, and predicted directly back into
pixels. Patch size is configurable whenever it divides the image size exactly.

Optional standard REG adds a jointly diffused DINOv3 feature state and block-4 REPA alignment.
The default `model.reg.token_mode=cls` state contains one CLS token. The opt-in `cls_spatial` mode
also adaptively pools the normalized 16×16 patch map to a configurable grid (4×4 by default),
applies parameter-free LayerNorm to each pooled token, and gives those tokens learned absolute
positions inside SiT. The frozen `vit_base_patch16_dinov3.lvd1689m` target is loaded only for REG
training and its four register tokens are discarded. REPA aligns only SiT image-patch tokens with
DINO patch features resized to the JiT grid. Sampling a REG checkpoint does not load DINOv3.

The opt-in wide-intermediate REG ablation predicts the complete clean REG state x₀ after block 4
with two conditioned, 2048-wide SwiGLU residual layers. The prediction is independently projected
back to the SiT hidden size and replaces the hidden REG prefix before blocks 5–12; spatial
positions are reapplied on reinjection. Standard REG remains the default REG architecture.

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

Change patch size with a structured override, for example
`--override model.patch_size=16`. The canonical recipe is 80 epochs, global batch 1024, Muon,
learning rate `2e-4` for 25 epochs followed by a 25-epoch linear decay to `2e-5`, EMA `0.9995`,
10% classifier-free label dropout, and logit-normal timesteps with location 0 and scale 1.

Images are randomly flipped on the training device through a checkpointed RNG stream. The same
flipped `[0,1]` image is normalized for DINOv3 in REG runs and scaled to `[-1,1]` for JiT.

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
pinned dataset/preprocessing fingerprint, validation-plan hash, resolved configuration, and W&B
identity. DINOv3 is outside model/EMA state. `last` and `best-fid` are resumable pointers.

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
the image and complete configured REG state. Sampling `cls_spatial` checkpoints requires the same
`model.reg.token_mode` and `model.reg.pool_size` overrides used for training.

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --config configs/canonical.yaml \
  --checkpoint runs/baseline-p32/checkpoints/last/ema-export.safetensors \
  --plan runs/baseline-p32/validation-plan.pt \
  --output-dir generated/baseline-p32

uv run torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --config configs/canonical.yaml \
  --reg \
  --checkpoint runs/reg-p32/checkpoints/last/ema-export.safetensors \
  --plan runs/reg-p32/validation-plan.pt \
  --output-dir generated/reg-p32

uv run torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --config configs/canonical.yaml \
  --reg-widehead \
  --checkpoint runs/reg-widehead-p32/checkpoints/last/ema-export.safetensors \
  --plan runs/reg-widehead-p32/validation-plan.pt \
  --output-dir generated/reg-widehead-p32
```

Generation is rank-partitioned and resumable. It writes an ADM-compatible `samples.npz`, a
preview, and a manifest containing image size, patch size, JiT grid, model variant, REG status,
predictor variant, token mode, pool size, checkpoint hash, validation-plan hash, and sampler
parameters. Resuming into an existing sampling directory rejects incompatible REG settings.

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
  --samples generated/baseline-p32/samples.npz \
  --reference data/VIRTUAL_imagenet256_labeled.npz \
  --output runs/baseline-p32/evaluation.json

uv run python scripts/eval.py \
  --samples generated/reg-p32/samples.npz \
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
`model.reg.cls_head_block=1`; the intermediate head must run before the final block.
