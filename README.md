# RAE latent-grid ablation

This project compares ImageNet-256 diffusion training with the pretrained
`1024x16x16` [RAEv2 DINOv3-L K7](https://huggingface.co/toilaluan/raev2-dinov3l-k7)
and its `1024x8x8` [compact adapter](https://huggingface.co/toilaluan/raev2-compressor).
Both codecs use the same DDT-IG architecture, training recipe, validation plan, and sampling
implementation.

V1 is class-conditional. It has no text conditioning or classifier-free guidance. The full and
early DDT predictions provide internal guidance at every sampling step.

## Environment

The project requires Python 3.12.

```bash
uv sync --extra dev
uv run huggingface-cli login
uv run wandb login                    # only needed when training with --wandb
uv run pytest -q
uv run ruff check .
```

Codec and dataset revisions are pinned in code and recorded in caches and checkpoints.

Download the official OpenAI ImageNet-256 ADM reference archive once:

```bash
mkdir -p ./data
curl -L \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz \
  -o ./data/VIRTUAL_imagenet256_labeled.npz
```

## Choose a codec

Define the shared paths once. Switch between full and compact latents by changing only `CODEC`.
All later commands derive their run and sample directories from it.

```bash
CONFIG=./configs/canonical.yaml
CACHE_DIR=./data/imagenet-rae-both
REFERENCE_NPZ=./data/VIRTUAL_imagenet256_labeled.npz

CODEC=rae                           # full 16x16 latent grid
# CODEC=rae_compact                 # compact 8x8 latent grid
```

To switch later, run either `CODEC=rae` or `CODEC=rae_compact` before the next command.

## Cache latents

Create one shared cache containing both codecs so experiments can switch with `CODEC` alone:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/cache.py \
  --config "$CONFIG" --codec both --output "$CACHE_DIR"
```

Verify an existing cache without loading the dataset:

```bash
uv run python scripts/cache.py \
  --config "$CONFIG" --output "$CACHE_DIR" --verify-only
```

The cache contains resumable fp16 safetensors shards, labels, source indices, a pinned manifest,
and per-codec statistics. For 1,281,167 examples, budget approximately 625.6 GiB for RAE,
156.4 GiB for compact, or 782 GiB for both.

## Train

Train the selected codec:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config "$CONFIG" \
  --codec "$CODEC" \
  --cache-dir "$CACHE_DIR" \
  --output "./runs/${CODEC}" \
  --reference-npz "$REFERENCE_NPZ" \
  --wandb
```

Enable heterogeneous patch forcing by adding `--use-patch-forcing`:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config "$CONFIG" \
  --codec "$CODEC" \
  --cache-dir "$CACHE_DIR" \
  --output "./runs/${CODEC}-patch-forcing" \
  --reference-npz "$REFERENCE_NPZ" \
  --use-patch-forcing \
  --wandb
```

The logit-normal timestep distribution defaults to location 0 and scale 1. Change it through
structured overrides when needed:

```bash
--override train.timestep_loc=-0.25 --override train.timestep_scale=1.5
```

Resume training and optionally validate immediately:

```bash
uv run python scripts/train.py \
  --config "$CONFIG" \
  --codec "$CODEC" \
  --cache-dir "$CACHE_DIR" \
  --output "./runs/${CODEC}" \
  --reference-npz "$REFERENCE_NPZ" \
  --resume "./runs/${CODEC}/checkpoints/last" \
  --validate-now --wandb
```

Other configuration changes use repeatable `--override KEY=VALUE` arguments. DDT dimensions are
flat `model.*` fields such as `model.encoder_depth`, `model.encoder_hidden_size`, and
`model.base_model_depth`.

## Objective and patch forcing

Time runs from noise to data:

```text
t = 0: pure noise
t = 1: clean data
x_t = t*x + (1-t)*epsilon
target velocity = x - epsilon
v_hat = (x_hat - x_t) / max(1-t, 0.05)
loss = MSE(v_hat_full, target) + MSE(v_hat_early, target)
```

Uniform training samples one timestep per example:

```text
t_max = sigmoid(N(loc, scale^2))
t_i = t_max for every spatial token
```

With patch forcing, each spatial token receives a potentially noisier timestep while the DDT is
conditioned on `t_max`:

```text
t_i = t_max - abs(N(0, (t_max/2)^2))
if t_i < 0: t_i = U(0, t_max)
```

Patch timesteps are shared across latent channels. Lower `t_i` means more noise.

## Sample

Generate the balanced 50,000-image archive for the selected codec:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --config "$CONFIG" \
  --codec "$CODEC" \
  --cache-dir "$CACHE_DIR" \
  --checkpoint "./runs/${CODEC}/checkpoints/last/ema-export.safetensors" \
  --plan "./runs/${CODEC}/validation-plan.pt" \
  --output-dir "./generated/${CODEC}" \
  --sampling-steps 50 \
  --guidance-scale 1.78 \
  --timestep-shift 1.0
```

Sampling integrates from `t=0` noise to `t=1` data. Internal guidance is always
`early + guidance_scale*(full-early)`; use scale `1.0` to select the full prediction. The positive
timestep shift controls the rational integration schedule, and `1.0` gives a linear schedule.

The output contains `samples.npz`, `samples.json`, and `preview.png`. Generation is resumable and
rejects incompatible checkpoints, plans, codecs, sample counts, guidance scales, or timestep
shifts.

## Evaluate and compare

Evaluate the selected codec:

```bash
uv run python scripts/eval.py \
  --samples "./generated/${CODEC}/samples.npz" \
  --reference "$REFERENCE_NPZ" \
  --output "./runs/${CODEC}/evaluation.json"
```

After evaluating both codec values, compare them:

```bash
uv run python scripts/compare.py \
  ./runs/rae/evaluation.json \
  ./runs/rae_compact/evaluation.json \
  --output ./runs/comparison.json
```

Evaluation streams stored or compressed NPZ inputs and reports FID. Comparison requires matching
validation-plan hashes.

## Validation and checkpoints

The first training launch creates one immutable balanced validation plan. Periodic and final EMA
validation use the same artifact layout as standalone sampling. Checkpoints include raw model and
EMA weights, optimizer state, RNG streams, cache and validation fingerprints, resolved config,
and W&B run identity. The `last` and `best-fid` pointers remain directly resumable.

The online sampler is configured with training-script flags. For example, use
`--eval-timestep-shift 2.0` to validate with a shifted schedule.

## Tiny smoke workflow

```bash
SMOKE_CACHE=/tmp/rae-smoke-cache
SMOKE_RUN=/tmp/rae-smoke-${CODEC}

uv run python scripts/cache.py \
  --codec both --output "$SMOKE_CACHE" --max-samples 32 \
  cache.batch_size=16 cache.shard_size=32 cache.workers=0

uv run python scripts/train.py \
  --config "$CONFIG" \
  --codec "$CODEC" \
  --cache-dir "$SMOKE_CACHE" \
  --output "$SMOKE_RUN" \
  --reference-npz "$REFERENCE_NPZ" \
  --max-steps 2 \
  --override model.encoder_hidden_size=128 \
  --override model.decoder_hidden_size=128 \
  --override model.encoder_depth=2 \
  --override model.decoder_depth=1 \
  --override model.encoder_num_heads=4 \
  --override model.decoder_num_heads=4 \
  --override model.base_model_depth=1 \
  --override compile.enabled=false \
  --override cache.workers=0 \
  --override train.global_batch_size=8 \
  --override train.micro_batch_size=4 \
  --override validation.interval=0 \
  --override validation.periodic_samples=1000 \
  --override validation.final_samples=1000 \
  --eval-sampling-steps 2
```

The codec reconstruction integration test remains opt-in:

```bash
RAE_RUN_CODEC_INTEGRATION=1 uv run pytest -q \
  tests/test_codec_reconstruction_integration.py
```
