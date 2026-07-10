# RAE latent-grid ablation

This package runs a controlled ImageNet-256 diffusion comparison between the pretrained
`1024×16×16` [RAEv2 DINOv3-L K7](https://huggingface.co/toilaluan/raev2-dinov3l-k7)
and its `1024×8×8` [compact adapter](https://huggingface.co/toilaluan/raev2-compressor).
The DiT dimensions, initialization, class/data order, optimizer schedule, global batch,
epochs, and immutable validation plan are shared. Codec weights, per-channel statistics,
token grid, and the resulting timestep shift are the only intended differences.

V1 is class-conditional. It deliberately has no text conditioning, classifier-free guidance,
or separately cached REPA targets. Spatial attention is bidirectional; causality exists only
across transformer depth.

## Environment and authentication

The project requires Python 3.12 and locks PyTorch 2.12, Transformers 4.57, Accelerate,
safetensors, torch-fidelity/torchmetrics, W&B, Ruff, and pytest.

```bash
uv sync --extra dev
uv run huggingface-cli login
uv run wandb login                       # only needed with --wandb
uv run pytest -q
uv run ruff check .
```

Codec and dataset revisions are immutable in code and recorded in every cache/checkpoint:

- RAE: `c1205c99df492a7e1941274a0eee09e4754aee0b`
- compact adapter: `12c4ae3f889adacaeea8bdec00f2c6d0995e5cb2`
- ImageNet repack: `1bd0400450249a7fe90c0aece37d0d03e7ea956a`

The dataset already contains 256×256 images. Caching performs only `RGB` conversion, tensor
conversion, and uint8-to-`[0,1]` scaling. There is no resize, crop, flip, color transform, or
training augmentation.

## Cache latents

One process can cache either codec:

```bash
uv run rae-cache --config configs/canonical.yaml --codec rae --output /data/rae-cache
uv run rae-cache --config configs/canonical.yaml --codec rae_compact --output /data/compact-cache
```

The recommended path caches both at once and reuses the already computed RAE feature when
calling the compact adapter:

```bash
torchrun --standalone --nproc-per-node=8 \
  .venv/bin/rae-cache --config configs/canonical.yaml \
  --codec both --output /data/imagenet-rae-both

uv run rae-cache --config configs/canonical.yaml \
  --output /data/imagenet-rae-both --verify-only
```

Raw fp16 tensors are written as atomic, resumable 256-sample safetensors shards. Each shard has
latents, labels, and original dataset indices. The manifest pins shapes, preprocessing,
revisions, dtype, sample count, and completed shards. Statistics are population statistics over
the exact stored fp16 values and have shape `[1024,1,1]` for each codec.

For 1,281,167 training examples, budget approximately 625.6 GiB for RAE, 156.4 GiB for compact,
or 782 GiB for the combined cache, plus filesystem overhead. The command performs a disk
preflight, removes interrupted `.partial` files, skips complete shards, bounds its asynchronous
D2H/write queue, and prints images/sec and queue occupancy.

## Launch matched training

The canonical model has 16 blocks, width 1024, 16 heads, a 4096-wide SwiGLU, patch size 1, and
an early x-prediction head after block 8. It uses RMSNorm, q/k normalization, a learnable 16×16
position table (bicubically interpolated at 8×8), dynamic 2D RoPE, AdaLN-Zero, and parameter-free
Depth-Attention. Gradient checkpointing and Accelerate regional compilation are on by default.

```bash
torchrun --standalone --nproc-per-node=8 .venv/bin/rae-train \
  --config configs/canonical.yaml --codec rae \
  --cache-dir /data/imagenet-rae-both --output runs/rae \
  --validation-interval 5 --checkpoint-interval 1 --wandb

torchrun --standalone --nproc-per-node=8 .venv/bin/rae-train \
  --config configs/canonical.yaml --codec rae_compact \
  --cache-dir /data/imagenet-rae-both --output runs/compact \
  --validation-interval 5 --checkpoint-interval 1 --wandb
```

Any structured field can be overridden with an OmegaConf dot-list argument:

```bash
uv run rae-train --codec rae --cache-dir /data/imagenet-rae-both \
  model.depth_attention.enabled=false \
  model.gradient_checkpointing=false compile.enabled=false \
  train.optimizer.kind=muon
```

Regional compilation happens before DDP. The raw model remains the optimizer, EMA, and
serialization authority; compiled wrappers are execution views only. Sampling constructs and
compiles a separate EMA view, which can reuse the repeated-block compiler cache. Checkpoints
reject `_orig_mod` keys.

The canonical optimizer schedule is AdamW at `2e-4` through epoch 25, linear decay to `2e-5` at
epoch 50, then a hold through epoch 80. Optional Muon applies only to eligible 2D transformer
matrices; AdamW retains embeddings, norms, biases, convolution/projection matrices, and heads.
Training uses BF16 autocast, fp32 flow/loss math, global batch 1024, clipping at 1.0, and EMA
0.9995.

## Objective and sampling

Cached latents are normalized on GPU using their own cache statistics. For a clean latent `x`:

```text
u = sigmoid(N(0, 1))
alpha = sqrt(C*H*W / 4096)               # 8 for RAE, 4 for compact
t = alpha*u / (1 + (alpha-1)*u)
x_t = (1-t)*x + t*epsilon
v = epsilon - x
v_hat = (x_t - x_hat) / max(t, 0.05)
loss = MSE(v_hat_full, v) + MSE(v_hat_early, v)
```

Sampling integrates the same shifted schedule with 50 Euler steps. Internal guidance is applied
in x-space as `x_early + 1.78*(x_full-x_early)` for `t>=0.1`; below that point the full head is
used. Shuffle, initialization, timestep, noise, and validation generators are isolated, so the
latent shape cannot change class order or sampled timesteps.

## Validation, W&B, and checkpoints

The first launch creates one immutable validation plan. Its 5,000-example periodic prefix has
five examples per ImageNet class; the 50,000-example final plan has 50 per class. Stable sample
IDs and per-sample noise seeds make generation invariant to batch size, rank count, and previous
RNG use. Two explicit fp16 visualization tensors cover both token grids. The plan hash is stored
in checkpoints and evaluation summaries.

`--validation-interval N` and `--checkpoint-interval N` count completed epochs. Zero disables
periodic work, but the completion validation/checkpoint still runs. Periodic validation reports
`FID-5K`; completion reports `FID-50K` and Inception Score. Validation uses EMA weights and a
persistent unaugmented ImageNet validation-pixel cache, synchronizes ranks, and restores training
without touching its RNG. A resumed interval boundary does not repeat validation unless
`--validate-now` is supplied.

Checkpoints are atomic `epoch-XXXX` directories containing raw model and EMA safetensors,
optimizer state (including Muon+AdamW), scheduler epoch, RNG streams, epoch/global step, cache
fingerprint, validation-plan hash, last validated epoch, W&B run ID, and resolved config. `last`
and `best-fid` pointers are maintained, with `keep_last_n` retention. Every retained checkpoint
also has a directly loadable EMA export.

W&B initializes only on global rank zero and resumes by stored run ID. It logs velocity losses,
LR, gradient norm, EMA, image/token throughput, timing, peak memory, epoch/step, validation
metrics, sampling NFE, provenance, and resolved config. Large checkpoint artifacts remain off
unless `wandb.checkpoint_artifacts=true` is explicitly set.

Resume and force validation with:

```bash
torchrun --standalone --nproc-per-node=8 .venv/bin/rae-train \
  --codec rae --cache-dir /data/imagenet-rae-both --output runs/rae \
  --resume runs/rae/checkpoints/last --validate-now --wandb
```

## Sample, evaluate, and compare

```bash
uv run rae-sample runs/rae/checkpoints/last/ema-export.safetensors \
  --codec rae --output samples/rae.png --classes 0 1 2 3

# Generate the FID set. `--fid-samples` defaults to 50,000, giving 50 images/class.
torchrun --standalone --nproc-per-node=8 .venv/bin/rae-sample \
  runs/rae/checkpoints/last/ema-export.safetensors \
  --codec rae --cache-dir /data/imagenet-rae-both \
  --output generated/rae --fid-samples \
  --plan runs/rae/validation-plan.pt

uv run rae-eval generated/rae validation/real-imagenet-validation \
  --plan runs/rae/validation-plan.pt --output runs/rae/evaluation.json

uv run rae-compare runs/rae/evaluation.json runs/compact/evaluation.json \
  --output runs/comparison.json
```

Comparison refuses mismatched validation-plan hashes.

FID sampling uses the labels, sample IDs, and independent noise seeds from the immutable plan.
It writes one numbered PNG per sample plus `.sampling.json`, refuses to mix checkpoints or plans
in one directory, and safely resumes an interrupted run. The sample count must be divisible by
the configured number of classes.

## Tiny smoke workflow

This exercises atomic cache/resume and two optimizer steps. It uses a smaller DiT and a shortened,
class-balanced final validation (one sample/class, two Euler steps), while leaving latent channel
dimensions unchanged:

```bash
uv run rae-cache --codec both --output /tmp/rae-smoke --max-samples 32 \
  cache.batch_size=16 cache.shard_size=32 cache.workers=0

uv run rae-train --codec rae --cache-dir /tmp/rae-smoke \
  --output /tmp/rae-smoke-run --max-steps 2 \
  model.hidden_size=128 model.depth=2 model.num_heads=4 model.mlp_size=256 \
  model.early_exit_block=1 compile.enabled=false cache.workers=0 \
  train.global_batch_size=8 train.micro_batch_size=4 \
  validation.interval=0 validation.periodic_samples=1000 \
  validation.final_samples=1000 validation.sampling_steps=2
```

CUDA integration tests that download the actual codecs or exercise pinned asynchronous transfer
are intentionally optional; the normal test suite is CPU-only and uses tiny real regional
compilation.

To reconstruct the included Angus image with both pinned codecs and write the labeled comparison
to `assets/angus_decoded.jpeg`, run:

```bash
RAE_RUN_CODEC_INTEGRATION=1 uv run pytest -q \
  tests/test_codec_reconstruction_integration.py
```

Set `RAE_CODEC_DEVICE=cpu` or `RAE_CODEC_DEVICE=cuda` to select the integration-test device.
