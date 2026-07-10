# RAE latent-grid ablation

This project runs a controlled ImageNet-256 diffusion comparison between the pretrained
`1024x16x16` [RAEv2 DINOv3-L K7](https://huggingface.co/toilaluan/raev2-dinov3l-k7)
and its `1024x8x8` [compact adapter](https://huggingface.co/toilaluan/raev2-compressor).
The DiT, initialization, data order, optimizer schedule, global batch, epochs, and immutable
validation plan are shared. Codec weights, per-channel statistics, token grid, and timestep
shift are the intended differences.

V1 is class-conditional. It has no text conditioning, classifier-free guidance, or separately
cached REPA targets. Spatial attention is bidirectional; causality exists only across
transformer depth.

## Environment

The project requires Python 3.12 and locks PyTorch 2.12, Transformers 4.57, Accelerate,
safetensors, torch-fidelity, W&B, Ruff, and pytest.

```bash
uv sync --extra dev
uv run huggingface-cli login
uv run wandb login                       # only needed with --wandb
uv run pytest -q
uv run ruff check .
```

Codec and dataset revisions are immutable in code and recorded in caches and checkpoints:

- RAE: `c1205c99df492a7e1941274a0eee09e4754aee0b`
- compact adapter: `12c4ae3f889adacaeea8bdec00f2c6d0995e5cb2`
- ImageNet repack: `1bd0400450249a7fe90c0aece37d0d03e7ea956a`

Download the [official OpenAI ImageNet-256 ADM reference NPZ](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz)
yourself; the scripts never download this input automatically.

```bash
curl -L \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz \
  -o /data/VIRTUAL_imagenet256_labeled.npz
```

## Cache latents

The dataset already contains 256x256 images. Caching performs RGB conversion, tensor
conversion, and uint8-to-`[0,1]` scaling only.

```bash
uv run rae-cache --config configs/canonical.yaml --codec rae --output /data/rae-cache

uv run torchrun --standalone --nproc-per-node=8 .venv/bin/rae-cache \
  --config configs/canonical.yaml --codec both --output /data/imagenet-rae-both

uv run rae-cache --config configs/canonical.yaml \
  --output /data/imagenet-rae-both --verify-only
```

Raw fp16 tensors are written as atomic, resumable safetensors shards. Each shard stores
latents, labels, and original dataset indices. The manifest pins shapes, preprocessing,
revisions, dtype, sample count, and completed shards. Statistics are computed over the exact
stored fp16 values.

For 1,281,167 training examples, budget approximately 625.6 GiB for RAE, 156.4 GiB for compact,
or 782 GiB for the combined cache, plus filesystem overhead.

## Train

Training, sampling, and evaluation are independent Python scripts. A single-device launch is:

```bash
uv run python scripts/train.py \
  --config configs/canonical.yaml --codec rae \
  --cache-dir /data/imagenet-rae-both --output runs/rae \
  --reference-npz /data/VIRTUAL_imagenet256_labeled.npz \
  --validation-interval 5 --checkpoint-interval 1 --wandb
```

For distributed training, replace `python` with `torchrun --standalone --nproc-per-node=8`:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --codec rae_compact --cache-dir /data/imagenet-rae-both \
  --output runs/compact \
  --reference-npz /data/VIRTUAL_imagenet256_labeled.npz --wandb
```

Structured config changes use repeatable `--override KEY=VALUE` arguments. Sampling algorithm
settings are script flags, not `validation.*` config fields.

```bash
uv run python scripts/train.py \
  --codec rae --cache-dir /data/imagenet-rae-both --output runs/rae-no-depth \
  --reference-npz /data/VIRTUAL_imagenet256_labeled.npz \
  --override model.depth_attention.enabled=false \
  --override model.gradient_checkpointing=false \
  --override compile.enabled=false \
  --override train.optimizer.kind=muon \
  --eval-batch-size 32 --eval-sampling-steps 50 \
  --eval-guidance-scale 1.78 --eval-guidance-start 0.1
```

Regional compilation happens before DDP. The raw model remains the optimizer, EMA, and
serialization authority; compiled wrappers are execution views only. Training uses BF16
autocast, fp32 flow/loss math, global batch 1024, clipping at 1.0, and EMA 0.9995.

Rank zero prints initialization, resume, epoch, validation, FID, checkpoint, and shutdown
events. Optimizer metrics print every `--log-interval` updates (default 10); W&B still receives
every optimizer update when enabled. Other ranks remain silent.

Resume and optionally force validation with:

```bash
uv run python scripts/train.py \
  --codec rae --cache-dir /data/imagenet-rae-both --output runs/rae \
  --reference-npz /data/VIRTUAL_imagenet256_labeled.npz \
  --resume runs/rae/checkpoints/last --validate-now --wandb
```

## Objective and sampling

For a normalized clean latent `x`:

```text
u = sigmoid(N(0, 1))
alpha = sqrt(C*H*W / 4096)               # 8 for RAE, 4 for compact
t = alpha*u / (1 + (alpha-1)*u)
x_t = (1-t)*x + t*epsilon
v = epsilon - x
v_hat = (x_t - x_hat) / max(t, 0.05)
loss = MSE(v_hat_full, v) + MSE(v_hat_early, v)
```

The default sampler integrates the shifted schedule with 50 Euler steps. Internal guidance is
`x_early + 1.78*(x_full-x_early)` for `t>=0.1`; below that point the full head is used. Stable
sample IDs and per-sample noise seeds make generation invariant to batch size, rank count, and
previous RNG use.

Create the default balanced 50,000-image archive (50 images per ImageNet class):

```bash
uv run python scripts/sample.py \
  --checkpoint runs/rae/checkpoints/last/ema-export.safetensors \
  --plan runs/rae/validation-plan.pt --output-dir generated/rae \
  --codec rae --cache-dir /data/imagenet-rae-both
```

Use `uv run torchrun --standalone --nproc-per-node=8 scripts/sample.py ...` to distribute sample
IDs across GPUs on a shared filesystem. The output directory contains:

```text
generated/rae/
├── samples.npz    # uint8 NHWC images in arr_0; int64 labels in arr_1
├── samples.json   # checkpoint/plan hashes, class counts, parameters, completion
└── preview.png    # labeled 4x4 grid of arr_0[0:16]
```

Sampling writes through a resumable NPY memmap and completion bitmap. Ranks write disjoint
positions, then rank zero atomically packages an uncompressed ZIP64 NPZ and removes temporary
storage. An incompatible checkpoint, plan, sample count, class layout, codec, or sampling
setting is rejected on resume. A 50,000-image archive is about 9.2 GiB and needs roughly twice
that temporary peak disk space during final packaging.

## Evaluate and compare

Evaluation streams `arr_0.npy` from stored or compressed NPZ containers, extracts
TensorFlow-compatible Inception-v3 pool features with torch-fidelity, and reports FID only.
The adjacent `samples.json` supplies the immutable plan hash.

```bash
uv run python scripts/eval.py \
  --samples generated/rae/samples.npz \
  --reference /data/VIRTUAL_imagenet256_labeled.npz \
  --output runs/rae/evaluation.json

uv run rae-compare runs/rae/evaluation.json runs/compact/evaluation.json \
  --output runs/comparison.json
```

For an external NPZ without an adjacent `samples.json`, pass `--plan` explicitly. Comparison
refuses mismatched validation-plan hashes.

## Validation and checkpoints

The first training launch creates one immutable validation plan. Its 5,000-example periodic
prefix has five examples per class; the final 50,000-example plan has 50 per class. Periodic and
final EMA validation produce the same `samples.npz`, `samples.json`, `preview.png`, and
`evaluation.json` layout beneath `runs/<name>/validation/epoch-...`.

`--validation-interval N` and `--checkpoint-interval N` count completed epochs. Zero disables
periodic work, while completion validation and checkpointing still run. Periodic validation
reports `FID-5K`; completion reports `FID-50K`. Checkpoints store raw model and EMA safetensors,
optimizer/scheduler state, RNG streams, epoch/step, cache fingerprint, validation-plan hash,
last validated epoch, W&B run ID, and resolved config. Atomic `last` and `best-fid` pointers are
maintained, and every retained checkpoint has a directly loadable EMA export.

## Tiny smoke workflow

This exercises cache resume and two optimizer steps with a smaller DiT. Final validation still
needs one balanced sample per class and the official reference statistics.

```bash
uv run rae-cache --codec both --output /tmp/rae-smoke --max-samples 32 \
  cache.batch_size=16 cache.shard_size=32 cache.workers=0

uv run python scripts/train.py \
  --codec rae --cache-dir /tmp/rae-smoke --output /tmp/rae-smoke-run --max-steps 2 \
  --reference-npz /data/VIRTUAL_imagenet256_labeled.npz \
  --override model.hidden_size=128 --override model.depth=2 \
  --override model.num_heads=4 --override model.mlp_size=256 \
  --override model.early_exit_block=1 --override compile.enabled=false \
  --override cache.workers=0 --override train.global_batch_size=8 \
  --override train.micro_batch_size=4 --override validation.interval=0 \
  --override validation.periodic_samples=1000 --override validation.final_samples=1000 \
  --eval-sampling-steps 2
```

The codec reconstruction integration test remains opt-in:

```bash
RAE_RUN_CODEC_INTEGRATION=1 uv run pytest -q \
  tests/test_codec_reconstruction_integration.py
```

Set `RAE_CODEC_DEVICE=cpu` or `RAE_CODEC_DEVICE=cuda` to choose its device.
