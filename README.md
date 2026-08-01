# SiT REPA and K-exploration

This repository trains a class-conditional SiT on ImageNet-256 in either RGB
pixel space or normalized FLUX.2 VAE space. It is intentionally limited to
three questions:

1. How does a modern SiT baseline train?
2. What does intermediate DINOv3 REPA alignment add?
3. What happens when every source image explores K independent noise samples
   and trains only on the best candidate?

Configuration schema v11 is a clean break. Old DDT/REG checkpoints are not
loadable. Existing schema-v4 REPA caches and FLUX.2 latent caches remain
reusable.

## Model and objective

The default SiT-B has 12 blocks, width 768, and 12 attention heads. It uses
one fused QKV projection, PyTorch SDPA/Flash Attention, factorized 2D RoPE,
per-head Q/K RMSNorm, affine-free RMSNorm, fused gate/value SwiGLU, and
adaLN-Zero conditioning from the sum of timestep and class embeddings.

The model predicts clean `x0`, not velocity. Training constructs

```text
xt       = t * x0 + (1 - t) * noise
v_target = x0 - noise
v_pred   = (x_prediction - xt) / max(1 - t, 0.05)
loss     = mse(v_pred, v_target)
```

Sampling likewise applies classifier-free guidance to x-predictions and only
then converts them to velocity for Euler or Heun integration.

## Experiment modes

| Preset | REPA | K | Selector |
| --- | ---: | ---: | --- |
| `configs/sit-baseline.yaml` | no | 1 | none |
| `configs/sit-repa.yaml` | yes | 1 | none |
| `configs/sit-k-final.yaml` | no | 4 | final velocity MSE |
| `configs/sit-k-final-repa.yaml` | yes | 4 | final velocity MSE |
| `configs/sit-k-repa.yaml` | yes | 4 | block-4 REPA cosine distance |

For K-exploration, every original sample gets one timestep and K independent
noises. Candidate scoring runs under no-grad. The winner is gathered and run
again through the full model with gradients, so rejected candidates retain no
activation graph and receive no backward pass.

The final selector evaluates all SiT blocks and chooses the smallest velocity
MSE. The early selector evaluates only through `repa.block`, chooses the
smallest token-averaged cosine distance to the clean DINOv3 target, and avoids
running rejected candidates through the remaining blocks.

Candidates are flattened into one large batch for maximum throughput. If that
does not fit, set a divisor of K:

```bash
--override exploration.candidate_chunk_size=2
```

Sweep K without changing the original data batch:

```bash
--override exploration.k=2
--override exploration.k=8
```

`train.global_batch_size` always counts original ImageNet samples, not
candidates.

## Setup and caches

Python 3.12 is required.

```bash
uv sync --extra dev
uv run hf auth login
uv run wandb login
uv run pytest -q
uv run ruff check .
```

REPA uses frozen DINOv3 spatial features resized and L2-normalized to the SiT
token grid. Cache them before throughput-sensitive runs:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/cache_features.py \
  --config configs/sit-k-repa.yaml
```

FLUX.2 produces normalized `128x16x16` diffusion tensors. Build its cache with
the same latent overrides used for training:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/cache_latents.py \
  --config configs/sit-baseline.yaml \
  --override model.input_space=flux2_latents \
  --override model.in_channels=128 \
  --override model.patch_size=2
```

Patch size 1 gives a 16x16 latent token grid; patch size 2 gives an 8x8 grid.
REPA cache identity includes this grid size.

## Training

Pixel baseline:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/sit-baseline.yaml \
  --output runs/sit-pixel
```

Cached latent K-final experiment:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/sit-k-final.yaml \
  --override model.input_space=flux2_latents \
  --override model.in_channels=128 \
  --override model.patch_size=2 \
  --override vae.cache.enabled=true \
  --output runs/sit-latent-k4-final
```

Cached latent early-REPA experiment:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/sit-k-repa.yaml \
  --override model.input_space=flux2_latents \
  --override model.in_channels=128 \
  --override model.patch_size=2 \
  --override vae.cache.enabled=true \
  --override repa.cache.enabled=true \
  --output runs/sit-latent-k4-repa
```

Resume with the identical resolved schema-v11 configuration:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/sit-k-repa.yaml \
  --output runs/sit-k4-repa \
  --resume runs/sit-k4-repa/checkpoints/last
```

Cached latents remove online VAE work. Cached REPA targets remove online DINOv3
work and allow latent training to omit RGB decoding entirely. BF16, static
compiled regions, SDPA, asynchronous host-to-device prefetch, and fused MLP
projections are enabled by default. Gradient checkpointing is off because the
no-grad selector already bounds K-way activation memory.

## Validation and reporting

Download the ADM ImageNet-256 reference statistics:

```bash
mkdir -p data
curl -L \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz \
  -o data/VIRTUAL_imagenet256_labeled.npz
```

Every validation row contains the FID together with:

- `progress/epoch`
- `progress/images_seen`
- `compute/cumulative_model_flops`

The same FID curve can therefore use epochs/images seen or model FLOPs as its
x-axis. FLOPs count SiT, the REPA projector, the K-way selector, and the winner
forward/backward. Frozen online VAE/DINO preprocessing, optimizer work, and
evaluation are excluded; wall-clock and throughput metrics are logged
separately.

Selection diagnostics include candidate-zero score, winning score, score
improvement, mean winning index, original images/s, and selector candidates/s.

Sample or evaluate a checkpoint with:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/sit-k4-repa/checkpoints/last \
  --evaluate

uv run python scripts/eval.py \
  --samples runs/sit-k4-repa/checkpoints/last/samples-*/samples.npz \
  --reference data/VIRTUAL_imagenet256_labeled.npz
```
