# Progressive DINO-to-Pixel LightningDiT

This repository contains one class-conditional ImageNet-128 experiment and its matched
pixel baseline. The progressive run trains on four normalized 8x8 token targets:

1. DINOv3-B/16 layer 12 for 20 epochs.
2. DINOv3-B/16 layer 8 for 20 epochs.
3. DINOv3-B/16 layer 4 for 20 epochs.
4. Flattened 3x16x16 RGB patches for 20 epochs.

All representations have shape `[B,64,768]`. DINO layers are one-indexed; CLS and four
register tokens are removed, and DINO's final norm is independently applied to every
selected intermediate output. DINO and RGB targets use full-training-set, per-channel
mean and standard deviation. Images are resized to 128x128 without augmentation so
cached features, RGB patches, alignments, and baseline batches remain paired.

## Model and objective

The shared denoiser is a local token-native integration of
[LightningDiT-B/1](https://github.com/hustvl/LightningDiT/blob/main/models/lightningdit.py):
12 blocks, width 768, 12 heads, RMSNorm, QK norm, two-dimensional RoPE, SwiGLU, AdaLN,
SDPA, and class conditioning. It predicts clean `x0` tokens and converts them to velocity
for a shifted-logit-normal linear Gaussian-flow loss. The velocity denominator is floored
at 0.05. The optional LightningDiT cosine term is zero by default.

Every stage uses a pointwise token detail head: projected noisy tokens and final
transformer context are summed, passed through two width-768 RMSNorm/SiLU residual MLPs
with expansion 2, and mapped through a zero-initialized output projection. At the RGB
stage, a zero-initialized DiP-style patch U-Net additionally processes each 3x16x16 patch
independently. Its resolution path is `16→8→4→2→1→2→4→8→16`, its down channels are
`3→64→128→256→512`, and the transformer context is concatenated at the 1x1 bottleneck.

Consecutive targets are initialized with full-rank 768x768 orthogonal Procrustes maps.
This borrows the alignment motivation from
[AsymFlow](https://arxiv.org/abs/2605.12964), but it is deliberately **not** AsymFlow's
low-rank asymmetric parameterization: these matrices are full-rank initialization
alignments. Cumulative projectors stay outside the nonlinear model. At export, the final
projector is folded into the linear input, output, and token-detail weights; the RGB
convolutional head is never rotated.

## Preparation

The single preparation workflow extracts layers 12, 8, and 4 in one DINO pass into
separate resumable BF16 memmaps, computes DINO/RGB statistics, and fits deterministic
50,000-image Procrustes maps with held-out diagnostics. A complete ImageNet run needs
about 352 GiB for the three DINO caches. Capacity and runtime directories are never
deleted automatically.

```bash
torchrun --standalone --nproc-per-node=8 scripts/prepare_representations.py \
  --config configs/experiment.yaml
```

The versioned manifest records cache completion, statistics, dataset and encoder
fingerprints, held-out MSE, orthogonality errors, matrix keys, and the explicit full-rank
parameterization description. Corrupt, incomplete, or mismatched artifacts are rejected.

## Training

```bash
torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/experiment.yaml \
  --output runs/progressive-seed0
```

Stage 1 trains the whole transformer, linear I/O, and token head; the RGB patch head is
not introduced yet. At later stages, I/O/detail heads and blocks `0,1,10,11` train fully.
Blocks `2…9` receive fresh rank-16, alpha-16, dropout-0.05 LoRA on attention and SwiGLU
linears. Base parameters and conditioning in those blocks stay frozen. Before each
transition, LoRA is merged into both model and EMA; fresh zero-output adapters are then
attached, the loader and compiled execution view are rebuilt, and AdamW/cosine state is
reset with one warmup epoch. Boundary checkpoints are written before and after the
idempotent transition.

The matched baseline uses exactly the exported RGB architecture, target normalization,
data order, RNG streams, objective, sampling, and evaluation settings, but initializes it
from scratch and trains every parameter for all 80 epochs:

```bash
torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/experiment.yaml \
  --override experiment.mode=pixel_baseline \
  --output runs/pixel-baseline-seed0
```

Follow-up identity-alignment, direct layer-12-to-RGB, and full-finetune settings are
available under `alignment` and `curriculum` overrides. Source consistency remains
available but defaults to zero so it does not confound the primary experiment.

## Sampling and evaluation

Sampling accepts only a completed RGB checkpoint. It runs one flow trajectory in
normalized RGB-token space, applies interval CFG, reverses dataset normalization,
unpatchifies, and clamps images to `[0,1]`.

```bash
python scripts/sample.py \
  --checkpoint-dir runs/progressive-seed0/checkpoints/last \
  --num-samples 50000 \
  --evaluate
```

Intermediate stages are evaluated with validation loss and feature PCA previews rather
than image FID. Final comparisons record FID-50k, IS, wall-clock time, images seen,
estimated FLOPs, peak memory, and total/trainable parameter counts.

## Verification

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check .
```

Configuration and checkpoints use clean schema version 3. Old DDT/Mage artifacts are
rejected with an explicit error and are not migrated or deleted.
