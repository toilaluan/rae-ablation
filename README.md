# Joint Semantic-Flow DDT

This repository trains one class-conditional ImageNet-128 model for 200 epochs. The
flow trunk denoises normalized DINOv3-B/16 layer-12 tokens, while a jointly trained
pixel branch reconstructs raw RGB from selected flow hidden states.

The 12-block flow encoder has a zero-initialized RAEv2 base prediction at block 8 and
a two-block AdaLN semantic decoder for the full prediction. Both heads predict clean
semantic tokens and receive independent flow-matching velocity losses. Sampling uses
single-pass internal guidance over raw timesteps in `[0.10, 1.0]`:

```text
guided_x0 = base_x0 + 1.78 * (full_x0 - base_x0)
```

There is no classifier-free guidance evaluation, unconditional batch, endpoint
mixing, direct pixel flow, or phased training.

## Pixel decoder

Post-block encoder features `[4, 8, 12]` and semantic-decoder features `[1, 2]` are
concatenated. Independent width-2048 paths produce transformer-refined coarse RGB
patches and per-patch context. The coarse `3x16x16` patches are decoded independently
by the DiP four-pool U-Net topology and folded into an absolute `[B,3,128,128]` RGB
prediction. Pixel L1 and frozen VGG LPIPS losses update the pixel path and flow trunk
without detachment.

## Preparation and training

Fresh preparation writes only the selected DINO BF16 cache and its channel
statistics. Manifest schema v3 remains reusable when an existing compatible manifest
also contains unused RGB stages or statistics.

```bash
torchrun --standalone --nproc-per-node=8 scripts/prepare_representations.py \
  --config configs/experiment.yaml

torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/experiment.yaml \
  --output runs/joint-seed0
```

The per-sample objective is:

```text
full_velocity_mse + base_velocity_mse + pixel_l1 + pixel_lpips
```

All weights default to `1.0`. RGB L1 uses `[0,1]`; LPIPS receives prediction and
target mapped to `[-1,1]`.

## Sampling

Sampling integrates only the internally guided semantic flow. It then performs one
conditional, un-guided `t=0` forward with pixel decoding enabled. RGB is clamped only
while writing previews and sample archives.

```bash
torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/joint-seed0/checkpoints/last \
  --num-samples 50000 \
  --evaluate
```

Configuration schema v8 and checkpoint/training-state schema v6 intentionally reject
all phase-era artifacts.

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check .
```
