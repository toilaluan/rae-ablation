# Semantic-to-Pixel DDT

This repository compares two class-conditional ImageNet-128 training modes with the
same dual-head DDT architecture:

- `progressive`: train a normalized DINO semantic endpoint, then normalized RGB pixels.
- `pixel_only`: train normalized RGB pixels directly.

Every endpoint is an `8x8` grid of 768-dimensional tokens. The semantic endpoint is a
configurable DINOv3-B/16 layer (layer 12 by default); RGB endpoints are flattened
`3x16x16` patches.

## Training distribution

During the semantic phase, every item loads both DINO and RGB. By default, 90% are pure
semantic (`pixel_ratio=0`) and 10% use a ratio sampled from
`sigmoid(N(0, 1))`. The endpoints are normalized independently and then interpolated.
Pure-semantic items use normalized RGB as an auxiliary pixel target. The pixel phase
and `pixel_only` mode load only RGB and always use `pixel_ratio=1`.

The item choice is deterministic from the experiment seed, epoch, and source index.
One DataLoader is reused across phases; its shared phase/epoch state remains visible to
persistent workers.

## Model and objective

The decoder is split at `model.decoder.semantic_depth`. Shared blocks feed a
zero-initialized semantic head; remaining refinement blocks feed a separate
zero-initialized pixel head.

- `semantic` mode runs the shared decoder and semantic head only.
- `mixed` mode runs both heads. Pixel refinement receives detached shared state and
  conditioning, so pixel losses cannot update the encoder or shared decoder.
- `pixel` mode skips the semantic head and trains the full trunk and pixel branch end to
  end.

The ratio-mixed clean prediction is converted to velocity for the main MSE. Only
`pixel_ratio=0` items receive the direct auxiliary RGB MSE. One optimizer, scheduler,
and EMA continue across the progressive boundary; only the compiled/DDP execution
wrapper is rebuilt.

## Preparation and training

Preparation writes a resumable BF16 memmap for the selected DINO endpoint and computes
independent DINO/RGB channel statistics over the full training set. Manifest schema v3
is retained. An existing compatible manifest may contain extra, unused DINO layers and
will be reused without preparation.

```bash
torchrun --standalone --nproc-per-node=8 scripts/prepare_representations.py \
  --config configs/experiment.yaml

torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/experiment.yaml \
  --override training.mode=progressive \
  --output runs/progressive-seed0

torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/experiment.yaml \
  --override training.mode=pixel_only \
  --output runs/pixel-only-seed0
```

Settings support repeated dot-list overrides, for example:

```bash
--override training.semantic.mixed_probability=0.2 \
--override training.semantic.mix_ratio.scale=0.75 \
--override model.encoder.depth=16
```

## Sampling and evaluation

During semantic training, evaluation integrates only a pure-semantic trajectory with
`pixel_ratio=0`, then makes exactly one class-guided pixel prediction at `t=0`, also
with `pixel_ratio=0`. Pixel training evaluates a pure-pixel trajectory with
`pixel_ratio=1`. Both paths denormalize the final RGB tokens before previews and
FID/IS.

Standalone sampling accepts a completed pixel-phase checkpoint:

```bash
torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/progressive-seed0/checkpoints/last \
  --num-samples 50000 \
  --evaluate
```

Sampling is deterministic across batch sizes and rank counts. Interrupted archives
resume from their completion maps.

## Verification and compatibility

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check .
```

Configuration schema v7 and checkpoint schema v5 intentionally reject older
training artifacts. Representation manifest schema v3 remains reusable
when it contains the selected semantic cache and RGB statistics.
