# Pixel/RAE token-mixing DDT

This repository implements a schema-v14 ImageNet-256 experiment that keeps the
fast-converging representation features of RAE tokens while generating through a
lightweight pixel reconstruction path:

`RGB → pixel codec → normalized 1024×8×8 pixel tokens`

`RGB → frozen RAEv2 encoder → normalized 1024×8×8 RAE tokens`

The two endpoints are normalized independently with exact train-set, per-channel
statistics. Training then samples a ratio `r` and constructs

`x₀ = r · pixel_tokens + (1 - r) · rae_tokens`

without renormalizing the mixture. The DDT receives `(xₜ, t, r, class)`, where
`xₜ = (1 - t)x₀ + tε`. At generation and evaluation, `r=1`; the generated endpoint
is denormalized with pixel-token statistics and decoded by the small pixel decoder.

## Pixel codec

The deterministic convolutional codec maps 256×256 RGB images to 1024×8×8 latents.
Its reconstruction objective is L1 plus LPIPS. The encoder is also aligned to frozen,
normalized RAE features using both corresponding-token cosine loss and the full
64×64 within-image cosine-similarity matrix. The alignment coefficient is chosen
each step so its gradient norm at the shared pixel-token bottleneck is 10% of the
reconstruction gradient norm.

Build the frozen RAE cache once, then train the codec:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/cache_tokens.py \
  --config configs/pixel-codec.yaml

uv run torchrun --standalone --nproc-per-node=8 scripts/train_pixel_codec.py \
  --config configs/pixel-codec.yaml \
  --output artifacts/pixel-codec
```

The codec command selects the best validation-LPIPS checkpoint and writes the
portable artifact at `artifacts/pixel-codec/best`. It contains frozen weights,
pixel-token mean/std, per-channel normalized pixel/RAE correlation, data provenance,
and the RAE-cache fingerprint. DDT checkpoints bundle that artifact so sampling does
not depend on an external mutable codec path.

## Mixing experiment

The main preset samples `r` from an endpoint-augmented logit-normal distribution:

- 10% exact `r=0` (RAE endpoint)
- 80% `sigmoid(N(0, 1))`
- 10% exact `r=1` (pixel endpoint)

The pixel-only preset is the compute-matched control and fixes `r=1`. Both use the
same model, optimizer, online RGB pixel encoding, cached RAE tokens, and flow
timestep distribution `sigmoid(N(0.8, 0.8))`.

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/ddt-token-mix.yaml \
  --output runs/ddt-token-mix

uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/ddt-pixel-only.yaml \
  --output runs/ddt-pixel-only
```

The DDT predicts clean `x₀` with full and internal-guidance base heads; both are
trained through velocity loss. Per-step logs include sampled-r statistics, both
endpoint statistics, mixed-token statistics, endpoint cosine correlation, and the
variance expected from the measured endpoint correlation.

## FID and visualizations

Periodic and final validation always integrate with `r=1`, decode with the bundled
pixel decoder, save image grids/archives, and compute FID against the configured
ImageNet reference archive. Standalone sampling uses the same path:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/ddt-token-mix/checkpoints/last \
  --evaluate
```

Internal guidance defaults to `base + 1.78 · (full - base)` for `t ∈ [0.1, 1]`.
Euler and Heun solvers are supported; CFG is available but defaults to inactive.

## Reproducibility and compatibility

RAEv2 sources remain revision-pinned. The existing raw BF16 RAE cache format is
unchanged and resumable; its exact FP64-accumulated normalization statistics are
reused. Schema-v13 experiment configurations and checkpoints are intentionally
rejected because K-exploration and the checkpoint-prior experiment were removed.

```bash
uv run pytest -q
uv run ruff check .
```
