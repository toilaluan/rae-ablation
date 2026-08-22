# DINOv3 semantic bridge ablations

This repository implements a reconstruction-first RAE experiment at ImageNet-128.
A frozen DINOv3-B/16 supplies paired representations from each image:

- `T`: the raw block-2 patch state, `[B, 768, 8, 8]`;
- `S`: final-LayerNorm block-12 patch tokens, `[B, 768, 8, 8]`;
- `X`: RGB patchified losslessly into `[B, 768, 8, 8]`.

Target, semantic, and RGB endpoints each use their own training-set statistics.
Schema v18 is a clean break: prior configurations, statistics, caches, pixel decoders,
and flow checkpoints are intentionally unsupported.

## Stage 1: raw block-2 to pixels

Train the transformer pixel decoder for 10 epochs, then freeze its best-LPIPS EMA
checkpoint. The objective is L1 + LPIPS with an optional delayed PatchGAN term.
Validation reports L1, LPIPS, PSNR, global SSIM, and reconstruction FID.

```bash
uv sync --extra dev

torchrun --standalone --nproc-per-node=8 scripts/generate_latent_statistics.py \
  --config configs/dino-cascade.yaml

torchrun --standalone --nproc-per-node=8 scripts/train_decoder.py \
  --config configs/dino-cascade.yaml \
  --output runs/pixel-decoder
```

Use `runs/pixel-decoder/checkpoints/best-lpips` for the target-flow experiment.

## Stage 2: matched semantic bridges

Both ablations use the same 131.3M-parameter LightningDiT-B/1-style model, optimizer,
batch size, 20 epochs, shifted logit-normal timestep distribution, standard-normal
prior, and immutable validation plan. The only changed variable is the endpoint:

```text
target route: S -> T -> frozen block-2 pixel decoder -> RGB
pixel route:  S -> X -> exact RGB unpatchification
```

For endpoint `E` (`T` or `X`), training independently samples
`epsilon ~ N(0, I)` and constructs `z_t = (1-t) E + t epsilon`. The model predicts
clean `E` with direct MSE. `S` is never part of the diffusion path: its 64 aligned
tokens combine with the timestep embedding to produce per-token AdaLN-Zero
modulation in every DiT block. ODE sampling recovers velocity as
`(z_t - E_hat) / t` and starts at `epsilon` when `t=1`.

An optional endpoint cache stores `T` and `S` once for both routes:

```bash
torchrun --standalone --nproc-per-node=8 scripts/cache_endpoints.py \
  --config configs/dino-cascade.yaml
```

Train the matched runs with the same seed and validation plan:

```bash
torchrun --standalone --nproc-per-node=8 scripts/train_target_flow.py \
  --config configs/dino-cascade.yaml \
  --override decoder.checkpoint=runs/pixel-decoder/checkpoints/best-lpips \
  --override vision.cache.enabled=true \
  --output runs/semantic-to-target

torchrun --standalone --nproc-per-node=8 scripts/train_pixel_flow.py \
  --config configs/dino-cascade.yaml \
  --override vision.cache.enabled=true \
  --output runs/semantic-to-pixel
```

Every validation evaluates both the default 40-step solver and direct one-step
prediction from the same deterministic, per-image Gaussian noise. Results are written
to `validation/epoch-*/metrics.json` with paired L1, LPIPS, PSNR, global SSIM,
reconstruction FID, and endpoint MSE.
Compare matching result files with:

```bash
uv run scripts/compare.py \
  runs/semantic-to-target/validation/epoch-0020-final/metrics.json \
  runs/semantic-to-pixel/validation/epoch-0020-final/metrics.json
```

`scripts/sample.py` re-evaluates a checkpoint on its immutable paired plan. Use
`--noise-seed` to select another reproducible standard-normal draw. It remains a
conditional reconstruction experiment because `S` comes from a source image.

## Verification

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check .
```
