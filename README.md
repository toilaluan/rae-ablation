# DINOv3 token-mixing diffusion

This repository trains an ImageNet-128 diffusion transformer directly in a frozen
DINOv3-B/16 feature space. The previous RAE, checkpoint prior, internal-guidance
head, K-exploration, and persistent token cache have been removed.

The default pipeline extracts two endpoints in one online DINO traversal:

- target: block 2 patch tokens;
- semantic: final block 12 patch tokens;
- normalization: the same frozen DINO final LayerNorm and shared dataset-wide
  per-channel affine transform are applied to both endpoints;
- token layout: CLS and four register tokens are removed, leaving `768x8x8`.

## Stages

Install the environment, then generate the shared latent statistics once. The command
pools block-2 and block-12 patch tokens over the configured training split and writes
the file configured by `vision.statistics_path`:

```bash
uv sync --extra dev
torchrun --standalone --nproc-per-node=8 scripts/generate_latent_statistics.py \
  --config configs/dino-tokenmix.yaml
```

The statistics file records its dataset and DINO fingerprints. Training rejects a
file generated with different source data, preprocessing, or encoder weights.
For channel `c`, both endpoints use `z'[c] = (z[c] - mean[c]) / std[c]`; the mean
and population standard deviation are pooled across both endpoints, all training
images, and all patch positions.

Then train the transformer pixel decoder:

```bash
torchrun --standalone --nproc-per-node=8 scripts/train_decoder.py \
  --config configs/dino-tokenmix.yaml \
  --output runs/pixel-decoder
```

The decoder maps normalized block-2 tokens to RGB using an eight-block transformer.
Its objective is L1 + LPIPS, with an optional spectral-normalized PatchGAN that starts
after `decoder.loss.gan_start_fraction` of training. Checkpoints contain raw and EMA
decoder weights, discriminator state, optimizer state, provenance, and previews.

Point `decoder.checkpoint` at the decoder run (or a specific checkpoint), then train
diffusion:

```bash
torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/dino-tokenmix.yaml \
  --override decoder.checkpoint=runs/pixel-decoder/checkpoints/last \
  --output runs/tokenmix-seed0
```

For each image, training samples `r` from a logit-normal distribution, with a 0.2
atom at `r=1`, and constructs:

```text
x0 = (1 - r) * final_tokens + r * block2_tokens
xt = (1 - t) * x0 + t * noise
velocity target = noise - x0
```

The model is explicitly conditioned on `t`, `r`, and the class label. REPA aligns
diffusion encoder block 8 with final DINO tokens through a
`768 -> 2048 -> 2048 -> 768` projector at coefficient 0.5. Sampling fixes `r=1`
and supports ordinary classifier-free guidance.

## Matched baseline

Run the baseline with the same decoder checkpoint, seed, data, model, optimizer,
REPA loss, validation plan, and sampling settings:

```bash
torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/dino-r1-baseline.yaml \
  --override decoder.checkpoint=runs/pixel-decoder/checkpoints/last \
  --output runs/r1-baseline-seed0
```

The only intended experimental difference is `mixing.mode=target_baseline`, which
sets every training example to `r=1`.

## Sampling and evaluation

```bash
torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/tokenmix-seed0/checkpoints/last \
  --evaluate
```

Sampling is deterministic from the saved class-balanced validation plan, resumable,
and emits an ADM-compatible NPZ, preview grid, manifest, and optional FID result.
Training and decoder training retain console logs, W&B metrics and image grids,
EMA weights, resumable optimizer/RNG state, provenance validation, and optional W&B
checkpoint artifacts.

## Efficiency ablation

Measure convergence against the matched DINO `r=1` baseline first. A FLUX.2 VAE
experiment is useful as phase two: it tests whether token mixing helps beyond a
semantic foundation-model space, but VAE latents do not naturally provide the two
semantically ordered endpoints used here. Define a comparable second endpoint and
hold decoder/model compute fixed before interpreting it as an efficiency result.
