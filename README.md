# Adjacent-Flow DINO-to-Pixel DDT

This repository compares a progressive representation curriculum with a matched
RGB-only baseline on class-conditional ImageNet-128 generation. Every target is an
`8x8` grid of 768-dimensional tokens: configured DINOv3-B/16 intermediate layers or
flattened `3x16x16` RGB patches.

The default ordered stages are DINO layers 12, 8, and 4 followed by RGB, with 20 epochs
per stage. The list is data-driven: changing `representations.stages` changes extraction,
cache count, curriculum length, and the width of the model's representation condition
`R` without code changes. RGB must remain the final stage.

## Adjacent-flow curriculum

Each representation is independently normalized with full-training-set channel means
and standard deviations. There is no Procrustes, AsymFlow, projector, VAE, or latent
alignment preprocessing. Clean representations are mixed directly before one shared
Gaussian noise tensor is added.

For each item in every non-final stage, the default target distribution is:

```text
85.0%  pure current representation
10.0%  current/next mixture with a Uniform(0,1) next weight
 2.5%  pure next representation
 2.5%  all configured levels with a uniform-simplex R
```

The mode and weights are deterministic from the experiment seed, epoch, and source
index. Pure examples load one representation, adjacent mixtures load two, and only the
2.5% all-level examples load every representation. The final stage is pure RGB. Pixel
baseline mode is pure RGB for all 80 epochs.

## Model and objective

Both modes use the same universal DDT and have identical parameter counts. Its default
encoder is width 768, depth 12, and 12 heads; its decoder is width 2048, depth 2, and 16
heads. Dimensions are independently configurable under `model.encoder` and
`model.decoder`.

DDT uses separate noisy-token projections for its representation encoder and AdaLN
decoder, RMSNorm, QK normalization, 2D RoPE, SwiGLU, SDPA, class global tokens, and a
single zero-initialized clean-target head. There are no token or convolutional detail
heads. The absolute `R` vector is converted through learned level embeddings, added to
encoder spatial inputs and decoder conditioning, and appended as an encoder global
token. `R` is never dropped during CFG.

Training predicts clean `x0` and evaluates velocity-space loss on a linear Gaussian flow
with shifted-logit-normal timesteps. Every model parameter trains throughout one global
AdamW optimizer, EMA, and 80-epoch cosine schedule. Stage boundaries change only the
data distribution; they do not mutate the model or reset optimization state.

## Preparation and training

Preparation extracts every configured DINO layer in one encoder pass, writes separate
resumable BF16 memmaps, and accumulates independent DINO/RGB normalization statistics.
The default three DINO caches require about 352 GiB for full ImageNet. This schema uses
`data/direct-representations` by default and intentionally does not reuse alignment-era
manifests or caches. Existing runtime data is never deleted automatically.

```bash
torchrun --standalone --nproc-per-node=8 scripts/prepare_representations.py \
  --config configs/experiment.yaml

torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/experiment.yaml \
  --output runs/progressive-seed0

torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/experiment.yaml \
  --override experiment.mode=pixel_baseline \
  --output runs/pixel-baseline-seed0
```

All public settings are nested in `configs/experiment.yaml` and support repeated
dot-list overrides, for example:

```bash
--override curriculum.mixing.adjacent.probability=0.2 \
--override model.encoder.depth=16
```

## Sampling and evaluation

Sampling accepts only a completed final-RGB checkpoint and always supplies RGB one-hot
`R`. It follows one normalized RGB flow trajectory, applies interval CFG, reverses RGB
normalization, unpatchifies, and clamps to `[0,1]`.

```bash
python scripts/sample.py \
  --checkpoint-dir runs/progressive-seed0/checkpoints/last \
  --num-samples 50000 \
  --evaluate
```

Intermediate stages report overall and mixture-mode validation losses plus PCA feature
previews. Final comparisons include FID-50k, IS, wall-clock time, images seen, estimated
FLOPs, peak memory, and parameter counts.

## Verification

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check .
```

Configuration, checkpoints, and representation manifests use clean versioned schemas.
Alignment-era LightningDiT and legacy DDT/Mage artifacts are rejected rather than
migrated or overwritten.
