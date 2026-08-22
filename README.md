# Cascade DDT

Cascade DDT trains class-conditional diffusion transformers for ImageNet-256 using an
image target plus aligned semantic tokens. Select the image target with
`model.image_target`:

- `mage_latents` (the default) uses deterministic native Mage-VAE posterior means
  `T=[B,128,16,16]`;
- `pixels` diffuses RGB `T=[B,3,256,256]` directly in `[-1,1]`, without loading Mage;
- semantic tokens `S=[B,768,8,8]` are normalized, unpooled final-layer DINOv3-B/16
  patch tokens extracted from a 128px view.

Mage patches use size 2 and pixel patches use size 32; semantic patches use size 1, so
every route has the same 8x8 transformer-token grid. Each image-target patch is
concatenated with its aligned DINO patch and independently projected into the DDT
encoder and decoder widths. Mage latents receive no Flux BN transform or dataset
normalization.

The transformer follows the encoder-heavy/decoder-light design of the
[RAEv2 DDT](https://github.com/nanovisionx/RAEv2/blob/main/src/stage2/models/DDT.py),
reimplemented with local primitives. Presets use encoder/decoder widths `[1152,2048]`,
depths `[28,2]`, heads `[16,16]`, RMSNorm, QK-normalized attention, 2D RoPE, SwiGLU,
and timestep/class attention tokens. Decoder AdaLN and prediction heads are
zero-initialized.

## Architectures

Select exactly one architecture with `model.architecture`:

- `latent` builds `LatentDDT`, a pure image-target baseline with no semantic modules;
- `shared_cascade` builds `SharedCascadeDDT`, with route-specific projections and heads
  around one shared DDT core;
- `decoupled_cascade` builds `DecoupledCascadeDDT`, containing independent `SemanticDDT`
  and `ConditionalLatentDDT` denoisers with no shared parameters.

Cascade training evaluates both Gaussian-flow routes every step. Semantic and image
timesteps/noise are independently sampled from separately configurable shifted
logit-normal distributions. A seeded 25% mask selects detached online full-head
semantic predictions in place of detached clean semantic conditioning. When class
dropout selects the null class, the image route receives an all-zero semantic
condition; classifier-free sampling uses the same convention. The single backward
pass optimizes:

```text
L_image_latent + L_semantic_full + semantic_base_loss_weight * L_semantic_base
```

The semantic condition selected for each example is concatenated into both image-route
DDT patch streams. The semantic base prediction is read from encoder block 8 and the
full prediction is read from the decoder, preserving internal semantic guidance.

## Pinned representations

The Mage-VAE is pinned to `mage-flow-community/Mage-Flow` revision
`5edf242ad658cb85050ee077c6629d59e5388fe7`. The inference implementation is attributed
to the [official Mage source](https://github.com/microsoft/Mage/blob/76bec2bb3818863f470de7e867c2dc7f1d0bfd83/mage_flow/models/modules/mage_vae.py),
and the repository validates the published
[VAE config](https://huggingface.co/mage-flow-community/Mage-Flow/blob/5edf242ad658cb85050ee077c6629d59e5388fe7/vae/config.json)
contract. Encoding accepts 256px floating-point images in `[-1,1]` and always uses the
posterior mean. Decoding returns clamped `[0,1]` 256px images.

Generate semantic-only normalization statistics before online cascade training or
semantic caching:

```bash
torchrun --standalone --nproc-per-node=8 scripts/generate_semantic_statistics.py \
  --config configs/cascade-shared.yaml
```

## Independent caches

Image latents and semantic tokens have independent resumable, distributed BF16 memmap
caches. Each builder performs capacity preflight, maintains a completion bitmap, uses
asynchronous pinned-memory copies on CUDA, and atomically finalizes its manifest.

```bash
torchrun --standalone --nproc-per-node=8 scripts/cache_image_latents.py \
  --config configs/latent-baseline.yaml

torchrun --standalone --nproc-per-node=8 scripts/cache_semantic_tokens.py \
  --config configs/cascade-shared.yaml
```

The full ImageNet training image-latent payload is approximately 78.2 GiB. Its
fingerprint contains only dataset preprocessing, Mage identity, deterministic-posterior
policy, image size, latent schema, shape, and dtype, so one cache works with all three
architectures and is unchanged by the DDT migration. The presets enable the image
cache, plus the semantic cache for cascade models. Training only opens completed cache
manifests; a missing or incompatible cache fails instead of rebuilding it. With both
caches enabled, cascade training reads no RGB and instantiates neither frozen encoder;
a cached latent baseline uses no RGB, DINO, or training-time VAE.

Direct-pixel training bypasses the image-latent cache while retaining compatibility
with an existing semantic-token cache. Existing Mage and semantic cache fingerprints
and manifests are unchanged.

## Train and sample

Three presets are provided:

```text
configs/latent-baseline.yaml
configs/cascade-shared.yaml
configs/cascade-decoupled.yaml
```

Example:

```bash
torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/cascade-shared.yaml \
  --output runs/cascade-shared-seed0

torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/cascade-shared-seed0/checkpoints/last \
  --semantic-steps 10 \
  --image-steps 40 \
  --evaluate
```

Use direct pixels for any architecture with an override:

```bash
torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/cascade-shared.yaml \
  --override model.image_target=pixels \
  --output runs/cascade-shared-pixels-seed0
```

Cascade sampling generates semantic tokens with internal full/base guidance, creates
fresh independent image-target noise, generates the target conditioned on semantic
tokens with class guidance, and either decodes through Mage-VAE or emits pixels
directly. Semantic and image routes have independent `sampling.*_timestep_shift`
settings and matching standalone CLI flags. The legacy `--timestep-shift` flag still
sets both. The latent baseline skips the semantic route while retaining the identical
image-noise seed stream for paired runs. Evaluation defaults to ImageNet-256
references.

Cascade training validation also writes `semantic-pca-preview.png`, a per-image PCA
visualization of the first 16 generated semantic feature maps, and logs it to W&B when
enabled.

DDT uses configuration and checkpoint schema version 2. Configs and checkpoints using
the former shared training or sampling timestep fields are migrated on load by copying
their values to both routes. Current Mage model, EMA, optimizer, RNG, and cache state
remain resumable. A Mage-target checkpoint cannot be loaded as a pixel-target model or
vice versa.

## Verification

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check .
```
