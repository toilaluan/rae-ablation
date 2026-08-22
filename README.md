# CascadeDiT

CascadeDiT trains class-conditional diffusion transformers for ImageNet-256 using two
aligned native representations:

- image latents `T=[B,128,16,16]` are deterministic posterior means from the native
  Mage-VAE;
- semantic tokens `S=[B,768,8,8]` are normalized, unpooled final-layer DINOv3-B/16
  patch tokens extracted from a 128px view.

The image-latent patch size is 2 and the semantic patch size is 1, so both routes use
the same 8x8 transformer-token grid. Mage latents are not patch-packed and receive no
Flux BN transform or dataset normalization.

## Architectures

Select exactly one architecture with `model.architecture`:

- `latent` builds `LatentDiT`, a pure image-latent baseline with no semantic modules;
- `shared_cascade` builds `SharedCascadeDiT`, with route-specific projections and heads
  around one shared transformer core;
- `decoupled_cascade` builds `DecoupledCascadeDiT`, containing independent `SemanticDiT`
  and `ConditionalLatentDiT` denoisers with no shared parameters.

Cascade training evaluates both Gaussian-flow routes every step. Semantic and image
timesteps/noise are independently sampled. A seeded 25% mask selects detached online
full-head semantic predictions in place of detached clean semantic conditioning. The
single backward pass optimizes:

```text
L_image_latent + L_semantic_full + semantic_base_loss_weight * L_semantic_base
```

Every conditional image block receives the 64 semantic patches through zero-initialized
patchwise AdaLN shift, scale, and gates. Timestep and class tokens are never modulated.

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
architectures. All online/cache combinations are supported. With both caches enabled,
cascade training reads no RGB and instantiates neither frozen encoder; a cached latent
baseline uses no RGB, DINO, or training-time VAE.

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
  --override image_latents.cache.enabled=true \
  --override semantic_tokens.cache.enabled=true \
  --output runs/cascade-shared-seed0

torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/cascade-shared-seed0/checkpoints/last \
  --semantic-steps 10 \
  --image-steps 40 \
  --evaluate
```

Cascade sampling generates semantic tokens with internal full/base guidance, creates
fresh independent image-latent noise, generates image latents conditioned on semantic
tokens with class guidance, and decodes through Mage-VAE. The latent baseline skips the
semantic route while retaining the identical image-noise seed stream for paired runs.
Evaluation defaults to ImageNet-256 references.

## Verification

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check .
```
