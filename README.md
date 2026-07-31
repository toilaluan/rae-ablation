# DDT with pixel or FLUX.2 latent diffusion

`jit-reg` trains a class-conditional DDT-B/16 on ImageNet-256. The diffusion
state can be either RGB pixels or normalized FLUX.2 VAE latents. REG jointly
denoises DINOv3 representation tokens, while REPA aligns an intermediate DDT
sequence with frozen DINOv3 targets. Pixel/latent input and REG/REPA are
independent choices.

This repository uses configuration schema v10. Older configs, checkpoints,
and training states are intentionally incompatible. Schema-v4 REG/REPA caches,
schema-v1 latent caches, and schema-v16 sampling manifests remain reusable.

| Input space | Diffusion tensor | `model.patch_size` | DDT grid |
| --- | ---: | ---: | ---: |
| `pixels` | `3×256×256` | `16×16` | `16×16` |
| `flux2_latents` | `128×16×16` | `1×1` | `16×16` |
| `flux2_latents` | `128×16×16` | `2×2` | `8×8` |

FLUX.2 first produces a raw `32×32×32` posterior mode for a 256×256 image.
Its official 2×2 patchification and stored batch-normalization statistics turn
that into the `128×16×16` tensor used for diffusion. `model.in_channels` and
`model.patch_size` describe this diffusion tensor directly: pixels use 3
channels and patch size 16, while FLUX.2 uses 128 channels and can use any
latent patch size that divides 16. REPA targets are resized to the resulting
DDT grid.

Following the
[DiffusionBench DDT design](https://github.com/End2End-Diffusion/diffusion-bench/blob/main/src/stage2/models/DDT.py),
DDT uses an unconditioned 12-layer, width-768 encoder followed by a shared
2-layer, width-2048 decoder. The encoder sequence is REG data, image patches,
REG-time tokens, image-time tokens, then class tokens. The primary REG/image
states receive their matching base-time embeddings and become token-aligned
decoder conditioning. Fresh REG and image embeddings are decoded jointly with
adaLN-Zero. One final normalization/modulation is shared by both outputs, with
only the dimension-specific linear projections kept separate.

Following the
[RAEv2 internal-guidance head](https://github.com/nanovisionx/RAEv2/blob/main/src/stage2/models/DDT.py#L212),
REG also has a zero-initialized head at encoder block 4. It predicts clean REG
tokens before the shared decoder and is trained with the same
velocity-equivalent clean-prediction loss as the final REG output. Sampling
treats the early prediction as the internal unconditional estimate:
`early + reg_guidance_scale * (final - early)`.

Image and REG time use independent Gaussian Fourier embedders. Conditioning
token counts are configured with `model.num_image_timestep_tokens`,
`model.num_reg_timestep_tokens`, `model.num_image_class_tokens`, and
`model.num_reg_class_tokens`. Conditioning tokens use
identity RoPE and are excluded from image/REG prediction and REPA targets.

Horizontal flipping is not performed anywhere in the data, cache, or training
pipelines.

## 0. Environment setup

Python 3.12 is required.

```bash
uv sync --extra dev
uv run hf auth login
uv run wandb login
uv run pytest -q
uv run ruff check .
```

The latent path downloads only the `vae` subfolder of
`black-forest-labs/FLUX.2-klein-4B`, pinned by the configuration revision. The
VAE is frozen and uses bf16 on CUDA and float32 on CPU.

Download the ImageNet-256 ADM reference statistics before training validation
or FID evaluation:

```bash
mkdir -p data
curl -L \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz \
  -o data/VIRTUAL_imagenet256_labeled.npz
```

The presets are:

- `configs/baseline-p16.yaml`: pixel DDT-B/16 baseline.
- `configs/latent-baseline-p16.yaml`: online FLUX.2 latent baseline.
- `configs/reg-repa-p16.yaml`: pooled 4×4 REG plus block-4 REPA.

Any pixel feature preset becomes a patch-1 latent preset with:

```bash
--override model.input_space=flux2_latents \
--override model.in_channels=128 \
--override model.patch_size=1
```

To test 2×2 latent patches and an 8×8 DDT grid, use:

```bash
--override model.input_space=flux2_latents \
--override model.in_channels=128 \
--override model.patch_size=2
```

Overrides are structured OmegaConf dot-list values. Repeat `--override` once
per setting. Unknown or retired fields fail validation.

## 1. Build caches

Cache construction is resumable and distributed. Each rank processes a
disjoint subset, while rank zero owns capacity preflight, finalization, and the
build lock. Re-running the same command continues missing entries or exits if
the cache is already complete.

### 1.1 Cache FLUX.2 latents

For the latent baseline:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/cache_latents.py \
  --config configs/latent-baseline-p16.yaml
```

For a latent REG/REPA run:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/cache_latents.py \
  --config configs/reg-repa-p16.yaml \
  --override model.input_space=flux2_latents \
  --override model.in_channels=128 \
  --override model.patch_size=1
```

The cache stores deterministic posterior modes after FLUX.2 patchification and
normalization. `vae.micro_batch_size` bounds VAE encode/decode work independently
of loader and training batch sizes. `vae.cache.batch_size` controls the cache
loader and host buffers.

For the pinned ImageNet training split of 1,281,167 images:

| Cache | Approximate size |
| --- | ---: |
| FLUX.2 latents, bf16 | 78.2 GiB |
| 4×4 REG, bf16 | 29.3 GiB |
| 16×16 REPA, bf16 | 469.2 GiB |
| Latents + REG | 107.5 GiB |
| Latents + REPA | 547.4 GiB |
| Latents + REG + REPA | 576.7 GiB |

The REPA rows above assume patch size 1 and a 16×16 target grid. With latent
patch size 2, REPA uses an 8×8 grid and approximately 117.3 GiB; the latent
cache itself remains 78.2 GiB because VAE output shape does not change.

The latent formula is:

```text
dataset_size × 128 × 16 × 16 × 2 bytes + completion bitmap
```

The preflight uses the actual dataset length, so `data.max_samples` scales the
required capacity for smoke tests:

```bash
uv run python scripts/cache_latents.py \
  --config configs/latent-baseline-p16.yaml \
  --override data.max_samples=1024 \
  --override vae.cache.batch_size=32
```

### 1.2 Cache REG and REPA targets

Build only the active feature components:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/cache_features.py \
  --config configs/reg-repa-p16.yaml
```

For a latent REG/REPA experiment, use the same input-space override as training:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/cache_features.py \
  --config configs/reg-repa-p16.yaml \
  --override model.input_space=flux2_latents \
  --override model.in_channels=128 \
  --override model.patch_size=1
```

REG and REPA have independent fingerprints. Changing heads, losses, optimizer
settings, target scale, timing, sampling, or the other feature does not
invalidate a reusable component. REPA targets are large; begin with
`data.prefetch_factor=1` during cached training if host-memory pressure is high.

## 2. Train

`train.global_batch_size` is divided evenly by the distributed world size.
Every loader batch performs one optimizer step; there is no gradient
accumulation or separate per-device batch setting.

### Pixel baseline

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/baseline-p16.yaml \
  --output runs/pixel-baseline-p16
```

### Online latent baseline

This loads the frozen VAE on every rank and encodes each training batch:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/latent-baseline-p16.yaml \
  --output runs/latent-online-p16
```

### Cached latent baseline

Build the latent cache first, then enable it for training:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/latent-baseline-p16.yaml \
  --override vae.cache.enabled=true \
  --output runs/latent-cached-p16
```

When no online DINO targets are required, cached latent training reads only
labels, indices, and latent tensors; RGB images are not decoded or transferred.

### Cached latent REG/REPA

Build both caches using the commands in step 1, then run:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/reg-repa-p16.yaml \
  --override model.input_space=flux2_latents \
  --override model.in_channels=128 \
  --override model.patch_size=1 \
  --override vae.cache.enabled=true \
  --override feature_cache.enabled=true \
  --override data.prefetch_factor=1 \
  --output runs/latent-reg-repa-p16
```

Online and cached sources are selected independently. For example, a cached VAE
with online DINO targets keeps RGB images in the loader, while cached VAE plus
cached REG/REPA omits them.

### Resume training

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/latent-baseline-p16.yaml \
  --override vae.cache.enabled=true \
  --output runs/latent-cached-p16 \
  --resume runs/latent-cached-p16/checkpoints/last
```

Resume validates the dataset, validation plan, feature-cache fingerprints, and
latent-cache fingerprint. Switching between online and cached sources or
changing a semantic cache input requires a new run.

Checkpoints are written beneath `<output>/checkpoints`. Periodic generated
samples and FID results are written beneath `<output>/validation`. Set
`wandb.enabled=true` and the relevant `wandb.*` fields through YAML or overrides
to enable experiment logging.

The fused Gram Newton–Schulz optimizer remains an explicit hardware-specific
option:

```bash
uv sync --extra dev --extra fused-muon
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/baseline-p16.yaml \
  --override train.optimizer.kind=fused_muon \
  --output runs/fused-muon
```

It requires CUDA 12.9 or newer and compute capability 9.0 or newer. Unsupported
hardware fails instead of falling back to another optimizer.

## 3. Sample and evaluate

Standalone sampling inherits `sampling.*` and the VAE identity from the
checkpoint. Explicit flags replace individual sampling defaults.

### Sample

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/latent-cached-p16/checkpoints/last
```

Latent checkpoints generate `128×16×16` diffusion states and decode them to
256×256 RGB before preview and NPZ creation. Outputs are stored in a
parameter-derived directory inside the checkpoint. Sampling is rank-partitioned
and resumes only when the checkpoint, validation plan, model schema, and
sampling parameters match the existing manifest.

Override the solver and guidance:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/latent-reg-repa-p16/checkpoints/last \
  --solver heun \
  --sampling-steps 50 \
  --guidance-scale 2.0 \
  --reg-guidance-scale 1.78 \
  --reg-timestep-shift 1.0
```

### Sample and compute FID

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/latent-cached-p16/checkpoints/last \
  --evaluate
```

Rank zero writes `evaluation.json` beside `samples.npz` after distributed
sampling completes.

### Evaluate an existing NPZ

For a repository-produced archive, the adjacent manifest supplies the
validation-plan hash:

```bash
uv run python scripts/eval.py \
  --samples path/to/samples.npz \
  --reference data/VIRTUAL_imagenet256_labeled.npz \
  --output path/to/evaluation.json
```

For an external NPZ without an adjacent complete manifest, also provide:

```bash
--plan path/to/validation-plan.pt
```

### Compare two evaluations

```bash
uv run python scripts/compare.py \
  runs/pixel/evaluation.json \
  runs/latent/evaluation.json \
  --output comparison.json
```

The evaluations must use the same validation-plan hash.

## Script argument reference

Run any command with `--help` for its concise parser-generated reference.

### `scripts/cache_latents.py`

| Argument | Type/default | Description |
| --- | --- | --- |
| `--config` | path; `configs/latent-baseline-p16.yaml` | Schema-v10 latent experiment configuration. The input space must be `flux2_latents`. |
| `--override KEY=VALUE` | repeatable; none | Structured configuration override. Common cache controls are `data.max_samples`, `data.workers`, `vae.micro_batch_size`, `vae.cache.directory`, `vae.cache.batch_size`, and `vae.cache.log_interval`. |

The command writes under `vae.cache.directory/<fingerprint>` and never
overwrites an incompatible cache.

### `scripts/cache_features.py`

| Argument | Type/default | Description |
| --- | --- | --- |
| `--config` | path; `configs/baseline-p16.yaml` | Schema-v10 experiment configuration. REG or REPA must be enabled. |
| `--override KEY=VALUE` | repeatable; none | Structured override controlling active components, dataset limits, cache directory, batch size, logging, or preprocessing identity. |

The command writes independent `reg/<fingerprint>` and `repa/<fingerprint>`
components beneath `feature_cache.directory`.

### `scripts/train.py`

| Argument | Type/default | Description |
| --- | --- | --- |
| `--config` | path; `configs/baseline-p16.yaml` | Schema-v10 experiment configuration. |
| `--override KEY=VALUE` | repeatable; none | Structured override applied before final validation. |
| `--output` | path; none | Run root. Rewrites checkpoint and validation-plan paths beneath this directory. Without it, configured paths are used directly. |
| `--resume` | checkpoint path; none | Resume model, EMA, optimizer, RNG streams, epoch/step, validation state, and cache identities from a compatible checkpoint directory. |
| `--log-interval` | integer; `10` | Training-step interval for console/W&B metric emission. |

Training also requires the configured FID reference file to exist because
validation is part of the run.

### `scripts/sample.py`

| Argument | Type/default | Description |
| --- | --- | --- |
| `--checkpoint-dir` | path; required | Self-contained checkpoint directory, or a checkpoint root containing `last`. |
| `--num-samples` | integer; checkpoint value | Number of class-balanced samples. Must be divisible by `model.num_classes`. |
| `--batch-size` | integer; checkpoint value | Per-rank generation batch size. VAE decode is further chunked by `vae.micro_batch_size`. |
| `--sampling-steps` | integer; checkpoint value | Number of global Euler/Heun solver steps. |
| `--solver` | `euler` or `heun`; checkpoint value | ODE solver. Heun uses a final Euler step. |
| `--guidance-scale` | float; checkpoint value | Classifier-free guidance scale. |
| `--reg-guidance-scale` | float; checkpoint value | Internal REG guidance scale between the early and final clean-token predictions. |
| `--timestep-shift` | positive float; checkpoint value | Image/latent local-clock shift. |
| `--reg-timestep-shift` | positive float; first checkpoint candidate | REG local-clock shift used by standalone sampling. |
| `--img-denoising-interval START END` | two floats; checkpoint value | Fractional global interval assigned to image/latent denoising. |
| `--reg-denoising-interval START END` | two floats; checkpoint value | Fractional global interval assigned to REG denoising. |
| `--cfg-start` | float; checkpoint value | Start of the CFG window in image-local time. |
| `--cfg-end` | float; checkpoint value | End of the CFG window in image-local time. |
| `--log-interval` | integer; checkpoint value | Sampling progress interval. |
| `--evaluate` | flag; false | Compute FID on rank zero and write `evaluation.json` after sampling. |
| `--reference-npz` | path; checkpoint evaluation path | Override the FID reference used with `--evaluate`. |

All interval endpoints must satisfy `0 <= start < end <= 1`; CFG allows equal
start/end endpoints.

### `scripts/eval.py`

| Argument | Type/default | Description |
| --- | --- | --- |
| `--samples` | path; required | ADM-compatible NPZ containing `arr_0` as `N×256×256×3` uint8 images. |
| `--reference` | path; required | NPZ containing reference `mu` and `sigma`. |
| `--output` | path; required | Destination `evaluation.json`. |
| `--plan` | path; optional | Validation plan for external archives. Optional when a complete adjacent sample manifest supplies the same hash. |
| `--batch-size` | integer; `64` | Inception feature-extraction batch size. |
| `--device` | device string; `auto` | Evaluation device such as `auto`, `cpu`, or `cuda:0`. |
| `--log-interval` | integer; `10` | Feature-extraction progress interval. |

### `scripts/compare.py`

| Argument | Type/default | Description |
| --- | --- | --- |
| `first` | path; required positional | First `evaluation.json`. |
| `second` | path; required positional | Second `evaluation.json`. |
| `--output` | path; optional | Write the comparison JSON. The comparison is always printed to stdout. |

## Model conditioning tokens

The standard presets use one image-timestep token and one class token. REG
presets also use one REG-timestep token:

```yaml
model:
  num_image_timestep_tokens: 1
  num_reg_timestep_tokens: 1  # must be 0 when REG is disabled
  num_image_class_tokens: 1
  num_reg_class_tokens: 1  # must be 0 when REG is disabled
  decoder:
    hidden_size: 2048
    depth: 2
    num_heads: 16
    mlp_ratio: 4.0
```

Each count may be zero. `TimestepTokenEmbedder` uses a fixed random Fourier
frequency vector, an MLP, and learned token offsets initialized with standard
deviation `1/sqrt(hidden_size)`. `ClassTokenEmbedder` owns one shared label
table and separate image/REG offsets, keeping runtime class activations private
to their streams. Encoder outputs plus their matching image or REG base-time
embeddings provide token-aligned decoder conditioning.

## Feature configuration

REG and REPA remain configured under `model.features`. REG predicts clean
DINOv3 tokens with a clock coupled to the image timestep. REPA aligns an
intermediate DDT encoder sequence with frozen targets and can cover image
tokens or image plus REG tokens.

```yaml
model:
  features:
    encoder:
      model_name: vit_base_patch16_dinov3.lvd1689m
      feature_dim: 768
      register_tokens: 4
    reg:
      enabled: true
      pool_size: 4
      target_std: 0.53
      spatial_rope: true
      loss_weight: 0.03
      internal_guidance_block: 4
      attention_mode: asymmetric  # joint | asymmetric
      join_after_block: 4
      dropout_probability: 0.0
    repa:
      enabled: true
      block: 4
      projection_hidden_size: 2048
      loss_weight: 0.5
      token_scope: image_and_reg  # image | image_and_reg
```

In asymmetric mode, REG data and both conditioning streams remain private.
Image data begins attending REG after `join_after_block`; REG dropout keeps
image queries isolated in both the encoder and decoder.

During training, image/latent times are sigmoid-normal draws. Each example then
draws `delta ~ N(train.reg_shift_loc, train.reg_shift_scale^2)` and sets
`t_reg = sigmoid(logit(t_image) + delta)`. The standard presets use
`reg_shift_loc: -0.8` and `reg_shift_scale: 0.8`.

Sampling uses one global solver loop with independent local clocks and
quantized denoising intervals. Image CFG uses class-conditional and null-class
image predictions inside its configured window. REG instead uses early and
final class-conditional REG predictions throughout the REG denoising interval;
null-class REG predictions are ignored.
