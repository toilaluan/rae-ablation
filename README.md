# RAE-DDT with internal guidance and K-exploration

This repository implements one schema-v13 pipeline:

`ImageNet-256 RGB → frozen RAEv2 compressor → normalized 1024×8×8 tokens → DDT`

Training never loads images or RAE weights. The cache builder is the only RGB/encode
path; sampling is the only RAE/decode path. Schema-v12 configurations, checkpoints,
and cache manifests are intentionally rejected without migration.

## Architecture

The model follows upstream `DiTwDDTHeadIG`, adjusted to an 8×8 input grid. A 1×1
projection produces width-768 spatial tokens. Twelve encoder blocks jointly process
64 spatial, four timestep, and eight class tokens. The base x-prediction head exits
after encoder block four. The full path projects to width 2048, applies two
adaLN-conditioned decoder blocks, and predicts x with a separate head.

Both heads train with flow matching:

`x_t = (1 - t) clean + t noise`

X predictions are converted to velocity using `max(t, 0.05)`. The loss is
`full_loss + train.base_loss_weight * base_loss`; the default base weight is 1.0.
Timesteps are logit-normal and use the compact-token shift 4.

## Pinned codec

- Base: `toilaluan/raev2-dinov3l-k7@46ce5a7e76df99eb5528020d3c65cef528a355fa`
- Compressor: `toilaluan/raev2-compressor@846e2213dcf6218d3b726448ae52d8104c925c0f`

The base is loaded first and injected into the compressor with remote code enabled.
CUDA uses BF16; CPU uses FP32.

## Cache

Build the resumable cache before training:

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/cache_tokens.py \
  --config configs/ddt-baseline.yaml
```

Raw outputs are stored as BF16 with a byte-per-example completion bitmap. Once all
values exist, ranks partition the memory map, accumulate per-channel sums and squared
sums in FP64, all-reduce them, and save FP32 population mean/std. An interruption
after encoding resumes at the statistics pass without loading the codec.

For the pinned 1,281,167-example train split, `values.bf16` is 167,925,121,024 bytes
(156.3925 GiB), the bitmap is about 1.22 MiB, and statistics are 8 KiB. Provision at
least 170 GiB free. An uncompressed 1024×16×16 BF16 cache would be about 625.57 GiB.

## Presets and training

The four presets are:

| Preset | K early | K full |
|---|---:|---:|
| `configs/ddt-baseline.yaml` | 1 | 1 |
| `configs/ddt-k-early.yaml` | 4 | 1 |
| `configs/ddt-k-final.yaml` | 1 | 4 |
| `configs/ddt-k-hybrid.yaml` | 4 | 2 |

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/ddt-k-hybrid.yaml \
  --output runs/ddt-k-hybrid
```

For K exploration, every early-stage candidate is scored with the base prefix under
no-grad. The globally best `k_full` candidates are scored by the full path, and only
the winning candidate receives the dual-head training forward/backward. Candidate
chunking is configurable per stage and stable ties prefer the lower original index.

## Sampling

Internal guidance is enabled by default:

`guided = base + 1.78 * (full - base)` for `t ∈ [0.1, 1.0]`.

CFG defaults to scale 1.0 (inactive). If explicitly enabled, IG is applied to each
conditional branch first and CFG combines them afterward. Euler and Heun are
available.

### Checkpoint prior experiment

By default the flow source is standard Gaussian noise. To use the base x0 prediction
of a completed, normally trained checkpoint as the source, set its directory and
choose either its raw training weights or EMA weights:

```yaml
prior:
  checkpoint_dir: runs/ddt-baseline/checkpoints/epoch-0080
  weight_source: ema  # model or ema
```

For each Gaussian draw, the frozen prior model runs `forward_base_prediction` at
`t=1` with the true class label. Training and sampling use that prediction as the
flow endpoint. Setting `checkpoint_dir: null` restores the Gaussian source exactly.
Checkpoint-prior runs require K=1, and the prior checkpoint must use the same model,
RAE, and token-cache provenance as the student. Student checkpoints record the
selected prior weight digest and verify it on resume and sampling without copying
the prior weights.

```bash
uv run torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/ddt-k-hybrid/checkpoints/last
```

## Verification

```bash
uv run pytest -q
uv run ruff check .
```

Remote CUDA codec validation is opt-in through the integration test marker.
