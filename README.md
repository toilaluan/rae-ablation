# Self-Flow SiT ablations

This repository trains a class-conditional SiT-B/2 on normalized FLUX.2 latents
for controlled Self-Flow and SIGReg experiments. The model uses a 12-block,
width-768 transformer with fused SwiGLU, fused QKV attention, Q/K RMSNorm,
factorized 2D RoPE, and tokenwise AdaLN-Zero conditioning. Timesteps and class
labels are never appended as sequence tokens.

## Experiments

| Config | Objective |
|---|---|
| `configs/vanilla.yaml` | Homogeneous flow matching |
| `configs/dual-timestep.yaml` | Dual-timestep flow matching |
| `configs/self-flow-ema.yaml` | Dual timestep plus blocks 4/8 aligned to a frozen EMA cleaner branch |
| `configs/sigreg-truncated.yaml` | Shared-weight cleaner branch through block 8, alignment gradients on both branches, and SIGReg |
| `configs/sigreg-full.yaml` | Full shared-weight cleaner branch with equal mixed/cleaner velocity losses, alignment, and SIGReg |

Alignment layers are one-based, sorted, and configured with
`objective.align_layers`. SIGReg is computed independently at every spatial
position across the local-rank batch and averaged over both branches and all
configured layers.

## Setup and training

```bash
uv sync --extra dev
uv run python scripts/train.py \
  --config configs/self-flow-ema.yaml \
  --output runs/self-flow-ema
```

Use `--override vae.cache.enabled=true` to train from a latent cache. Build or
resume that cache with:

```bash
uv run python scripts/cache_latents.py --config configs/vanilla.yaml
```

The FLUX.2 latent encoding and cache fingerprint remain unchanged from the
previous repository version. Model configs and checkpoints use schema 11 and
older DDT artifacts are intentionally unsupported.

AdamW is the default optimizer. PyTorch Muon remains available with
`--override train.optimizer.kind=muon`.

## Sampling and evaluation

```bash
uv run python scripts/sample.py \
  --checkpoint-dir runs/self-flow-ema/checkpoints/last \
  --evaluate
```

Sampling supports Euler or Heun integration, classifier-free guidance, matched
validation seeds, resumable sample archives, and FID evaluation.

## Checks

```bash
uv run ruff check .
uv run pytest -q
```
