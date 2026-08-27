# Joint Semantic-Flow DDT

This repository pretrains a semantic-to-RGB pixel decoder, then trains one
class-conditional ImageNet-128 flow model for 200 epochs. The flow trunk denoises
normalized DINOv3-B/16 layer-12 tokens. Its pixel branch begins from the standalone
decoder and learns additional conditioning from selected flow hidden states.

The 12-block flow encoder has a zero-initialized RAEv2 base prediction at block 4 and
a two-block AdaLN semantic decoder for the full prediction. Both heads predict clean
semantic tokens and receive independent flow-matching velocity losses. Sampling uses
single-pass internal guidance over raw timesteps in `[0.10, 1.0]`:

```text
guided_x0 = base_x0 + 1.78 * (full_x0 - base_x0)
```

There is no classifier-free guidance evaluation, unconditional batch, endpoint
mixing, or direct pixel flow.

## Pixel decoder pretraining

The standalone decoder maps normalized `[B,64,768]` DINO tokens to RGB using two
semantic input projections, the configured spatial pixel transformers, a coarse
`3x16x16` patch projection, and the patch-isolated DiP U-Net. Its objective is the
same RGB L1 plus VGG-LPIPS loss used during flow training. Class labels are neither
loaded nor required.

The default cached mode reuses the existing ImageNet training cache and prepares a
validation cache. The generic dataset configuration accepts any revision-pinned
Hugging Face image dataset and image column. Online mode avoids BF16 token caches:
it makes one distributed DINO statistics pass, then runs the frozen encoder per
training batch.

```bash
torchrun --standalone --nproc-per-node=8 scripts/prepare_pixel_decoder_inputs.py \
  --config configs/experiment.yaml

torchrun --standalone --nproc-per-node=8 scripts/pretrain_pixel_decoder.py \
  --config configs/experiment.yaml \
  --output artifacts/pixel-decoder
```

Pretraining defaults to 50 epochs. Periodic directories under `checkpoints/` contain
raw and EMA decoder weights, semantic mean/std, optimizer/scheduler state, and exact
dataset/DINO/architecture provenance. There is deliberately no automatic “best”
selection; set `training.decoder.pretrained_checkpoint` to the chosen epoch and
`pretrained_weights` to `ema` or `model`.

## Flow-conditioned decoder

Post-block encoder features `[2, 4, 6, 8, 12]` and semantic-decoder features `[1, 2]` use
the configured pixel fusion. `layer_attention` normalizes and projects each source,
mixes the sources plus a learned DiP-context token across network depth, and emits
spatial and context pooled states. Each pooled state is concatenated with the flow
model's predicted-clean semantic tokens and projected into the corresponding
pretrained decoder stream.

Predicted-clean tokens are always detached from the RGB objective. If the pretraining
corpus has different DINO channel statistics, a checkpointed affine conversion maps
flow-normalized tokens into the decoder's normalization. The new hidden-state slices
are initialized to zero and the semantic slices are copied from pretraining, so the
initial joint RGB function exactly matches the standalone decoder.

Use the alternative fusion with:

```bash
--override model.pixel_decoder.fusion.kind=concatenate
```

## Flow preparation and training

Fresh preparation writes the selected training DINO BF16 cache and its channel
statistics. It also writes a separate approximately 94 MiB reconstruction-validation
cache containing the first ImageNet validation example from each class. Existing
compatible training caches remain reusable; only the held-out cache is added.
Manifest schema v3 remains reusable when an existing training manifest also contains
unused RGB stages or statistics.

```bash
torchrun --standalone --nproc-per-node=8 scripts/prepare_representations.py \
  --config configs/experiment.yaml

torchrun --standalone --nproc-per-node=8 scripts/train.py \
  --config configs/experiment.yaml \
  --output runs/joint-seed0
```

For the first `training.decoder.frozen_epochs` (20 by default), the pretrained pixel
transformers, context MLP, coarse head, and DiP detailer are frozen. The flow model,
hidden-state fusion, and new input projections train, and RGB gradients may reach the
flow trunk only through hidden states. At the boundary the decoder is unfrozen,
compiled/DDP execution is rebuilt, and the existing optimizer and scheduler continue
at the same learning rate.

The per-sample joint objective is:

```text
full_velocity_mse + base_velocity_mse + pixel_l1 + pixel_lpips
```

All weights default to `1.0`. RGB L1 uses `[0,1]`; LPIPS receives prediction and
target mapped to `[-1,1]`.

Decoder optimization defaults to every outer step with one update and pixel gradients
flowing into the flow trunk through hidden states. `training.decoder.update_interval=N` runs pixels only
every `N` outer steps; `updates_per_active_step=K` reuses detached hidden states for
the remaining `K-1` decoder-only updates. Average decoder work therefore scales with
`K/N`. Set `backpropagate_to_flow=false` to isolate all pixel gradients from the flow
trunk. `samples_per_batch=M` selects the `M` lowest-timestep (cleanest) samples on each
rank with fixed-shape `topk` indices and runs the pixel decoder only on that subset.
The checked-in preset uses `64`; setting it to `null` uses the full local batch.

At every validation interval, the EMA model also reconstructs the deterministic
1,000-image held-out set. Clean DINO tokens are passed unchanged while the model is
conditioned at `sampling.final_decode_timestep` (`0.5` by default). Validation logs
sample-weighted RGB L1, VGG-LPIPS, and PSNR, and saves an eight-row original versus
reconstruction preview. The stochastic joint-loss validation and periodic 10k FID
remain enabled.

## Sampling

Sampling integrates only the internally guided semantic flow. It then performs one
conditional forward at `sampling.final_decode_timestep` with pixel decoding enabled.
The checked configuration uses `0.5`; `scripts/sample.py` exposes
`--final-decode-timestep` for explicit overrides. RGB is clamped only while writing
previews and sample archives.

```bash
torchrun --standalone --nproc-per-node=8 scripts/sample.py \
  --checkpoint-dir runs/joint-seed0/checkpoints/last \
  --num-samples 50000 \
  --evaluate
```

Configuration schema v10 and checkpoint/training-state schema v7 reject older user
configurations. Schema-v9 checkpoint configs are migrated with a legacy final-decode
timestep of `0.0`, preserving their previous sampling behavior. Standalone decoder
artifacts use schema v1. Joint checkpoints embed all decoder weights,
normalization buffers, and the selected pretraining-weight checksum, so sampling and
resume do not require the original standalone checkpoint.

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check .
```
