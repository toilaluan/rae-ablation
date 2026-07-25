# `exp.yaml`: intermediate semantic prediction

The baseline JiT predicts clean pixels \(x_0\) directly. REPA adds an intermediate DINO
patch-alignment loss, but its projected features are not passed to later denoising blocks.
Standard REG jointly denoises a DINO token, yet predicts its clean state only after the final
block.

`exp.yaml` instead jointly diffuses 17 DINOv3 tokens (CLS + pooled 4×4 spatial tokens) and
predicts their clean semantic \(x_0\) after block 4. The clean CLS prediction conditions the
remaining blocks, while a wide CLS latent is compressed into four continuation tokens. This
matches the JiT objective—predict clean pixels and clean semantics—and lets later blocks use the
current semantic estimate during pixel denoising. The two 2048-wide SwiGLU MLP blocks are
inspired by RAE's wide processing of representation tokens. REPA and patch reprojection are
disabled in this experiment.

```text
image_tokens = patch_embed(noisy_image)
reg_t = diffuse([DINO_CLS, pool_4x4(DINO_patches)], timestep)

h = concat(project(reg_t), image_tokens)
h = SiT_blocks_1_to_4(h)

reg_x0, wide_cls = wide_MLP(h.reg, timestep, class_label)
conditioning += project(reg_x0.CLS)
h = concat(project_to_4_tokens(wide_cls), h.image)

h = remaining_SiT_blocks(h, conditioning)
image_x0 = pixel_head(h.image)
loss = image_velocity_loss + 0.03 * reg_velocity_loss
```

## Extra FLOPs with 16×16 patches

For 256×256 images, 16×16 patches produce 256 image tokens. The table keeps the experiment fixed
at 17 input REG tokens, block-4 prediction, a 2048-wide head, and four continuation tokens while
using the [standard SiT S/B/L/XL widths and depths](https://github.com/willisma/SiT/blob/main/models.py).

| Backbone | Baseline | `exp.yaml` | Extra FLOPs |
| --- | ---: | ---: | ---: |
| SiT-S/16 (384 × 12) | 12.40G | 14.70G | +2.30G (+18.50%) |
| SiT-B/16 (768 × 12) | 46.60G | 50.08G | +3.48G (+7.47%) |
| SiT-L/16 (1024 × 24) | 162.15G | 168.13G | +5.98G (+3.69%) |
| SiT-XL/16 (1152 × 28) | 238.12G | 245.69G | +7.57G (+3.18%) |

These are approximate per-example forward FLOPs with one multiply-add counted as two FLOPs.
They exclude backward, CFG duplication, losses, and DINO extraction. The checked-in YAML uses
32×32 patches; use `--override model.patch_size=16` for this comparison.

Most of the overhead comes from carrying semantic tokens through self-attention. The better FID
is plausibly due to making clean semantic predictions available during denoising, although a
matched comparison should also use the same augmentation and target-cache settings.
