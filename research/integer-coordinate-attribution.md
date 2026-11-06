# Fixed scales, cold-prior fitting, and the actual rate signal

**2026-11-06 — one frozen integer-coordinate trajectory.** The fixed scales
remove the earlier tiny-scale optimizer failure. The new measurements identify
ordinary-coordinate fitting and PPMd prior-state transfer as the next quality
bottleneck. The current symbol entropy term is small relative to the data
gradient. Original production weights remain the reference.

## A controlled prior-state interaction

Epoch 4 was selected once: it had the lowest target-weighted native NLL on the
two cold held-out halves among completed epochs 1–7. Evaluate that exact W on
the **same 1,041,397 targets**, with identical tokens, article cuts, neural
resets, and either causal mature or fresh PPMd priors:

| PPMd input state | Original W NLL | Epoch-4 NLL | Change, nats/target |
|---|---:|---:|---:|
| Causal history from stream 0 | .9632509281 | .9672471097 | +.0039961816 |
| Fresh, same text | 1.0032902250 | 1.0002763637 | −.0030138613 |

The interaction is **+.0070100429 nats/target**. The original benefit of
history, .0400392969 nats/target, becomes .0330292540 at epoch 4. This is a
controlled observation for this text and checkpoint: fitting the cold inputs
helps the fresh-prior case and weakens use of the mature inputs. PPMd feeds
the model through its probability embedding and normalization; direct prior
logit mixing is disabled in the original configuration.

This additional selection costs **175.643060 CPU s**. The separately queued
terminal checkpoint retains its paired mature/fresh evaluation and the real
mixed-coder round trips.

## Separate the data and code-length gradients

The frozen objective is

```
J = CE_mean + KL(p_original || q_student)_mean + 2 Hsoft(z) / 586459020
```

For free per-example output probabilities, its data term has solution
`q* = (onehot(y) + p_original) / 2`. Taking expectation over a population with
conditional target distribution `u(y|x)` gives
`q* = (u + p_original) / 2`. When `u = p_original`, the population solution
is the teacher. Repeated empirical labels and the original population
therefore give different targets. Shared model parameters, hard forward
quantization and projected Adam determine the actual trajectory.

The read-only [gradient probe](integer-coordinate-gradients.py) evaluates
original coordinates and the immutable epoch-2 coordinates. Four whole pieces
are selected in advance as the first 1025–1536-target pieces in their respective
sources: prefix training **1346**, q2 training **1048**, prefix held-out
**1057**, independent mature **1454**. Two training pieces reproduce the
existing teacher-cache logits exactly. Optimizer updates and exports are **0**.

At the first committed hard-change checkpoint:

| Piece | ΔCE | KL | ‖gCE‖₂ | ‖gKL‖₂ | ‖2gH/N‖₂ |
|---|---:|---:|---:|---:|---:|
| Prefix training | −.00792658 | .01863372 | .0422076 | .0118561 | 1.38423e−6 |
| q2 training | −.01089668 | .02819168 | .0515692 | .0172322 | 1.38423e−6 |
| Prefix held-out | +.01213974 | .01344427 | .0409962 | .00973360 | 1.38423e−6 |
| Independent mature | +.00318819 | .01122215 | .0287245 | .00713793 | 1.38423e−6 |

The measured combined data gradient exceeds the H gradient by approximately
**22,372–40,312×** on these pieces. On the changed-symbol subset alone, the
triangle bound `‖gCE+gKL‖ ≥ |‖gCE‖−‖gKL‖|` gives **9,852–32,238×**.
All H coordinates have magnitude below Adam epsilon **1e−8**, with maximum
**2.19662e−9**. Only **.33–.40%** of data-gradient coordinates are below that
epsilon at epoch 2.

The endpoint CE–KL cosines are positive, **.0504–.3804**. The KL gradient
also has positive dot product with the committed hard displacement,
**.01448–.03631**: its negative-gradient direction locally pulls against that
displacement. These measurements compare distinct vectors. The actual Adam
update uses accumulated first/second moments and projection. Endpoint
directional derivatives and the measured finite loss changes are recorded
separately. Original-point KL scalar is exactly zero; its FP32 coordinate
gradient norm is **4.94e−9–2.12e−8**, showing numerical reduction residuals.

Gradient-probe resources: **29.697984 CPU s**, peak RSS **842,936 KiB**,
single-thread CPU. All **429** original parameters and the immutable
coordinates pass their final equality checks.

## Where the symbols and entropy move

The [sequential census](integer-coordinate-census.md) processes all
**111 matrices / 5,868,864 coordinates**. At epoch 2:

- **3575** hard symbols change across **1258** rows. All changes are in
  ordinary-scale rows.
- The eight tiny Q/K rows contain **1536** coordinates, all at ±7. Their
  signed BF16 scales, hard symbols and fractional coordinates stay exact.
  The measured task-gradient row norms are approximately **1e−28–1e−24**;
  H points outward at those saturated coordinates and projection retains them.
- Hard entropy increases **77.835671 nats**, corresponding to **28.073 B**
  for two ideal weight code streams. Actual dual static15 W grows **24 B**.
- Trainer FP32 soft H increases **10,224.910156 nats**. Every matrix's soft H
  increases; hard H decreases in **28** matrices, stays equal in **13**, and
  increases in **70**. The largest soft increases are ordinary MLP matrices.
- `gH(z2)·(z2−q0) = +5.01743e−5`; its cosine is **+.07300**. Fractional
  movement follows the data fit, with a positive soft-entropy component.

The additive row ledger uses the same FP32 soft assignments with FP64 mass
reduction. Its soft ΔH is **10,225.031731 nats**; the trainer FP32 scalar is
stored alongside it. Both H gradients differentiate the frozen FP32 formula.
The full [census receipt](results/integer-coordinate-census-epoch02.json)
contains transitions, signed scales, row contributions and endpoint gradients.
Runtime is **5.904749 CPU s**, peak RSS **366,388 KiB**.

## The observed curve and the complete-package question

| Epoch | Changed symbols | Dual W growth, B | Cold held-out weighted ΔNLL |
|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 |
| 2 | 3575 | 24 | +.0012344003 |
| 3 | 15,631 | 110 | −.0042818136 |
| 4 | 37,702 | 320 | −.0044090752 |
| 5 | 65,990 | 598 | −.0037501872 |
| 6 | 104,621 | 1016 | −.0020227140 |
| 7 | 148,934 | 1570 | −.0000878732 |

Online, target-weighted training CE+KL decreases from **.86215794** in
epoch 1 to **.81336994** in epoch 7. Each mean covers the same **1,044,469**
training targets, with changing weights during its epoch. The independent
cold native curve peaks at epoch 4 and then loses most of that improvement.
Hard H at epoch 7 predicts dual W growth **1570.984 B**; actual growth is
**1570 B**.

For complete-package accounting, let `G` be actual payload bytes saved:

```
new net saving = G + 2*(W_original − W_candidate) − delta_program_bytes
```

The existing static15 component saves 52,022 B; the remaining current
record threshold is **952,225 B**. If the epoch-4 cold mean applied to all
**586,459,020** targets, and NN-to-coder transfer were 1, its net saving would
be **465,984.63 B**. Under those explicit conditions it needs transfer
**2.04275**, or raw NLL change at most **−.00900665** at transfer 1, to cover
the remainder. The controlled mature result gives a different distributional
signal.

The queued real coder measures this transfer on matched target masks, with
first-token fallbacks, post-mixer/post-SSE losses, range closure and actual
round-trip bytes recorded separately. The bounded **64-epoch / 48-CPU-hour**
trajectory remains intact. Its terminal native/mixed measurements settle this
fixed method; the causal prior-state mismatch identifies the question for a
future representative-data iteration.

[Machine-readable curve, gradients and prior-state control](results/integer-coordinate-attribution.json).

## Reproduce the read-only diagnostic

Use the same CPU helpers and installed dependencies as the trainer:

```sh
python3 -B research/integer-coordinate-gradients.py \
  --checkpoint models/6m-q4-fp32.tch --kernel /YOUR/temp/cpu-kda.so \
  --coordinates /YOUR/temp/trajectory/epoch-02-z.pt \
  --cases /YOUR/temp/gradient-cases.json \
  --output /YOUR/temp/gradient-attribution.json

python3 -B research/integer-coordinate-census.py \
  --checkpoint models/6m-q4-fp32.tch \
  --coordinates /YOUR/temp/trajectory/epoch-02-z.pt \
  --output /YOUR/temp/symbol-census.json --self-check
```

`cases` is the exact four-piece manifest in the machine-readable receipt.
The probes consume immutable inputs; trainer files and recipe stay fixed.
