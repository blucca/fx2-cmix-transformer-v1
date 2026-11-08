# A first-block embedding coefficient carries the teacher conflict

One bounded measurement on the running eight-block checkpoint moves the
[objective audit](teacher-objective-audit.md) from output logits to its
**actual 75 integer-coordinate and 59 raw parameters**.

The original checkpoint had completed 18 epochs: step 1472, partway through
its nineteenth. Selection was fixed in advance to the first complete
1,025–2,048-target training piece in manifest order. This was piece 8, with
**1,199 targets**, capture bounds `[5321,6521)` and its own `[0,1199]` NN reset.
The existing f16 causal priors and f32 twelve-block teacher cache were reused.

## Measured directions

Five data vector-Jacobian products separate CE, KL, weighted hidden loss and
both composite objectives; 75 independent rate VJPs bound memory. One
student forward supplies the graph. Optimizer instances and updates: 0.

| Parameter group | CE/KL cosine | Original gradient · KL gradient | Proposed 2KL gradient · KL gradient |
|---|---:|---:|---:|
| All 134 | −.859338261 | −3.089855900 | +7.786521084 |
| 75 bounded z | +.180350996 | +.000250273 | +.000368039 |
| 59 raw F32 | −.859382239 | −3.090106173 | +7.786153045 |

The original total is `CE + KL + 0.1 hidden MSE + rate`; the proposed total
is `2 KL + 0.1 hidden MSE + rate`. The latter is an inspected loss, with zero
training updates. In ordinary gradient descent, a negative dot with KL means
that the descent direction locally raises teacher loss.

The original hard-stage group rates are `0.01` for z and `0.0001` for raw.
After applying this 100-fold difference, the same point still has:

| Direction `v = −diag(group LR) * gradient` | KL directional derivative | CE directional derivative |
|---|---:|---:|
| Original | +.000306507890 | −.001268278538 |
| Proposed 2KL | −.000782295693 | +.001854653666 |

These are history-free derivatives of this one piece. The original AdamW
update retains historical moments, per-element preconditioning, hard-STE
geometry and the actual complete batch; it continues on its original run.

## Locate the opposing parameter

`raw/blocks.0.token_embedding_coefficient.value` contributes
**−3.351009849** of the original teacher dot; all remaining parameters sum to
**+0.261153949**. With the original group rates, this scalar contributes
−.000335100985, versus +.000028593094 from the rest. All ten parameters with
negative original teacher dots are raw parameters; all 75 z tensors have
positive dots on this piece.

This localizes the competing pressure inside the original learning method.
The broad eight-layer capacity residual, independent long-article support
and same-weight coded savings remain the next complete measurement targets.
This measured point belongs to the original CE+KL trajectory. The later
[independent teacher-only trajectory](teacher-only-distillation.md) has entered
its complete 32-epoch comparison; its native and coded measurements retain
their own checkpoint identities.

## Hidden-loss alignment at the measured checkpoint

The same compact result contains the ordinary parameter gradients of full KL
and **0.1 × final-hidden MSE**. Their pairing at this nineteenth-pass hard-STE
point is:

| Parameter group | KL / weighted-hidden cosine | KL gradient · weighted-hidden gradient |
|---|---:|---:|
| All 134 | +.914533311 | +1.191338923 |
| 75 bounded z | +.257986742 | +.000003049 |
| 59 raw F32 | +.914556957 | +1.191335874 |

Both groups have positive dots. With their original positive group step
sizes, descent along the weighted-hidden gradient locally decreases full KL
on this piece. This is the same one forward and the same published VJPs:
3,970,790 trainable elements, 1,199 causal targets, completed epochs 18,
step 1472. The original AdamW trajectory retains its historical moments and
complete-batch updates.

For the later [actual-label path objective](byte-path-quotient.md#actual-label-visitation-and-finite-capacity),
shared-parameter competition concerns its actual path directions. The
full-KL/hidden alignment here supplies a measured late-stage control for
that interpretation; the complete trained-weight/native/coded comparison
measures the resulting performance.

## Resource and reproducibility receipt

The single executed diagnostic used **9.005903 CPU seconds** including its
supervisor, **684,212 KiB peak process RSS**, and **10.605163 wall seconds**.
It ran on one spare CPU, with one BLAS/PyTorch thread, a 4 GiB address-space
limit and a 180 CPU-second / 240 wall-second budget. Teacher forward calls,
new prior/cache files, native/codec calls and optimizer updates are zero.

All 134 parameter gradients are finite. Direct composite VJPs match the
separate components to relative L2 error **1.90e−6 / 2.14e−6**. Hidden/output
unused parameters and the 59 raw rate zeros are explicitly zero-projected.
[Complete compact parameter counters](results/depth-objective-gradients.json)
preserve the individual tensors, paired dots, losses and measured cursor.

The [measurement tool](depth-objective-gradients.py) adapts paths and standalone
resource limits for reproduction. Its model/gradient accumulation block is
byte-identical to the single executed script; digest metadata was removed.
Run it against the original job's existing checkpoint and teacher cache:

```sh
python3 research/depth-objective-gradients.py \
  --job /YOUR/temp/original-depth-job --kernel /YOUR/temp/cpu-kda.so \
  --output /YOUR/temp/gradient-audit --cpu 1
```

Use `--source /YOUR/original/source/research` for that job's frozen source.
The published point can be replayed without model dependencies:

```sh
python3 research/check-depth-objective-gradients.py
```
