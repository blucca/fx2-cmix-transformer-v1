# When training rewards departure from the teacher

The original eight-block run lowered its online objective while its deployed
held loss rose. A source audit gives a controlled next experiment:
**replace CE + KL with 2 KL, retaining the original hidden and rate terms.**
The existing 32-epoch run and its single selected-weight mixed measurement
continue. This proposed ablation has zero model or codec runs.

## Original measurements

These are target-weighted, pre-update training means. Native scores use each
epoch's hard export on the same 473 held pieces / 1,082,505 targets.

| Epoch | CE | Teacher KL | 0.1 hidden MSE | Rate | Online objective | Held native |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 1.046434169488 | .152056791857 | .016506986950 | .035445732988 | 1.250443681282 | 1.168803883378 |
| 12 | 1.007020550239 | .143042755044 | .016685147837 | .035458391215 | 1.202206844336 | 1.145141392609 |
| 16 | .991812411294 | .144364694031 | .017158837640 | .035468412487 | 1.188804355452 | 1.149255265863 |

From e12 to e16, CE falls 0.015208139, while KL, hidden error and rate rise.
The complete online objective falls **0.013402489**; held native loss rises
**0.004113873 nats/target**. This late drift is 2.30% of e12's 0.178893054
held gap against the frozen twelve-layer reference. The broader capacity
residual and the independent giant-article audit remain measurement targets.
The [original complete e16 result](results/depth-hard-e16.json) preserves their
separate scopes; the audit's [compact accumulators](results/teacher-objective-audit.json)
add the actual target-weighted rate/objective totals.

## Preserve output curvature when changing the target

For student `p=softmax(z)`, frozen teacher `q`, one-hot label `y`, and
`F=diag(p)-p*pT`, gradients and Hessians in **output-logit coordinates** are:

| Output loss | Gradient | Hessian | Gradient at p=q |
|---|---|---|---|
| Original CE + KL(q||p) | 2p-y-q | 2F | q-y |
| KL(q||p) | p-q | F | 0 |
| Proposed 2 KL(q||p) | 2p-2q | 2F | 0 |

The factor two preserves the original output curvature and relative
hidden/rate coefficients. The teacher point loses the one-hot residual
pressure. Model parameter updates still pass through the existing STE,
activation quantization and Adam geometry.

The actual trainer sums CE/KL over targets and divides data gradients by the
batch target count. Hidden MSE averages 192 columns before this same target
mean. Every step then adds `2 H_soft(z)/586459020` for the two weight copies.
The proposed full objective is therefore:

```
2 mean_target KL(teacher || student)
+ 0.1 mean_target mean_192 (hidden_student-hidden_teacher)^2
+ 2 H_soft(z) / 586459020
```

## Measured parameter direction

[A single 1,199-target parameter-gradient measurement](depth-objective-gradients.md) now checks the same question on an original trained checkpoint. CE/KL cosine is −0.859338; the original LR-scaled descent direction raises KL locally while the proposed 2KL direction lowers it. A first-block raw embedding coefficient carries the opposing contribution. The original Adam trajectory and its actual coded measurement continue.

## Complete reuse and measurement contract

The existing teacher cache holds 205 f32 logits and 192 f32 final RMS features
for each of 1,313 complete training pieces: 2,941,307 contexts, 4,670,795,516
data bytes. These contexts cover 0.501745% of all NN targets; 32 repeated
epochs expose 94,121,824 targets. The f16 causal PPMd priors are separate inputs.

A future ablation starts from the same original outer-layer mapping, with
fresh Adam and a distinct run identity. Teacher-cache reuse checks the original
checkpoint, complete-piece inputs and same CPU inference path. Current
`--resume` and cache identities bind the entire old recipe. The new objective
logger must also use 2KL: the original logger's fixed KL coefficient is one.
CE remains a diagnostic from the already-computed logits.

The existing every-four-epoch export/native cadence costs 2,738.816966 CPU
seconds per epoch on e12-to-e16. Its **32-epoch planning center is 24.345 CPU
hours**, including that cadence. A future complete method retains the original
hard/static15 exports, independent held/long/audit scopes, its own single
same-weight mixed measurement, actual double-W/B charges and formal archive
admission. Current actual mixed results and any reusable compact model from
the upstream author decide the next complete investment.

Run the self-contained [standard-library replay](check-teacher-objective-audit.py):

```
python3 research/check-teacher-objective-audit.py
```
