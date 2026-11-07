# Original teacher targets and the observed-label output oracle

Two complete read-only measurements connect the original twelve-block W0
teacher to the fixed eight-block learning objective: **474 complete-piece
teacher forwards** against native W0, then **all 1,313 original cached
training pieces / 2,941,307 targets** for an exact output-only label oracle.

[Full teacher fidelity, including all 474 piece rows](results/teacher-target-fidelity.json) ·
[Full frozen-cache oracle, including all 1,313 piece rows](results/teacher-label-oracle.json) ·
[Teacher measurement source](teacher-target-fidelity.py) ·
[Label-oracle source](teacher-label-oracle.py) ·
[Complete depth32 coded result](depth-complete-32-mixed.md)

## Same-W0 observed-target fidelity

The CPU teacher uses the original twelve-block checkpoint,
`native_scaled_gemm=True`, frozen signed scales, full causal f16 PPMd priors
and every original complete piece boundary. Its zero-export identity matches
the original **2,904,325 B W0 static15** file. Scoring uses FP32 model logits
and FP64 logsumexp/target accumulation.

| Observed-target scope | Pieces | Targets | Native W0 NLL | CPU teacher NLL | CPU−native |
|---|---:|---:|---:|---:|---:|
| Complete sampled held | 473 | 1,082,505 | .966248338794 | .966346411131 | **+.000098072336** |
| Included held-long subset | 3 | 107,647 | 1.258614301310 | 1.258840193189 | **+.000225891879** |
| Independent giant article | 1 | 131,071 | .038986426132 | .038892187255 | **−.000094238876** |

All values are nats per observed target. Long-piece ordinals are 127, 157,
181; the giant is article243425 at source `[570595966,570727038)`.

Across the 473 complete held pieces, the target-weighted mean absolute
piece-mean delta is **.000729168564**; the unweighted median is
**.000660183768**, nearest-rank p95 **.003264960987**, and maximum
**.008838541310** at piece303 / 144 targets. Signed piece counts are
**229 positive / 244 negative / 0 zero**. The full result preserves every
piece mean and the derived short/long statistics.

This comparison's metric is observed-target NLL and complete-piece means.
All-205-class CPU/native distribution distances use paired complete
probability rows. The separately published
[original teacher-objective audit](teacher-objective-audit.md) retains its
own 205-column measurement scope.

The complete fidelity job uses **486.186503 CPU s** including its supervisor,
**493.982065 wall s**, and **3,331,712 KiB** peak child RSS. All **474 / 474**
forwards and **1,213,576** targets complete on one CPU/library thread.
The original cache, native exports and optimizer stay unchanged.

## Exact CE+KL label pull over the complete frozen cache

Let `q` be the frozen teacher's 205-class distribution at a training context,
`y` the observed next token and `r` an independently free output distribution.
The old output terms are

```text
J(r) = −ln r_y + KL(q || r)
r*   = (q + one_hot(y))/2
CE(r*) = ln2 − ln(1+q_y)
KL(q || r*) = ln2 + q_y ln(q_y) − q_y ln(1+q_y)
```

The coefficients of `−ln r_j` are `q_j + 1[j=y]`; normalization gives `r*`.
At the teacher output `r=q`, this objective is the observed-label teacher CE.
The optimum's observed-label mass is `(1+q_y)/2`, while every other class
gets `q_j/2`. This supplies an explicit output-level teacher-away direction
for the equal-coefficient observed-label CE + teacher KL terms.

The completed measurement reads every original FP32 **205-column** cache
row, computes probabilities/logarithms in bounded FP64 tiles and weights
each target once. It uses the original 1,313 full training pieces and labels.

| Complete-cache target-weighted statistic | Value |
|---|---:|
| Mean teacher probability at observed label, q_y | **.663905104246** |
| Teacher observed-label CE = J(q) | **.974118938239** |
| Oracle observed-label CE | **.216628074974** |
| Oracle KL(q‖r*) | **.162446955450** |
| Oracle output objective J(r*) | **.379075030424** |
| Output-objective descent from teacher | **.595043907815** |

The free-output oracle applies independently at each original training
context. The eight-block model's shared parameters, hidden-MSE term and
weight-rate regularization retain their separate training roles. The
[earlier teacher-objective audit](teacher-objective-audit.md) preserves its
bounded sample; this result covers the entire frozen training cache.

The cache-only job uses **5.077018 CPU s** including its supervisor and
**69,492 KiB** peak child RSS. Teacher forwards, native calls, optimizer
updates and cache writes are **0**. All per-piece means and original cache
recipe identity accompany the full empirical JSON.

## Reproduce the existing measurements

Both published sources are the original standalone CLI files. Use the
original checkpoint, CPU kernel, causal manifest/baseline and read-only
teacher cache produced by the [complete-piece method](depth-distillation.md).
Source paths point to its existing `cpu_model` modules. Output directories
are fresh experiment scratch paths.

```sh
python3 -B research/teacher-target-fidelity.py \
  --checkpoint /YOUR/checkpoint.tch --kernel /YOUR/cpu-kda.so \
  --data /YOUR/data-manifest.json --baseline /YOUR/baseline.json \
  --source /YOUR/cpu-source --output /YOUR/temp/teacher-fidelity

python3 -B research/teacher-label-oracle.py \
  --manifest /YOUR/data-manifest.json --cache /YOUR/teacher-cache \
  --output /YOUR/temp/teacher-label-oracle
```

`ORIGINAL/` labels in empirical metadata identify retained run artifacts;
the published JSON carries the complete measured piece rows. The next
complete experiment is a teacher-only fixed-eight-block control, using the
same original cache, zero8 initialization, fresh optimizer, complete
32-epoch export schedule and its own selected-weight codec measurement.
The [original 2KL comparison](teacher-objective-audit.md) preserves output
curvature and the original hidden/rate coefficients. Its execution is pending.
