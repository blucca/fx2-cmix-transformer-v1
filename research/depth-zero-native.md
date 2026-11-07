# Eight-block native loss before learning

The original **zero8** export measures the retained eight-block mapping before
its first optimizer update. On the same full held scope, learning through e12
repairs **74.33%** of its initial loss gap against W0. The independent giant
article improves at e4, then passes zero8 at e8 and continues to regress.

This control separates the starting mapping loss from the subsequent learning
trajectory. The original [32-epoch method](depth-distillation.md) and its sole
terminal-selected same-weight mixed measurement continue unchanged.

## Fixed model and score identities

W0 is the original pretrained twelve-block reference, **2,904,325 B**. Zero8 is
the original **1,955,187 B** `step-0000.tfwc3`: original blocks
`0,1,2,3,8,9,10,11`, renumbered into six KDA and two attention blocks, with the
four retained skip pairs. Its 294-tensor static15 export preserves the mapped
INT4 symbols, raw values, signed BF16 scales and original RoPE arrays. Changed
weight symbols, z coordinates, raw values and scale delta are all zero.

The existing eight-block `activation-probe eval-pieces` evaluated the original
f16 causal priors and explicit boundaries. Vocabulary=205, attention=`KVI8`,
RoPE offset=0 and positive-u8=0 match the original W0 score identities. The
held scopes are excluded from this adaptation's updates; W0 remains the
original pretrained reference.

| Scope | Pieces | Input rows | Scored targets | Role |
|---|---:|---:|---:|---|
| Full held (`held`) | 473 | 1,082,978 | 1,082,505 | Original selection support |
| Held long (`held_long`) | 3 | 107,650 | 107,647 | Subset of full held, report-only |
| Held short (`held_short`) | 470 | 975,328 | 974,858 | Complement within full held |
| Independent max-audit (`maximum_length_audit`) | 1 | 131,072 | 131,071 | Separate giant article, report-only |

The held-long indices are `[127,157,181]` in original zero-based manifest order.
The max-audit stays separate from the held total and candidate ranking. Each
explicit `[a,b)` piece resets state and scores rows `a` through `b-2` against
next tokens `a+1` through `b-1`. The final prior/EOF row is excluded; the first
token has no preceding prediction. This gives `input_rows − pieces` targets.
The held and audit prior files contain **444,020,980 / 53,739,520 B** respectively
(205 f16 columns per input row).

For a scope's piece set I, every reported mean uses its target weight:

```
N_I = sum_i targets_i
L_I = fsum_i logits_sum_i / N_I
```

All 473 bounds and per-piece losses, both input halves, FP32-logit and
probability loss sums, original probability counters and all four complete
zero8/W0 native outputs are in the [full scientific result](results/depth-zero-native.json).

## Same-scope measurements

Raw native loss is in nats per scored target. Values below are rounded to 12
decimal places; the complete result retains the original precision.

| Scope | W0 | Zero8 | e4 | e8 | e12 | e16 |
|---|---:|---:|---:|---:|---:|---:|
| Full held | .966248338794 | 1.663139060115 | 1.187158519088 | 1.168803883378 | 1.145141392609 | 1.149255265863 |
| Held long | 1.258614301310 | 2.052032208420 | 1.518069217219 | 1.498125758357 | 1.466907443103 | 1.472786730094 |
| Held short | .933964335620 | 1.620196210248 | 1.150618280488 | 1.132439087813 | 1.109610935827 | 1.113529866338 |
| Independent max-audit | .038986426132 | .222665869994 | .186305475316 | .224976626208 | .268978606267 | .275808524938 |

The learned checkpoints reuse the original [e4](results/depth-first-native-e4.json),
[e8](results/depth-continuous-terminal-e8.json), [e12](results/depth-first-hard-e12.json)
and [e16](results/depth-hard-e16.json) measurements. Each is its native hard
export at that completed epoch, including the continuous-stage e4/e8 exports.

| Scope | Zero8 − W0 | e4 − zero8 | e8 − zero8 | e12 − zero8 | e16 − zero8 |
|---|---:|---:|---:|---:|---:|
| Full held | +.696890721321 | −.475980541027 | −.494335176737 | −.517997667506 | −.513883794252 |
| Held long | +.793417907110 | −.533962991201 | −.553906450063 | −.585124765316 | −.579245478326 |
| Held short | +.686231874629 | −.469577929760 | −.487757122436 | −.510585274421 | −.506666343910 |
| Independent max-audit | +.183679443862 | −.036360394677 | +.002310756214 | +.046312736274 | +.053142654944 |

On full held, e12 recovers `.517997667506 / .696890721321 = 74.329826996%`
of the zero8-to-W0 gap. Its remaining W0 gap is **+.178893053815**;
e12→e16 returns **+.004113873254**. Both held subsets share the recovery and
late return.

On the independent giant, e4 initially recovers **.036360394677**, then e4→e16
returns **+.089503049621**. The e16 W0 gap **+.236822098806** decomposes into
the starting zero8 gap **+.183679443862** plus the learned change
**+.053142654944**. The starting component is **77.56%** of that measured gap.
This audit gives the late learning drift its own control alongside the broader
initial eight-block distortion.

## Actual resource scope

One original read-only execution ran the two scopes serially. Its recorded
resources are:

| Scope | Child CPU seconds | Child wall seconds | Peak process RSS, KiB |
|---|---:|---:|---:|
| Full held | 85.086023 | 85.209645 | 92,868 |
| Independent max-audit | 11.585860 | 11.595560 | 91,928 |

**96.722153 CPU seconds** includes the supervisor and both native children;
supervisor CPU is .050270 seconds and child CPU is 96.671883 seconds.
Scientific wall time is **96.827034 seconds**. The **92,868 KiB** peak is the
largest individual process RSS across the supervisor and children. These
counters describe this baseline execution; the original training ledger
retains its separate 48-CPU-hour cap.

CPU affinity `[1]`, numerical-library threads=1, address-space limit **4 GiB**,
and the **180 CPU-second / 240 wall-second** gates passed. Native calls=2;
reused native calls=0. New training updates, builds, data and scientific reruns
are zero. Publication reuses these successful native outputs and the
[original passed stdlib receipt](results/depth-zero-native-check.json).

## Next complete method

The [curvature-matched 2KL objective](teacher-objective-audit.md) addresses the
observed teacher-away drift after e4 on the giant and after e12 on full held.
The next complete measurement must also preserve the already measured repair
of the initial mapping gap. The [parameter-gradient diagnostic](depth-objective-gradients.md)
provides a local mechanism for the competing label/teacher pressure.

**Collect the original 32-epoch terminal and its sole selected-W mixed receipt
before investing in the complete 2KL method.** The next 3–5 active-hour task
covers the distinct objective/run identity, fresh Adam, validated reuse of the
original complete teacher cache, full native curve and its own sole same-weight
mixed measurement. Its full resource ledger includes training, exports and
native evaluation; the existing cadence places a 32-epoch cache-hit planning
center at **24.345 CPU hours**, with a **48 CPU-hour learning cap** and a
separate mixed budget of **8 CPU hours**. Actual selected-W coded bytes,
double-W/B costs and complete original-file restoration determine net archive
`S = P + 2B + 2D + O + 2W + 69`. The same four
zero8/W0 scopes remain fixed comparison anchors.

## Reproduce

The [stdlib replay](check-depth-zero-native.py) consumes the complete result,
including its two inline raw logs, without model dependencies:

```sh
python3 research/check-depth-zero-native.py
# Or supply another complete result:
python3 research/check-depth-zero-native.py /YOUR/temp/zero-native/result.json
```

It checks original scope/count identities, complete bounds and final-row mask,
target-weighted FP32-logit/probability means, W0 and e4/e8/e12/e16 differences,
and the recorded CPU/wall/RSS budgets. The [compact readout](results/depth-zero-native-compact.json)
provides the same four-scope comparison and recommendation.

The [measurement source](depth-zero-native.py) keeps the executed evaluation,
weight-preservation, aggregation and deadline logic. Publication adds standalone
CLI paths and CPU choice; relative manifest score paths resolve under `--job`.
The complete result and [reference context](results/depth-zero-native-context.json)
use logical `JOB/…` asset names.

For an independent reproduction, provide the original eight-block job layout:
`training/step-0000.tfwc3`, `native/depth8/activation-probe`,
`incoming/data-manifest.json`, `incoming/baseline.json`, and the existing
score tokens/f16 priors/i32 bounds named by the manifest. The manifest has the
original `causal-whole-piece-v1` source identity, samples `causal` / `long-audit`
and their fixed score scopes. The existing [CPU student/export](depth_student.py),
[native family build](depth-native-build.py) and [causal input method](causal-priors.md)
supply their public construction paths. Score paths in a reconstructed
manifest can be absolute or relative to the job root.

```sh
python3 research/depth-zero-native.py \
  --job /YOUR/temp/original-depth-job \
  --output /YOUR/temp/zero-native --cpu 1
```

`--context` can select the original compatible reference-context file.
The source first reuses matching existing `training/` or `check/` zero logs;
otherwise it invokes the two read-only native scores once, under the original
limits. It writes result/log/checkpoint files into the fresh output directory.
This publication made **zero new native/model/codec calls**.
