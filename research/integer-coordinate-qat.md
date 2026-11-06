# Fixed-scale integer-coordinate CPU QAT

[`integer-coordinate-qat.py`](integer-coordinate-qat.py) learns only the
111 integer-coordinate matrices. It reuses [`cpu_model.py`](cpu_model.py),
the original whole-piece batching, [`cpu_weights_export.py`](cpu_weights_export.py),
the static15 recoder and the native evaluator. The original weights remain
the production baseline.

## Frozen method

Initialize FP32 coordinates with the original hard symbols:

```
z0 = round(original_weight / original_signed_BF16_scale).clamp(-7, 7)
forward_weight = STE(round(z)) * original_signed_BF16_scale
J = CE_mean + KL(original_W_distribution || student_distribution)_mean
    + 2 * sum(per_matrix_soft_entropy_nats(z)) / 586459020
```

The original BF16 weight scales, activation scales and every other model
parameter stay fixed, including the original signed 6–8e-15 Q/K scales.
Inference retains the original −7…7 symbols, signed scales, quantized
activations, 12 layers and 1024-token window. Entropy acts directly on bounded
`z`, using centers −7…7 and fixed temperature **0.5**.

Projected AdamW updates only `z`: **lr .01**, betas **(.9, .99)**, epsilon
**1e-8**, zero decay; each update clips coordinates to **[-7,7]**. The original
W teacher is frozen. Its FP32 logits are cached per complete piece with the
same original PPMd priors and position reset; cache storage is about
**820 B per training target**. CE and KL sums accumulate across whole pieces
until at least **32,768 targets**, then divide by the actual batch target count.
Each piece retains its complete gradient.

The original 429-key model state stays fixed for direct tensor and BF16-bit
audits. Export uses `export_state(model,z,scales)` to place hard integer-times-
signed-scale weights into the existing payload exporter. The recoder retains
all five original derived tensors. Checkpoints store `z`, AdamW, RNG and the
epoch/piece cursor; `--resume` reloads the original checkpoint for the fixed
parameters.

## Actual preflight, 2026-11-06

- Full zero-step static15: **2,904,325 B**, byte-identical to the original.
- Original complete piece **[11700,13047), 1346 targets**: student/teacher
  logits identical; CE **0.992835488397**, teacher KL **0**.
- **111/111** finite, nonzero gradient tensors; norm **0.041474457253**.
- One update changes **5,610,626** fractional coordinates, max |Δz|
  **0.00999999046**. Original **429** tensors and every scale's BF16 bits stay
  identical. All eight tiny scale rows retain their signed values.
- Hard symbols stay unchanged on this first .01 update; hard range
  **[-7,7]** and the post-update static15 bytes equal the zero-step file.
- New check CPU **14.58904 s**, peak RSS **705,432 KiB**. The historical
  original native baseline cost **82.945466 CPU s** is recorded separately.
  The check restores initial `z`/Adam/RNG and leaves a ready step-zero checkpoint.

The first raw receipt displayed post-update entropy under the same field as
the objective's pre-update entropy. The final source records these separately:
`soft_entropy_nats` for the objective, `current_soft_entropy_nats` for geometry.
The original raw preflight retains its measured values: **15,413,912.224609375**
before the update, **15,414,186.32421875** after it.

## Complete budget and coverage

One fixed trajectory: **64 epochs / 48 CPU hours maximum**, one CPU thread,
**7 GiB address-space limit**, including export/native scoring. Checkpoints
occur every eight updates and at epoch boundaries. `STOP` requests a clean
batch-boundary save; failures retain the last saved checkpoint and stop.
Every epoch measures real static15 bytes and native held-out NLL.

The initial learning manifest covers **807 complete pieces / 1,044,469 targets**
from the first halves of the prefix and independent q2 1 MiB samples. The
full method also includes mature PPMd sample evaluation, real mixed-coder
propagation and complete-package adjudication. These remain the quality and
compression decision stages following learning.

`--check-only` runs the full zero-step export and one real small-piece update,
then restores zero. `--zero-only` exports and saves zero. The CLI accepts the
existing checkpoint/kernel/manifest/native baseline/v2 template/recoder/evaluator
paths and an output directory; `python3 -B research/integer-coordinate-qat.py --help`
lists the interface.

## Mature validation and first committed hard change

[The mature-prior measurement](mature-ppmd.md) is complete. The same text
with causal PPMd history improves the original NN by .0400393 nats/target
versus fresh priors. The first epoch retains every hard symbol and all native
scores; epoch 2 changes 3,575 symbols, grows W by 12 B, and changes the two
cold held-out NLLs by +.00412299 / −.00169225. Its independent mature NLL
increases by .00590677. All original scales remain fixed. The single 64-epoch /
48-CPU-hour trajectory is complete; the original online mixed pair has started.
[Full machine-readable receipts](results/integer-coordinate-mature-ppmd.json).

## Measured rate/distortion attribution

[The new attribution study](integer-coordinate-attribution.md) separates
CE/KL/H gradients at original and immutable epoch-2 coordinates, accounts for
all 3575 changed symbols, and compares the cold-best epoch-4 W on identical
text with mature/fresh PPMd priors. The controlled changes are
**+.00399618 / −.00301386 nats/target**. Data gradients dominate the current
soft-H gradients; all eight tiny rows retain every coordinate. The complete
single trajectory has now finished all 64 epochs; its original terminal
real mixed-coder stages continue. See the [terminal readout](#complete-64-epoch-terminal).


## Complete 64-epoch terminal

**2026-11-07.** The fixed method finishes **64/64 epochs**, **1904 updates**
and **66,846,016 target exposures**. The [complete compact curve and native
accumulators](results/integer-coordinate-terminal-64.json) preserve all epoch
results. Each epoch visits the same 1,044,469 training targets. All recorded
updates have 111 finite, nonzero gradient tensors; the 64 hard exports retain
the original scale bits and bounded −7…7 symbols.

The cold-held curve reaches its minimum at **epoch 4** (−.0044090752
nats/target). At epoch 64 the same two held portions average **+.0913543653**.
Meanwhile target-weighted online CE falls **.8621579448 → .4933005459**, and
teacher KL rises **0 → .1446452560**. This complete curve locates the late
regression in empirical cold-input fitting after the earlier scale failure
was repaired. The original causal-input and eight-block trajectories address
separate mechanisms and retain their frozen protocols.

| Original native scope | Targets | Original W NLL | W64 NLL | Change, nats/target |
|---|---:|---:|---:|---:|
| Cold prefix held | 523,194 | .4859066183 | .5507937874 | +.0648871691 |
| Cold q2 held | 516,391 | .9394297830 | 1.0576000266 | +.1181702436 |
| Independent mature tail | 1,041,397 | .9632509281 | 1.1871684597 | +.2239175317 |
| Same tail, fresh PPMd | 1,041,397 | 1.0032902250 | 1.2171125577 | +.2138223327 |
| Full cold q2, including training | 1,048,367 | 1.1102643653 | .8856565292 | −.2246078361 |

Original/candidate target counts and all piece boundaries match in each row.
On the identical mature/fresh tail, the benefit of PPMd history changes from
**.0400392969 to .0299440980 nats/target**; the interaction is
**+.0100951989**. W64 is **2,914,354 B**, growing **10,029 B per copy /
20,058 B across both copies**. Changed hard symbols total **2,036,366 /
5,868,864 (34.70%)**. The hard-entropy change predicts 20,121.865 B of ideal
dual-W growth, close to the actual static15 increase.

### Full-q2 score and exact learning roles

The original complete q2 codec sample contains these native targets:

- **521,332 training targets:** 104 complete pieces, bytes `[0,521436)`.
- **516,391 held targets:** 103 complete pieces, bytes `[531039,1047533)`.
- **10,644 remaining targets:** the crossing piece `[521436,531039)` and
  the EOF piece `[1047533,1048576)`.

The native `first_half`/`second_half` statistics split target offsets at byte
524,288; that split cuts a complete piece. The saved learning manifest supplies
the training/held roles. The complete held interval has exactly the same neural
resets and retained prior rows as its matching full-q2 subset. Subtracting its
loss accumulators from the full sample gives the **531,976-target
training-containing complement**: NLL **1.2760941034 → .7187503632**. The
full-q2 improvement and held regression therefore coexist on this actual W.

Training on the fixed Hutter corpus is part of the compression method. Its
benefit is charged against both stored W copies and the complete archive.
The original mixed audit keeps the mature-tail and full-q2 scopes separate;
a full-q2 transmission ratio describes this training-containing sample.
When its interpretation needs the held region, the existing trace interval
reader can select `[531039,1047533)` while replaying the preceding range
state and retaining probabilities from the original full online history. The preserved trace supplies this accounting directly.

### Resources and remaining measurement

The original training controller reports **89,684.526272 CPU s (24.9124 h)**
and peak **6,283,588 KiB RSS**. Its export/native descendants are included.
The separate mature-selection queue reports **298.649476 CPU s**, including
the earlier first-hard point and the final mature/fresh programs
(**205.056920 CPU s** for those final two).

The original sole mixed audit has selected **this exact terminal W64** and
started the paired v22/static15 round trips. This native-terminal report
preserves that selection. Actual payload bytes, guarded→mixer→u16
transmission and complete archive size follow the original audit.
**Stop new engineering for this fixed recipe; finish its queued measurement.**

Replay the compact published result with the standard library:

```sh
python3 research/check-integer-coordinate-terminal-64.py
```
