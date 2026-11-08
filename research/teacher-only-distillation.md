# Eight-block teacher-only control

This independent control keeps the eight-block architecture and native codec of
[the completed CE + KL study](depth-complete-32-mixed.md), changing the output
objective to **2 KL(teacher || student)**. Label cross-entropy is recorded with
detached logits. Final-hidden alignment and the double-weight rate term stay
fixed. The original trajectory and its cached targets are retained.

## Mechanism and fixed method

For each batch with `T` actual targets, the effective objective is

```
(2 * sum KL(teacher || student)
 + 0.1 * sum_t mean_192((student_hidden - teacher_hidden)^2)) / T
 + 2 * sum_tensor H_soft(z; tau=0.5) / 586459020
```

The original output term `CE(label) + KL(teacher || student)` has a stationary
probability target halfway between the teacher and the observed label. The
[complete cached-target assay](teacher-targets-and-label-oracle.md) measured
teacher divergence `0.162446955450` nats at this midpoint. At temperature one,
both output objectives have the same student-logit Hessian; the new control
isolates the target change while keeping its coefficient scale.

- Layers `[0, 1, 2, 3, 8, 9, 10, 11]`; 75 bounded quantized coordinates and
  59 retained raw F32 tensors. Signed BF16 weight/activation scales are fixed.
- Fresh initialization, fresh AdamW state, seed 419. Eight continuous epochs
  followed by 24 hard-coordinate epochs; learning rates and optimizer settings
  match the original study.
- Same 1,313 causal whole training pieces and 2,941,307 targets. Each fourth
  epoch exports a hard/static15 weight and runs native evaluation over the
  1,082,505 held targets, its 107,647-target long subset, and the independent
  131,071-target giant piece.
- Existing F32 teacher logits (205 columns) and final hidden states (192 columns)
  are read-only. Reuse checks their teacher checkpoint, canonical data manifest,
  and native-scaled CPU GEMM identity. A missing file stops the new trajectory.
  Student loss settings in the legacy cache metadata are kept as provenance.
- The actual backward and its scalar logger share `data_objective`. The default
  CE + KL recipe and default paths retain their previous identity.

## Admission and execution

`depth-distillation.py --teacher-only --teacher-cache OLD/training/teacher-cache`
uses the same required resource arguments as the original trainer, with a fresh
`--output`. `depth-training-job.py --job NEW --teacher-only --teacher-cache ...`
launches one independent complete trajectory after the real preflight receipt.
The admitted worker reuses its original incoming data, model checkpoint, CPU
kernel, recoder, and canonical-attention native family.

The bounded target is 32 epochs within 172,800 CPU seconds, including this
control's preparation, failures, and native evaluations. A budget terminal
retains its completed coverage and checkpoint. One selected positive-epoch,
byte-different trained weight receives its own paired prefix8/q2 measurement,
within 28,800 CPU seconds. Selection uses the same full held native proxy as the
original study; giant/held-long scores remain report-only.

`depth-mixed-audit.py --retained-build` retains the full deployed-native receipt
and charges zero new build CPU. Passed original W0 traces, encoded bytes, and
golden decodes are reused. The candidate encodes and decodes both samples with
its own frozen weight. Full-stream archive measurements are a later gate.

A self-contained check in the CPU/PyTorch environment is
`python3 research/check-teacher-only.py`; it loads zero models or datasets.
Passing `--cache`, `--checkpoint` and `--data` adds the real-cache identity check.
The minimal `check-teacher-only.py` imports the production helpers and checks
205-column logit VJP `2*(student_probability - teacher_probability)`, hidden MSE
VJP, detached CE, the reported scalar, and the actual read-only cache identity.
The real preflight additionally performs one cached whole-piece update,
verifies all 134 gradients, restores initialization and optimizer state, and
compares the zero-step static15 bytes with the retained original eight-block
export. The unchanged native family uses its existing passed admission.

## Current measurements

Admission passed on the retained 1,199-target causal piece `[0, 5321, 6521]`.
All 134 gradient tensors were finite and nonzero; the effective objective was
`1.3302973539272462`, with report-only CE `1.2836053767136677`, KL
`0.6251614708220392`, and hidden MSE `0.44593175994643974`. The restored
zero-step weight is byte-identical to the original eight-block export. New
preparation used `25.952396032` CPU seconds, with zero teacher resampling,
cache writes, or native builds. The independent complete trajectory and its
automatic single-weight mixed queue are running.

[Real admission](results/teacher-only-preflight.json) ·
[205-column objective/VJP check](results/teacher-only-objective-check.json)

The original complete CE + KL control increased the mature coded interval by
28,333 bytes; q2 excluding direct training input increased by 36,021 bytes.
The independent teacher-only trajectory's measurements will be reported with
its completed curve and the same selected weight's actual coded samples.

## First complete production epoch

The independent run completed its first full epoch: **1,313 pieces, 2,941,307
targets, 78 optimizer steps**. Pre-update, target-weighted means were CE
`1.261342768708` (report-only), teacher KL `0.283563684847`, final-hidden MSE
`0.273606457375`, and effective objective `0.629884257683`. All 134 gradients
were finite and nonzero at every recorded step; the read-only cache had zero
misses. The full [epoch record](results/teacher-only-first-epoch.json) includes
all 78 step records and the original epoch accumulator.

The fixed CE + KL control's first epoch had teacher KL `0.288198360923`. This is
an online-training comparison. The new 32-epoch curve, scheduled native held/long/
giant evaluations, and sole selected-weight coded samples remain in progress.

### The frozen residual is spread across articles

The completed CE + KL control's selected e12 worsened **457 of 473 held pieces**,
covering **1,080,647 of 1,082,505 targets**. Its three longest held pieces account
for 9.94% of targets and 11.58% of the total native loss increase; the remaining
470 pieces retain a `+0.175646600207` nats/target gap. All five length bands have
positive target-weighted residuals. The epoch record preserves these complete
control aggregates alongside the first new training observation. The new
control tests global teacher-target recovery with the fixed architecture.


## First native checkpoint: epoch 4

The original unique watcher reported the complete e4 native aggregate. Both
trajectories are at the same continuous-training epoch, with a hard-exported
weight for native scoring. Shared data, native family, scale, optimizer, hidden
alignment and rate settings retain the fixed method above.

| Native scope | Targets | Original CE+KL e4 | Teacher-only e4 | Fresh minus old |
| :--- | ---: | ---: | ---: | ---: |
| Full held | 1,082,505 | 1.187158519088 | 1.191965251350 | +0.004806732262 |
| Long held subset | 107,647 | 1.518069217219 | 1.516227199677 | −0.001842017541 |
| Independent giant | 131,071 | 0.186305475316 | 0.206609864589 | +0.020304389273 |

The primary full-held and independent-giant NLLs are higher in the fresh control;
the long subset's NLL is lower. All three fresh domains retain a positive gap to
the original 12-block W0: **+0.225716912556 / +0.257612898368 / +0.167623438458**
nats/target. The 470 held pieces outside the long subset have a derived
fresh-minus-old gap of **+0.005540908901** nats/target.

The long subset is included in the full-held count. Independent coverage here
is **1,213,576 targets**. Fresh e4 weighs **1,957,707 B**, versus **1,957,973 B**
for old e4: **532 B** improvement across the two weight copies. Its total
two-weight saving versus W0 is **1,893,236 B**; binary and payload components
retain their separate accounting.

The independent 32-epoch curve and its sole selected-weight mixed-codec
measurement continue. This e4 report uses the actual native aggregate; complete
raw checkpoint receipts will be collected together at the real terminal.
[Full same-epoch comparison](results/teacher-only-first-native-e4.json).


## Native checkpoint: epoch 8

The unique watcher has now returned the e8 native checkpoint, at the end of
the continuous stage. The matched CE+KL e8 is retained from its completed
original trajectory. Both are hard-exported weights scored by the same native
family and held-data contract.

| Native scope | Targets | Original CE+KL e8 | Teacher-only e8 | Fresh minus old |
| :--- | ---: | ---: | ---: | ---: |
| Full held | 1,082,505 | 1.168803883378 | 1.155372473714 | −0.013431409664 |
| Long held subset | 107,647 | 1.498125758357 | 1.478042330055 | −0.020083428301 |
| Independent giant | 131,071 | 0.224976626208 | 0.241003127283 | +0.016026501075 |

The new objective now has a lower NLL on the primary full-held scope and its
long subset. The derived other 470 held pieces also have a lower NLL. The
independent giant has a higher NLL. Against the original 12-block W0, fresh
e8 retains gaps of **+0.189124134920 / +0.219428028745 / +0.202016701152**
nats/target.

Within the fresh trajectory, e4 to e8 lowers full-held NLL by
**0.036592777636** and long-subset NLL by **0.038184869622**; giant NLL rises
by **0.034393262694**. Primary selection and independent-audit behavior are
reported on their own scored scopes.

Fresh e8 weighs **1,959,171 B**, against old e8's **1,960,160 B**. Its two
weight copies improve the weight charge by **1,978 B** versus old e8 and save
**1,890,308 B** versus W0. Within the fresh trajectory the asset grew by
**1,464 B** from e4. Actual mixed-codec bytes use the selected weight at the
complete trajectory's terminal; binary, payload and weight components remain
separately measured.

The complete 32-epoch run, its 24 hard-coordinate epochs, and the single
selected-weight paired codec measurement continue. The next native checkpoint
is e12. This report uses the actual watcher aggregate and the retained old
control; additional model/native runs and remote reads are zero.
[Complete e8 comparison and e4-to-e8 changes](results/teacher-only-native-e8.json).
