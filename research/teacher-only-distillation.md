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
