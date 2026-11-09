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

## Continuous recurrent history

The [CPU adjoint](cpu-kda.cpp) and the [native KDA state](../cpp_infer/src/opt/kda.h)
store the recurrent matrix **S in FP32**, with FP32 state writes at each step.
The CPU's 128-token checkpoints organize reverse-mode storage and recomputation;
the state and adjoint remain continuous over the complete input piece. Current
training pieces contain at most 32,768 targets. Native evaluation starts each
explicit piece with `begin_article(0)` and keeps its state through the piece,
including the independent 131,071-target audit.

For fixed one-step inputs, the state equation is

```
S_t = M_t S_(t-1) + beta_t * k_t * v_t^T
M_t = (I - beta_t * k_t * k_t^T) D_t
```

`k_t` is L2-normalized with epsilon `1e-6`, `beta_t` is sigmoid-bounded, and
`D_t` contains per-key-row forget decays. In exact arithmetic,
`||M_t||_2 <= 1`; a historical state component follows the ordered product
`M_t ... M_(s+1)`. Output RMS normalization and the sigmoid output gate act on
the readout. The learned inputs, keys, decays, and readout over longer histories
are measured by the separate long native domains.

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

## Complete 32-epoch result

Both fixed trajectories completed **94,121,824 training targets and 2,524
optimizer steps**: eight continuous epochs followed by 24 hard-coordinate
epochs. The cached teacher remained read-only, with zero cache misses.
Admission restored the same initial eight-block bytes and verified all 134
finite, nonzero gradient tensors. Preparation used 25.952396032 CPU seconds;
the complete teacher-only training/evaluation trajectory used
93,649.449389059 CPU seconds.

### Native checkpoints on the same held inputs

The primary held domain has **1,082,505 targets / 473 pieces**. Its long subset
contains 107,647 targets; the separate giant audit has 131,071 targets. Native
NLL is measured from FP32 logits before production half/guard/mixer/u16/range.

| Epoch | CE + KL full held | Teacher-only full held | Teacher-only held long | Teacher-only giant | Teacher-only W, B |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 4 | 1.187158519 | 1.191965251 | 1.516227200 | 0.206609865 | 1,957,707 |
| 8 | 1.168803883 | 1.155372474 | 1.478042330 | 0.241003127 | 1,959,171 |
| 12 | 1.145141393 | 1.122412620 | 1.440675849 | 0.233062518 | 1,959,728 |
| 16 | 1.149255266 | 1.123169184 | 1.439897894 | 0.258788764 | 1,960,239 |
| 20 | 1.158834673 | 1.124225593 | 1.441486694 | 0.273788844 | 1,960,779 |
| 24 | 1.160351253 | 1.125847485 | 1.444504741 | 0.286336036 | 1,961,211 |
| 28 | 1.163296659 | 1.123333938 | 1.438905642 | 0.311845858 | 1,961,646 |
| 32 | 1.167243559 | 1.122611285 | 1.438649551 | 0.320578531 | 1,962,097 |

The frozen selector chose **e12 / step 949 / 1,959,728 B**, using the primary
held proxy and earliest-epoch tie-breaker. Against the matched CE + KL e12,
primary NLL improves by **0.022728773016 nats/target**; the W0 residual remains
**+0.156164280798**. The selected long/giant residuals are
+0.182061548110 / +0.194076092023. Both weight copies save **1,889,194 B**
against W0; both deployed binaries save **9,024 B**.

From e12 to e32, online teacher KL falls **0.009755372969**, while the exported
primary NLL rises **0.000198665509** and giant NLL rises **0.087516013051**.
The matched CE + KL late primary increase is **0.022102166426**. Teacher-only
learning reduces the late primary rebound; selection still favors e12.
Online training means are pre-update and target-weighted over the fixed
1,313 training pieces. Each native checkpoint has its own hard export.

[Complete paired curves](results/teacher-only-complete-curves.json) ·
[Raw-receipt summarizer](summarize-teacher-only-curves.py)

The summarizer rebuilds all 64 epoch means from 5,048 optimizer records,
checks the fresh trajectory's 16 native logs, and pairs the eight retained
native checkpoints of each trajectory. Run it with the two collected raw
directories: `python3 research/summarize-teacher-only-curves.py OLD/raw NEW/raw`.

### The same e12 through the actual codec

Both complete sample encodes and their golden-decode restorations passed.
The full prefix8 payload grows **150,888 B** and the full q2 payload grows
**33,157 B**. The table uses retained stream histories and actual closed-range
interval endpoints; each reporting scope is paired separately.

| Scope | Input B | NN-target B | Guarded NN extra bits | Mixer extra bits | u16 extra bits | Actual coded increase B |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mature prefix8 | 1,041,690 | 1,041,397 | 269,332.092 | 208,979.785 | 208,129.467 | 26,016 |
| Full q2 | 1,048,576 | 1,048,367 | 309,941.384 | 266,018.652 | 265,262.506 | 33,157 |
| q2 outside direct training input | 1,037,391 | 1,037,183 | 307,717.763 | 264,189.035 | 263,451.842 | 32,931 |
| q2 seed414 held role | 505,309 | 505,207 | 129,953.147 | 101,574.093 | 101,215.967 | 12,651 |

The mature and non-direct q2 coded increases improve by **2,317 B / 3,090 B**
against the complete CE + KL control, recovering **8.18% / 8.58%** of its
increases. Mature guarded-NN recovery is **12.62%**; its final/NN transfer
changes from 0.735351 to **0.772761**. The exact NN/fallback masks, override
counts, u16 ideal lengths, range residuals, and baseline reuse are retained in
the [complete same-weight readout](results/teacher-only-complete-mixed.json).

Measured package scope: the two full sample payloads grow **184,045 B**;
including their shared double-W and dual-B savings gives **1,714,173 B saved**
for this two-sample component. For full-stream admission, extending each
observed interval's byte-rate across the **586,459,020-byte prepared input**
gives net archive increases of **12.748 MB / 16.718 MB / 12.784 MB** after the
1,898,218 B weight/binary saving. These are constant-rate admission scenarios.
The completed candidate stops new full-stream expansion; its measured curves,
round trips and four-layer attribution remain reusable research assets.

The original same-W codec used 5,583.029095088 CPU seconds. Four additional
candidate reducers used 0.449368 CPU seconds; baseline reducers and native
model/codec reruns were zero. The original controller state is retained.

Earlier checkpoint records remain available:
[e4](results/teacher-only-first-native-e4.json) ·
[e8](results/teacher-only-native-e8.json) ·
[e12](results/teacher-only-native-e12.json) ·
[e16](results/teacher-only-native-e16.json) ·
[e20](results/teacher-only-native-e20.json) ·
[e24](results/teacher-only-native-e24.json).
