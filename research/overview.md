# Research overview: prediction loss to complete archive size

Maintained by Blucca / Codex, this research builds on **Vladimir Ivanov’s** published 6M transformer compressor and **Kaido Orav / Ibrahim Marcouch’s** statistical experts. Original authorship and licenses are preserved. The production reference retains the upstream weights, dictionary, article order, PPMd and outer mixer. The research contribution is a complete measurement and implementation path: **native loss → same-weight guarded NN/mixer/u16 → actual range bytes → complete two-copy archive**.

## The causal story

The learned predictor supplies substantial value beyond causal PPMd. On the existing held inputs, the frozen NN improves every one of **473 pieces**, by **0.589101 nats/target** overall; the independent giant-article audit improves by **1.179914 nats/target**. [Measured prior value](causal-prior-value.md) establishes the reference that weight compression and depth reduction must preserve.

A same-text cold-versus-mature PPMd control exposed a training-input mismatch. [Whole-stream causal priors](causal-priors.md) address that mechanism with one traversal of the prepared stream, article-cluster sampling and explicit complete-piece resets. These inputs support both the original causal weight study and the eight-block student.

For deployment measurement, the **identical selected hard weight file** enters the retained v22 codec. Guarded NN probabilities influence neural bit inputs, FXCM and its argmax contexts; online mixing, SSE and final u16 frequencies determine the range-coded bytes. [The mixed runner](causal-mixed-audit.py) and [strict interval reader](mixed-interval.cpp) align neural targets and fallback bytes while reporting native loss, guarded NN loss, mixer output, u16 loss and actual encoded length separately. The mature interval retains its continuous-prefix history; fresh q2 is a separate, training-containing sample.

## Published coverage — 7 November 2026

| Method | Published scientific result | Actual-code measurement |
|---|---|---|
| [Fixed-scale integer coordinates](integer-coordinate-qat.md#complete-64-epoch-terminal) | All **64 epochs** completed. Online CE fell while mature native NLL increased **0.223917532 nats/target**; both weight copies grew **20,058 B**. | The [complete same-W mixed audit](results/integer-coordinate-mixed-terminal-64.json) failed its gate: mature code grew **20,835 B**, independent q2 held code **7,085 B**. This fixed recipe stops. |
| [Original causal-prior study](causal-priors.md) | The original **16-epoch** method is implemented end to end. E2 leads the published eleven-epoch native curve, with held change **−0.001283234 nats/target**. | Its original selected-same-W mixed readout is pending. |
| [Eight-block causal distillation](depth-distillation.md) | The original **32-epoch** trajectory continues. E12 is the current native minimum, recovering **74.33%** of the starting held mapping gap; its remaining W0 gap is **+0.1788930538 nats/target**. | Eight-block actual range-code measurements: **0**. Its one terminal-selected trained-weight mixed measurement follows the trajectory. |

[Zero8](depth-zero-native.md) is the native eight-block initialization control. The separate maximum-article audit gives long-history behavior its own readout alongside the randomized held mean.

## Reusable capabilities

- **Full-piece CPU adjoints:** [method and measurements](cpu-backward.md), [Python operators](cpu_ops.py), and [KDA kernel](cpu-kda.cpp) retain full-piece gradient flow through KDA, attention and QAT.
- **Causal inputs, sampling and resets:** [input contract](causal-priors.md), [sample planner](causal-sample-plan.py), and [capture source](causal-ppmd.cpp) preserve all 205 prior columns and explicit piece boundaries.
- **Native eight-block export:** [student/export source](depth_student.py) and [native-family builder](depth-native-build.py) deliver the genuine 294-tensor inference file and accepted eight-block arithmetic described in [the method](depth-distillation.md).
- **Lossless static15 weights:** [format and receipt](weights-static15.md), [recoder](weights-recode.cpp). All 434 original tensor payloads match; measured two-copy development-package savings are **52,022 B**, including the decoder charge.
- **Official-input inverse:** [completed provenance](prepared-original-provenance.md) and [prepare adapter](prepare.cpp) restore the official **1,000,000,000-byte** input through the original transform and sorting helpers.

## Formal candidate archive measurements

Receive the causal method’s original mixed readout; finish the depth trajectory and its sole selected-weight mixed measurement. Their actual payload and component costs supply full-stream admission. The [complete archive runner](causal-fullstream.md) executes native `cmix -e` for W0 and the admitted W1, reads both authentic packages, then verifies each archive’s official-original restoration.

The score is **`S = P + 2B + 2D + O + 2W + 69`**: range payload plus both binary, dictionary and weight copies, one article-order asset and fixed headers. [Original native auxiliary delivery](original-release-native-aux.md) already verifies the real package sections. Candidate S, paired full-original restoration, and whole-program CPU, memory and active-time qualification are the remaining end-to-end measurements.


- [Native observer values and CPU shadow gradients](native-observer-training-contract.md):
  three-layer forward contract, actual-index scalar loss and a standard-library
  replay of the saved observations and labelled FP32 controls.
