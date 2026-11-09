# Keep all twelve blocks; learn a seven-symbol weight grid

This experiment keeps the original twelve-block, 192-wide model: nine KDA blocks,
three attention blocks, and all six skip connections. It reduces the matrix
alphabet from fifteen symbols to seven and trains against the original teacher's
complete output distribution.

## One fixed method

For each of the 111 quantized matrices, initialize its integer coordinates with

```
q7 = sign(q15) * floor((3 * abs(q15) + 3) / 7)
s7 = BF16_RNE(s15 * 7 / 3)
```

The signed BF16 scale conversion uses the exact integer algorithm from the
[measured quant7 recoder](weights-quant7.md). All weight and activation scales,
raw F32/BF16 parameters, layer positions, and skip connections stay fixed.
Only 5,868,864 external matrix coordinates are optimized. Their feasible interval
is `[-3, 3]`; the optimizer projects them into it after every update. Within this
interval, the existing hard integer forward and export paths produce the required
seven symbols.

The loss is

```
2 * KL(teacher || student) + 2 * H_soft / 586459020
```

Cross-entropy is measured on detached logits. The label contributes zero to the
optimization gradient. Hidden-state MSE has coefficient zero. Soft entropy uses
seven centers and temperature 0.5; the existing fifteen-bin hard histogram has
eight zero-frequency bins and the same entropy as a seven-bin histogram.

The fixed run uses seed 414, AdamW with learning rate 0.01, betas `(0.9, 0.99)`,
epsilon `1e-8`, and zero weight decay. It targets 32 complete epochs over the
existing causal whole-piece manifest: 1,313 pieces and 2,941,307 targets per epoch.
Every fourth epoch is exported and measured with the original full12 native
runtime. The primary native holdout has 1,082,505 targets; its three long pieces
and the separate 131,071-target giant audit are reported separately.

## Reuse and identity

The frozen teacher cache contains F32 logits and final hidden states for every
complete training piece. This method reads only its logits. Its target identity
is the original checkpoint, canonical data manifest, and integer-before-BF16-fold
CPU math. The cache keeps its original provenance; the new recipe and trajectory
have their own identity. A missing cache file stops the run.

CPU computation uses `native_scaled_gemm=True`. The converted scale is written
into both the captured quantizer scale and its actual model parameter before the
new frozen snapshot is taken. Forward computation and export consequently use
the same scale.

The output has 429 payload tensors and 434 stored/derived tensors. The existing
full12 static15 loader and original v22 codec accept the seven-symbol subset.
The initial export must match the existing `--quant7` recoder byte for byte.
A one-piece VJP check restores the initial coordinates, empty optimizer, and RNG
before the full training resumes.

## Measurement and delivery

[Trainer](integer-coordinate-qat.py) selects `--recipe q7-teacher-only-v1`.
[Serial job](q7-training-job.py) runs the preflight and resumes that restored
initialization. [The helper](q7_teacher_only.py) owns the signed BF16 mapping,
output objective, and read-only target identity.

The [mixed audit](depth-mixed-audit.py) selects `--q7-teacher-only --retained-build`.
It reuses the original full12 baseline codec results and measures exactly one
positive-epoch hard export that differs from the initialization. Selection uses
the primary native holdout proxy minus the charge for both weight copies, with
the earliest epoch winning a tie. A run whose completed hard exports all match
the initialization retains the original delivery baseline.

The selected weight is used for both continuous prefix8 and cold article-q2
encode/decode, with its own trace. Native quality, guarded neural probabilities,
mixer probabilities, u16 probabilities, and closed range-code bytes retain their
respective measurement scopes. Completed epochs and CPU-bounded coverage are
reported with the actual run.

## Small reproducible check

```
python3 -B research/check-q7-teacher-only.py
# With PyTorch installed, also exercise the actual tensor map and toy VJP:
python3 -B research/check-q7-teacher-only.py --torch
```

The standard-library check covers all fifteen source symbols, nine signed/tiny
BF16 fixtures against an independent rational RNE oracle, projected symbols,
hard entropy equivalence, a 205-way KL gradient finite difference, target identity,
and the missing-cache path. The actual cached-piece and complete-run results are
published as they finish.

## First measured cached-piece result

The [actual preflight](results/q7-teacher-only-preflight.json) completed 1,199
targets with finite, nonzero gradients in all 111 coordinate tensors. The first
update changed 5,443,182 floating coordinates while retaining every hard symbol
and the 2,044,814-byte initial archive. The archive matches the existing quant7
reference byte for byte, including all 434 stored/derived tensors.

Initial teacher KL was 0.365677859805 nats/target; detached CE was
1.036659272538. The effective objective was 0.768070476218. Initialization,
optimizer, and RNG were restored, with zero teacher cache misses. The complete
32-epoch run has started; native curves and its selected-weight codec results
follow the original measurement sequence.

## Offline complete-or-bounded readout

[The readout](q7-terminal-readout.py) consumes the small originals collected at
the mixed queue's actual terminal. It checks the q7 recipe against admission,
reconstructs the target-weighted `2KL + 2H/N` objective, and reports every online
epoch as hard. Completed epochs, a partial final epoch, and native evaluations
retain their actual coverage. The existing full12 native reference is reused;
the q7 initialization has its own weight identity and an unmeasured native scope
in this run.

The output includes the full-held, contained long-three, remaining short-piece,
and independent giant curves; the selected and final evaluated epochs are named
separately. The selected weight's own codec pairs and fixed-scope trace attribution
are joined by their recorded weight identity. A retained-original outcome stops
the candidate-attribution path. The training CPU ledger already includes its
preparation carry; the codec and four-reader ledger are reported separately.

```sh
python3 -B research/q7-terminal-readout.py --check
# After the original queue's terminal collection and its same-W attribution:
python3 -B research/q7-terminal-readout.py \
  --collection /path/to/q7-small-originals \
  --attribution /path/to/q7-attribution/attribution-readout.json
```

This readout writes `terminal-readout.json` inside the supplied collection.
Running it without `--attribution` preserves the native curve and labels the
selected-weight attribution as pending. It makes zero network or model calls.

The [initialized rate diagnostic](q7-rate-initialization.md) reconstructs the
soft entropy, static4096 payload cost and local hard-step direction from the
existing exact histogram, with the admitted training recipe unchanged.

## First native holdout: hard epoch four

The original watcher reported [epoch four](results/q7-native-e4.json) on
2026-11-08 at 16:26 UTC. Training is continuing toward the reserved 32 epochs;
the selected-weight codec queue is waiting for its training terminal.

| Native scope | Targets | Original W0 | Full12 q15 CE+KL e4 | Full12 q7 teacher-only e4 |
|---|---:|---:|---:|---:|
| Primary full-held | 1,082,505 | 0.966248339 | 0.973713877 | 1.514144681 |
| Contained long-three | 107,647 | 1.258614301 | 1.267215315 | 1.881413753 |
| Independent giant | 131,071 | 0.038986426 | 0.041370920 | 1.301486586 |

Values are raw native nats/target. The long-three belongs to the primary scope
and is counted there once. Target-weighted subtraction gives the remaining 470
pieces / 974,858 targets: q7 NLL 1.473589632, versus W0 0.933964336 and the older
q15 e4 value 0.941304491.

The old control uses the same twelve blocks, causal split, seed, hard phase from
its first update, 111 trainable matrices and frozen raw parameters.
This is a whole-method comparison: alphabet/scale initialization, the output
objective and CPU forward math change. The old student/teacher use scaled FP32
linear products; q7 and its cached teacher use integer-unit products followed
by the native BF16-scale fold. Native reports share the original full12 family.
Relative to that old e4, q7 adds 0.540430804 primary,
0.614198438 contained-long and 1.260115665 independent-giant nats/target.
The [separate e0 control](#training-relative-to-the-initial-quantized-model)
now supplies the initialization's complete native domains. The admitted
1,199-target training-piece result retains its own measurement scope.

The e4 q7 archive is 2,073,263 bytes, 28,449 above its initialization. It saves
1,662,124 bytes across both copies against W0. For the original kappa=1 native
selection proxy, that saving allows 0.015722503 nats/target; the measured primary
residual is 0.547896342. The current proxy is therefore −56,259,419 bytes.
This proxy guides selection; the selected weight's own G→M→u16 probabilities
and closed codec bytes are measured after the complete trajectory.

The reserved complete32 and sole selected-weight mixed audit continue. The next
scientific input is a new e8 native aggregation or the original queue's actual
terminal/error. Full raw originals are collected once at that terminal.

```sh
python3 -B research/check-q7-native-stage.py
```

The small check verifies the public old control, contained-scope subtraction,
both weight copies and the native selector arithmetic. New model, native and
codec calls for this readout: zero.

[The coordinate-step control](q7-coordinate-step.md) describes how the exact
signed-BF16 scale map changes pre-round Adam displacement in weight units.
Its scalar, shared-gradient-history scope is reported separately from the
actual complete training trajectory.

## Integer-unit arithmetic certificate

The [normal-BF16/FP32 certificate](integer-unit-exactness.md) checks hard STE
values, scale recovery and the exact integer reduction bound. For this seven-
symbol grid, every 768-term INT8 dot has an absolute partial-sum bound of
294,912. The fold is a FP32 product of two BF16-derived scales. Native curves
and the selected weight's own codec retain their end-to-end measurement scope.


The [KDA energy certificate and conditioned-key control](kda-state-energy.md)
separate fixed-control state propagation from the key/decay/value forcing of a
changed model. A sharp energy bound and a two-key value-fitting example connect
long histories to state scale; the actual q7 native and same-weight results
continue on the fixed measurement path.

## Native checkpoint: hard epoch eight

The original watcher cached [hard e8](results/q7-native-e8.json) at
2026-11-08 21:27 UTC. The reserved complete32/48CPUh trajectory and sole
selected-weight mixed audit within 8CPUh continue.

| Native scope | Targets | W0 | Old full12 q15 causal e8 | Q7 e8 | Q7 e8 minus e4 |
|---|---:|---:|---:|---:|---:|
| Primary full-held | 1,082,505 | 0.966248339 | 0.995284567 | 1.281830556 | -0.232314125 |
| Contained long-three | 107,647 | 1.258614301 | 1.293673241 | 1.632676920 | -0.248736834 |
| Independent giant | 131,071 | 0.038986426 | 0.047069149 | 0.897504535 | -0.403982050 |

Values are raw native nats/target. Long-three is counted inside full-held;
independent coverage is 1,213,576 targets. The remaining 470 pieces / 974,858
targets have q7 NLL 1.243088956, versus W0
0.933964336 and old q15 e8 0.962335517; q7 e8 minus e4 is
-0.230500676. Relative to old e8, q7 adds
0.286545988 primary, 0.339003678 contained-long and
0.850435386 independent-giant nats/target. The old CE+KL
control uses scaled FP32 GEMM for student and teacher; q7 and its cached teacher
use integer-unit products followed by the BF16-scale fold. This remains the
whole-method comparison on the same native family and causal split.

Q7 e8 weighs 2,073,147 bytes: 116 fewer than e4, 28,333 above its
initialization, and 1,662,356 saved across both copies against W0. The original
kappa=1 selector allows 0.015724698 nats/target; the observed primary residual
is 0.315582217, giving proxy -31,699,810.61 bytes. The [separate e0 control](#training-relative-to-the-initial-quantized-model)
now measures the initialization's native scope. Same-weight probabilities and closed coded bytes follow
the original terminal audit; next input is new e12/native or actual terminal/error.
This local report adds zero network, model, native, collector or codec calls.

```sh
python3 -B research/check-q7-native-stage.py research/results/q7-native-e8.json
```


## Training relative to the initial quantized model

On 2026-11-09, a [complete e0 control](results/q7-initial-native.json) measured
the preserved 2,044,814-byte initialization with the same full12 evaluator and
held inputs. The initial file matched the original quant7 reference byte for
byte. Two native calls covered 1,213,576 independent targets in 103.131388 CPU
seconds, with a peak child RSS of 94,212 KiB. The original W0 results were reused.

Let `Q = N(e0) - N(W0)` denote initial quantization loss, and
`T_e = N(e) - N(e0)` denote the change from learning. Their sum is the measured
residual `N(e) - N(W0)`. All values below are native nats/target.

| Scope | Q: initial quantization | T4: learning change | T8: learning change | Total e8 residual |
|---|---:|---:|---:|---:|
| Primary full-held | +0.361833348 | +0.186062994 | -0.046251131 | +0.315582217 |
| Contained long-three | +0.410441185 | +0.212358267 | -0.036378567 | +0.374062618 |
| Independent giant | +0.247456043 | +1.015044116 | +0.611062066 | +0.858518109 |

At e8, learning has recovered 12.7824% of the primary initial loss and 8.8633%
of the contained-long loss. The giant's loss grows by 0.611062066 from the
initialization. The initial quantization raises loss on all 473 primary pieces.
These observations separate the initial grid distortion from the trajectory's
different domain responses. The primary holdout and same-weight byte audit
continue to govern selection; the long and giant scopes retain their report-only
role. The original complete32 trajectory continues unchanged.

[The exact e0 measurement source](q7-initial-native.py) reads the preserved
initial weight and original manifest; its two `eval-pieces` commands leave the
training files intact. The [terminal reader](q7-terminal-readout.py) now adds
`delta_nll_q7_e0`, an e0 paired-piece column, and the separate control CPU
charge to the original complete/bounded readout. Its local check uses these
actual e0/e8 data and the original admitted scalar:

```sh
python3 -B research/q7-terminal-readout.py --check
```


## Latest stage aggregate: hard epoch sixteen

The original unique watcher returned e16 on 9 November 2026 at 16:27 CST.
Full held / contained long / independent giant NLL falls
**0.041138109 / 0.050443388 / 0.060530037 nats/target** from e12.
The primary initialization loss has recovered **46.1159%**; its W0 residual is
**+0.194970680**. The hard export is **2,076,739 B**, saving **1,655,172 B**
across both weight copies. This [stage readout](results/q7-native-e16-stage.json)
uses the cached native aggregate; the complete raw curve and sole selected-W
codec measurement follow the original training terminal.
