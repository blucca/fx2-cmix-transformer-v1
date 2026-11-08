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
