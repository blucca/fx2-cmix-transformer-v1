# One more positive activation bit, unchanged weights

The 12 MLPs use ReLU², so their down-projection activations are nonnegative.
The original quantizer uses only `[0,127]`. This fixed ablation uses `[0,254]`
and halves the activation scale and folded output scale, preserving the
physical clipping range and the original **2,930,652-byte weight file**.

## Fixed standalone screen: passed

Two 1 MiB slices of the exact prepared stream, fresh state per slice, original
205-token PPMd priors, article boundaries, 2^17-token pieces and KVI8 attention:

| Slice | Original NLL | u8 NLL | Improvement, nats/token | Original / u8 seconds |
|---|---:|---:|---:|---:|
| Prefix | 0.4637935360 | 0.4631379847 | 0.0006555513 | 76.029 / 76.474 |
| Middle, offset 293,232,482 | 1.1102643653 | 1.1093539076 | 0.0009104577 | 88.306 / 89.090 |

Across **2,095,604 predicted tokens**, weighted improvement is **0.0007830732
nats/token**, with **0.75%** more measured evaluation time. Both halves of each
slice improve. Nonzero down activations increase about 11.5%. The frozen screen
required both slice improvements, at least 0.0005 pooled, and at most 5% more time.
The full screen, compilation and smoke test took 370 seconds on one pinned CPU.
[Exact measurements and frozen protocol](results/positive-activation-A.json).

## Real coder screen: 227 B saved; continuation gate missed

Fresh paired trials used the same machine and one pinned CPU. The prefix ran
original then u8; the second slice reversed that order. Each variant completed
real encoding and decoding, and both original streams match their saved
goldens byte for byte.

| Slice | Original bytes | u8 bytes | Saved bytes | Original / u8 coding seconds |
|---|---:|---:|---:|---:|
| Prefix, 1 MiB | 84,839 | 84,763 | 76 | 691.210 / 705.345 |
| Middle, 1 MiB | 208,696 | 208,545 | 151 | 744.624 / 773.476 |
| Total | 293,535 | 293,308 | **227** | **1,435.834 / 1,478.821** |

The verified saving is **0.000866 bits/input-byte**, with **2.99%** more two-way
coding time. The frozen screen required nonnegative savings on each slice,
**at least 263 B combined**, at most 5% extra coding time, exact restoration,
and at most 4 KiB additional double-binary packaging. The size gate misses by
36 B; all other gates pass. **Stop this fixed recipe before long confirmation.**
The 227 B improvement remains a verified result, and the switch stays disabled
by default. [Exact paired measurements and protocol](results/positive-activation-B.json).

The development packaging check already measures **202,304 → 202,316 B** with
the same UPX 5.2.1 `-9`: **+24 B across the two executable copies**. Neural
weights are unchanged. The official full-file score also includes the payload
and all required side information.

## Implementation

`FX2_POSITIVE_U8` defaults to zero. The packed sparse4 kernel decodes nibbles
back to signed weights, uses activations as the unsigned `maddubs` operand,
and widens signed accumulators every nine pairs (`9×2×254×7=32,004`). Both the
sparse and dense fallback use this path. The shared PPMd prior retains its
original scale and range. Reference glue uses unsigned packing as well.

The focused scalar check covers 0/127/128/254, ±7, odd tails and both sides of
the 9/18-pair boundaries, all six sparse4 entry points, folded scales, and
nearest-even ties in both directions. A 64 KiB model check reproduces the old
objects' complete probability fingerprint and NLL with the default disabled.

## Reproduce

Use the source/data preparation in [the research guide](README.md). Keep all
generated files in a scratch directory. Build baseline with `=0` and candidate
with `=1`; statistics are research-only and omitted from the actual coder.

```sh
python3 research/build.py --source . --out "$SCRATCH/build/u8" \
  --define FX2_POSITIVE_U8=1 --define FX2_ACTIVATION_STATS=1 \
  --extra research/activation-probe.cpp --strict-extra

# Repeat the build with FX2_POSITIVE_U8=0 into build/baseline.
"$SCRATCH/build/baseline/activation-probe" --self-test
"$SCRATCH/build/baseline/activation-probe" capture \
  "$DATA/stream.bin" "$SCRATCH/tokens.u8" "$SCRATCH/priors.f16" \
  "$DATA/vocab.bin" 0 1048576
"$SCRATCH/build/baseline/activation-probe" eval \
  "$SCRATCH/tokens.u8" "$SCRATCH/priors.f16" models/6m-q4-fp32.tfwc2
"$SCRATCH/build/u8/activation-probe" eval \
  "$SCRATCH/tokens.u8" "$SCRATCH/priors.f16" models/6m-q4-fp32.tfwc2
```

Repeat capture at offset 293232482 with fresh output names; evaluate u8 first
on that second slice. Priors occupy 410 bytes per input token and are streamed
during evaluation. The research main uses strict double NLL; production model
compiler flags and the original float16 conversion path are preserved.
