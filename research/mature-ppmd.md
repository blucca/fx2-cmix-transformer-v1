# Mature PPMd priors and online mixed accounting

The integer-coordinate trajectory retains its original two cold training
slices. This independent measurement uses the final complete articles inside
a continuous 8 MiB prefix, with **7,344,758 bytes of causal PPMd history**.
The NN resets at its original article separators and 2^17-token cuts.

## Capture only the retained rows

`activation-probe` accepts an optional absolute `WARM_START`:

```sh
activation-probe capture STREAM TOKENS PRIORS VOCAB OFFSET LENGTH [WARM_START]
# This measurement: OFFSET=7344758 LENGTH=1041690 WARM_START=0
```

PPMd learns every byte from WARM_START through OFFSET+LENGTH. Only rows
[OFFSET,OFFSET+LENGTH) reach the prior file. Row i follows byte i and feeds
the NN's prediction of byte i+1; each piece's first token uses the coder's
PPMd fallback. Default capture starts at OFFSET, preserving the cold path.

The output is **1,041,690 tokens / 427,092,900 B of priors**, covering
**293 complete pieces / 1,041,397 NN targets**. The original PPMd maps a
14,000 MiB sparse file; the measurement allows 18 GiB virtual address space.
Observed mature capture peak RSS is **1,330,248 KiB**, and native NN RSS
**94,196 KiB**. RAM and virtual mapping size are recorded separately.

The existing native evaluator and original static15 weights give:

| Same text and NN resets | Raw NLL, nats/target |
|---|---:|
| PPMd with stream[0,7344758) history | 0.9632509280782524 |
| Fresh PPMd at offset 7344758 | 1.0032902249810889 |

The initialization intervention improves the original NN score by
**0.040039296902836474 nats/target**. This is a baseline-state measurement.
QAT's new weight and payload gains are measured on its committed exports.
The original training manifest, model weights and inference kernels stay fixed.
Preparation used **128.609377 CPU seconds**, including build, retained-row
check, capture and native baseline; the same-text control used **100.221161 s**.

```sh
python3 research/mature-ppmd-check.py PROBE STREAM VOCAB TEMP_OUTPUT
```

This minimal check compares the retained 32 KiB against one complete 64 KiB
capture byte for byte, including all 205 f16 columns and a rejected warm start.
The actual preparation also matches the original cold prefix's first 64 KiB.

## Measure actual online mixed propagation

A single original v22 trace probe is relinked with the already verified
static15 loader. All original NN, coder, mixer and main objects are retained.
The reference relink is byte-identical to the original probe. Both W0 and W1
use this common executable. The pair comprises one continuous 8 MiB prefix
and the existing cold q2 1 MiB sample; baseline goldens are **1,072,279 B**
and **208,769 B**. Candidate encodes are followed by actual decode/byte checks.
Heavy mixed runs follow the training process's exit on the same free machine.

```sh
c++ -O3 -std=c++17 -fno-fast-math -ffp-contract=off \
  research/mixed-interval.cpp -o TEMP/mixed-interval
python3 research/mixed-interval-check.py TEMP/mixed-interval TEMP/check
TEMP/mixed-interval TRACE TOKENS VOCAB SCORE_BEGIN ENCODED_BYTES
```

The streaming reducer verifies the whole exact range byte count and every
selected source byte. It separates all bytes from the NN-target subset, then
reports guarded NN bit loss, pre-SSE mixer loss, final u16 loss and actual
continuous range-byte increments. Vocabulary overrides use their actual
final probability in the pre-SSE accounting. A single eight-byte fixture
checks a 9 B full code, 4 B interval and 3 neural targets.

Final packaging uses `S = payload + 2B + 2D + O + 2W + 69`. Integer-coordinate
exports use the same static15 decoder, so their incremental inference code
cost is zero against W0/static15. Original v22 non-PGO/UPX5.2.1 development
binaries measure **181,644 → 181,956 B**: the initial lossless static15
component saves **52,030 B** after the double-binary cost. Formal full-stream
S, PGO resources and prize eligibility have their separate full-run result.

[Machine-readable baseline and receipts](results/integer-coordinate-mature-ppmd.json).
