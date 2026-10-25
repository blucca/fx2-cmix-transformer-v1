# Seven-value weights: a fixed rate/distortion screen

The lossless [static15 container](weights-static15.md) also permits an explicit
model-compression experiment. `--quant7` maps the original 15 weight values
to seven and rescales their paired rows. The default recoder preserves the
original model. Text compression and decompression use the same selected
model and retain the exact input bytes.

## Fixed transformation

For all 111 INT8 weight matrices, use
`q7 = sign(q) * floor((3 * abs(q) + 3) / 7)`.
Multiply each paired `.weight.scale` by `7/3`, rounding to nearest-even BF16.
The implementation performs the rational rounding in integers, including
mantissa ties and exponent carry. All 25,946 row scales are finite, nonzero
normal BF16 values; **16 are negative**, and their signs are preserved.
The 119 activation scales, other tensors, names, shapes and order stay unchanged.

The recoder reloads the output and checks all **434 tensors** against this
transformation. A default-mode run reproduces the previous static15 file
byte for byte. The v3 decoder and inference kernels require no changes.

| Weight file | Bytes |
|---|---:|
| Original 15-value v2 | 2,930,652 |
| Same-value static15 v3 | 2,904,325 |
| Fixed seven-value v3 | **2,044,814** |

The seven-value file saves **885,838 B per copy** against the original.
[Transformation and budget receipt](results/weights-quant7.json).

## Predeclared real-coder gate

The official package has two weight and executable copies. Under the
existing development-executable comparison and unchanged dictionary/order:

```
2 * 885,838                 two-copy weight saving
- 41,952                   v26 + v3 executable increase versus v22
- 1,004,247                current minimum record improvement
= 725,477 B                available full-payload regression budget
```

This screen reserves zero credit for v26's payload improvement. Spread over
the fixed 586,459,020-byte prepared stream, the budget is **1,297 B per MiB**.
The first sample is the original `article-q2-1m`: offset **293,232,482**,
length **1,048,576**, original v26 code **208,696 B**. The gate is a verified
new code of **209,993 B or less**. Both encode and decode run, with direct
comparison to the original slice, within 1,200 CPU seconds.

## Real-coder result: stop this fixed mapping

The complete q2 round trip produces **252,814 B**, up **44,118 B / 21.14%**
from the original v26 code. The regression is **34.01 times** the 1,297-byte
budget. Decoding restores the original 1 MiB slice byte for byte.

| Phase | Wall seconds | CPU seconds | Coding seconds | Peak RSS, KiB |
|---|---:|---:|---:|---:|
| Encode | 451.203 | 434.394 | 403.186 | 7,889,032 |
| Decode | 486.844 | 440.234 | 439.561 | 7,904,348 |

Both phases plus the probe relinks consume **874.754 CPU seconds**. A reference
relink reproduces the original probe byte for byte; the experimental probe
reuses those model objects and changes only the previously measured v3 loader.

**Stop additional slices and long runs for this fixed mapping.** The original
model and lossless static15 component remain in use. A read-only calculation
on the old tensor histograms finds that one least-squares scale per tensor
removes just 5.44% of its unweighted INT8-space squared error. That is a
weight-space diagnostic; the measured coder loss decides this experiment.
No scale or bit-depth sweep follows this result. Complete enwik9 size and
final packaging remain the prize-level measurements.

## Reproduce the weight file

From the repository root, with assertions enabled:

```sh
mkdir -p /YOUR/temp/quant7
g++ -O2 -std=c++17 research/weights-recode.cpp \
  cpp_infer/src/weights_io.cpp -o /YOUR/temp/quant7/recode
/YOUR/temp/quant7/recode models/6m-q4-fp32.tfwc2 \
  /YOUR/temp/quant7/quant7.tfwc3 --quant7
```

Use a fresh output path and the v3-capable loader for the experiment.
The original model and previous compressed streams stay available.
