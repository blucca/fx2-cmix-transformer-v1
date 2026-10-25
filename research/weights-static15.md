# Lossless per-tensor INT4 weight coding

The original `FX2TFWC2` file contains 5,868,864 quantized weights in 111
matrices. Its shared shift-5 binary tree pays adaptation noise on these
mostly stationary symbols. A small static histogram per tensor reduces the
weight file by **26,327 bytes**, retaining every tensor byte.

## Measured result

| Item | Original | Static15 |
|---|---:|---:|
| Complete weight file | 2,930,652 B | **2,904,325 B** |
| v26 development executable, UPX 5.2.1 `-9` | 202,304 B | 202,620 B |
| Two-copy weights + executable saving | | **52,022 B** |

All **434 tensors** have identical names, dtypes, shapes and data bytes,
including the regenerated RoPE tables. The executable comparison recompiles
only the load-time weight decoder with the original Clang 17 flags and
relinks the original v26 objects. A reference relink exactly reproduces the
previous executable. Both unpacked binaries occupy 509,920 bytes; UPX
exposes the 316-byte executable increase. [Exact receipt](results/weights-static15.json).

The official package includes two copies of each component, so the measured
development saving is `2 * (26,327 - 316) = 52,022 B`. This component covers
about 5.2% of the current 1,004,247-byte record threshold. Complete enwik9
size, final PGO packaging and prize qualification remain future measurements.

## Why this model

The [read-only census](results/weights-census.json) compares these INT4 costs:

| Model | Bytes |
|---|---:|
| Global histogram, oracle | 2,777,440 |
| Per-tensor histogram, oracle | 2,773,771 |
| Per-row / per-column histograms, oracle | 2,734,466 / 2,733,453 |
| Per-tensor causal KT | 2,775,000 |
| Causal row / column model, parent strength 64 | 2,792,381 / 2,791,407 |
| Shift-5 tree reset at each tensor | 2,803,699 |
| Static 4096-total model, including 28 B/tensor | 2,776,881 |

Oracle histograms exclude their descriptions. Causal models include cold
start; their displayed costs omit range rounding. The reset shift-5 result
is a diagnostic, while the actual file comparison preserves the original
tensor order and all non-INT4 encodings. Row and column models lost against
the simpler tensor model, so their implementation stops here.

## Format and reproduction

`FX2TFWC3` adds encoding tag 6 to the existing adaptive container. Each INT4
tensor supplies 14 little-endian uint16 frequencies through the existing
metadata model. Frequencies are positive and sum to 4096; the fifteenth is
implied. The encoder allocates one count per symbol, distributes the
remaining 4081 proportionally, and assigns rounding remainders in descending
order, breaking ties by symbol. A byte-wise range coder then codes the
15-symbol alphabet. The original tensor order, values, other payload models
and RoPE reconstruction are retained.

No Python ML package or retraining is needed. From the repository root:

```sh
mkdir -p /YOUR/temp/weights
g++ -O2 -std=c++17 research/weights-census.cpp \
  cpp_infer/src/weights_io.cpp cpp_infer/src/weights_io_compressed.cpp \
  -o /YOUR/temp/weights/census
/YOUR/temp/weights/census models/6m-q4-fp32.tfwc2 \
  > /YOUR/temp/weights/census.json
g++ -O2 -std=c++17 research/weights-recode.cpp \
  cpp_infer/src/weights_io.cpp -o /YOUR/temp/weights/recode
/YOUR/temp/weights/recode models/6m-q4-fp32.tfwc2 \
  /YOUR/temp/weights/6m-q4-fp32.tfwc3
```

The recoder reloads both files and compares every tensor before reporting
success. Keep assertions enabled for this research check. The existing
Python v2 writer and default model path stay unchanged; use the new C++
loader with an explicitly selected v3 file.

## Opt-in model-size experiment

The separate [seven-value rate/distortion screen](weights-quant7.md) adds an
explicit `--quant7` recoder mode. Its tensor values change; the default mode
and all lossless results above stay unchanged.
