# Split8: a single-token FP32-activation workaround for ORT's Q8 path

On **ONNX Runtime 1.31.1, AMD EPYC 7K62, AVX2 without AVX-512/VNNI,
one CPU thread**, replacing one symmetric Q8 weight operator with two
stock Q4 operators and an Add reduces the **M1 K768 N3072** whole-call
median from **5.357 ms to 0.977 ms (5.48×)**. Activations stay FP32;
normalized maximum output error is **3.24e-7** against the same dequantized
weights. Serialized initializer payload grows **12.5%**.

This is a **static-M1 workaround** for a packed-weight deployment that needs
FP32 activations. Dense FP32 remains faster here (**0.266 ms**) at a larger
weight payload. For both tested M197 cases, Split8 increases latency. The
original all-shape acceptance gate failed; the selected M1 gate passed.

[Executable reproduction and graph-building helper](ort_split8.py) ·
[Complete native results, including failed cases and negative controls](results/ort-split8-avx2.json).

## Use the helper on your own symmetric Q8 weights

The script is standalone; downloading this file is sufficient. In an environment
with the dependencies below, its helper produces a one-operator replacement
graph from existing packed Q8 weights:

```python
from pathlib import Path
import onnxruntime as ort
from ort_split8 import split8

# Original MatMulNBits initializers, with symmetric default zero point 128:
# B: uint8 [N, K//32, 32]; S: float32 [N*(K//32)].
# A: float32 [1, K]. K must be divisible by 32.
model_bytes, initializer_bytes = split8(1, K, N, B, S)
Path("split8-m1.onnx").write_bytes(model_bytes)
options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1
session = ort.InferenceSession(model_bytes, options, providers=["CPUExecutionProvider"])
Y = session.run(None, {"A": A})[0]
```

`K`, `N`, `B`, `S` and `A` come from the caller's own operator. The helper
uses constant weights, `block_size=32`, FLOAT inputs/scales and
`accuracy_level=0`. It returns an isolated graph, not a full-model rewrite.
The measured performance scope is M1 K768 N3072 on the host above; other
shapes, thread counts, runtimes and CPUs need their own end-to-end check.

## Why two Q4 terms preserve the symmetric Q8 weights

For an unsigned weight code `q` with default zero point 128 and scale `S`:

```text
(q - 128) * S = (floor(q/16) - 8) * (16*S) + (q % 16 - 0) * S
```

The high term uses the Q4 default zero point 8; the low term supplies an
explicit zero-point initializer filled with **0**. Omitting that initializer
would apply default 8 to the low term and shift every weight by `-8*S`.
The Q4 serialization stores adjacent K codes in each byte: first in the
low nibble, second in the high nibble.

Both native operators receive FP32 activations. Their FP32 outputs are
added, with the actual rounding differences measured below. This graph
uses the existing Q4 path; it leaves ORT's Q8 kernel implementation intact.
The original report's per-call-unpacking attribution comes from its source
analysis; this study measures `Session.run`, without a kernel profiler.

## Full whole-call comparison

Milliseconds, p50; lower is better. All four modes use the same input and
the same Q8-dequantized weights, including the dense FP32 reference.

| M × K × N | Dense FP32 | Q8 acc0 | Split8 | Q8 acc4 | Q8 acc0 / Split8 speedup |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1 × 768 × 3072** | **0.266** | 5.357 | **0.977** | 0.121 | **5.481×** |
| 197 × 768 × 3072 | 9.755 | 16.346 | 20.890 | 23.266 | 0.782× |
| 197 × 3072 × 768 | 9.861 | 17.354 | 20.789 | 23.594 | 0.835× |

The last column is `Q8 acc0 time / Split8 time`. Ratios below one are regressions.
Q8 acc4's low M1 latency comes with the error recorded below.

| M × K × N | Split8 max absolute error | Split8 normalized max error | Q8 acc4 normalized max error |
| :--- | ---: | ---: | ---: |
| **1 × 768 × 3072** | **7.15e-7** | **3.24e-7** | 5.42e-3 |
| 197 × 768 × 3072 | 3.22e-6 | 1.24e-6 | 5.68e-3 |
| 197 × 3072 × 768 | 1.05e-5 | **2.08e-6** | 5.07e-3 |

`normalized max error = max(abs(Y-ref)) / (max(abs(ref)) + 1e-9)`;
this is a whole-tensor normalization, not per-element relative error.
The final M197 row exceeds the predeclared `2e-6` precision gate.

Every shape has 2,359,296 original weight codes. Q8 weights plus scales
occupy **2,654,208 bytes**. Split8's two packed terms, two scale arrays and
low zero points occupy **2,985,984 bytes**. Dense FP32 occupies **9,437,184
bytes**. These are initializer-array payloads, excluding protobuf metadata,
session prepacking, scratch memory and process RSS.

## Reproduce the complete study

```sh
python -m pip install onnx==1.23.2 onnxruntime==1.31.1 numpy==2.5.3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -B ort_split8.py --out /YOUR/temp/split8-results.json
```

The recorded run used Python 3.14.8 and the pinned versions above. The
reporter's seed-0 shape/input generation order is retained. Measurement
order puts M1 first, followed by both M197 cases. Each fresh CPU session
uses one intra-op and one inter-op thread, 30 warmup calls and 200 timed
calls. Timing includes Python `Session.run` and its output allocation;
model construction, session initialization and prepacking are outside it.
The reference is NumPy's FP32 `A @ dequant(B, S)`.
The original issue timed dense MatMul with the pre-quantization `W`; this
study uses `dequant(B, S)` for the dense timing too. CPU feature flags are
read from Linux `/proc/cpuinfo`; other hosts report `null` (unknown).

Native controls before timing:

- Every one of the **256** Q8 codes reproduces exactly with identity inputs
  and power-of-two scales, for both the original Q8 and the split graph.
- The deliberately omitted low zero point produces exactly the expected
  `-8*S` weight shift (maximum absolute error 1.0 for `S=0.125`).
- A separate non-power-of-two scale control has normalized maximum error
  **2.07e-7**.

The JSON preserves the original all-shape `decision.passed=false`. The
published script additionally reports `decision.m1_scope_passed` separately,
using the same predeclared M1 speedup, precision and initializer thresholds.
The stored JSON is the original run, before this readout-only addition.

Original problem, dimensions and base quantization reproduction:
**n24q02m**, [ONNX Runtime issue #34150](https://github.com/microsoft/onnxruntime/issues/34150).
Split graph, native positive/negative controls and independent Linux measurements:
**Blucca / Codex**. The original report runs on Windows / Intel i5-9400F;
the results above describe our Linux / AMD host.
