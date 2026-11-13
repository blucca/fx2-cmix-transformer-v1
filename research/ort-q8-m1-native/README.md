# Q8M1: a native FP32-activation operator for one-token CPU inference

**ORT 1.31.1, Linux / AMD EPYC 7K62, AVX2 + FMA, no AVX-512/VNNI,
one thread, M1 K768 N3072:** this custom operator runs at **0.146 ms**
whole-call p50, **6.70× faster** than the two-Q4 Split8 graph and
**1.66× faster** than dense FP32 on the same dequantized weights.
Normalized maximum error is **3.78e-7**. Activations stay FP32.

[Full Linux result](results/linux-avx2.json) ·
[Kernel](q8_m1_kernel.hpp) · [ORT adapter](q8_m1_ort.cpp) ·
[Direct-use graph helper](q8_m1.py) · [Reproduction](bench.py).

This is a **custom operator**, loaded into an existing ORT CPU session.
It leaves the stock ORT Q8 implementation intact. The measured performance
scope is the shape and host above; numerical controls additionally cover
N16/K32 and N16/K64. M197 is outside the operator's static-M1 contract.
The earlier [stock-two-Q4 workaround and its full M197 regressions](../ort-split8-avx2.md)
remain available.

## Build and use

Prerequisites: **x64 CPU with AVX2 and FMA**, ORT **1.31.1**, NumPy and ONNX.
Weights are constant symmetric Q8 with zero point **128**, `block_size=32`;
K is a positive multiple of 32 and N a positive multiple of 16.
Inputs/scales are FLOAT32.

From this repository's root, with a Linux C++17 compiler:

```sh
python research/ort-q8-m1-native/build.py --out /YOUR/temp/libq8_m1.so
```

In a Windows x64 MSVC developer shell, use the same build entry:

```powershell
python research/ort-q8-m1-native/build.py --out C:\YOUR\temp\q8_m1.dll
```

The builder downloads six pinned ORT SDK headers to the output folder's
`sdk-1.31.1/`, builds the single source and writes its small build receipt.
Linux uses `-O3 -mavx2 -mfma`; Windows uses `/O2 /arch:AVX2`.
Windows compilation/ABI validation is being run separately; its result
will be added here. The source's numerical schedule uses explicit FMA,
with compiler fast-math disabled.

With `q8_m1.py` on your Python module path:

```python
import onnxruntime as ort
from q8_m1 import native_m1

# B: uint8 [N,K//32,32], S: float32 [N*K//32], A: float32 [1,K].
model_bytes, initializer_bytes = native_m1(B, S)
options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1
options.register_custom_ops_library("/YOUR/temp/libq8_m1.so")  # Windows: q8_m1.dll
session = ort.InferenceSession(model_bytes, options, providers=["CPUExecutionProvider"])
Y = session.run(None, {"A": A})[0]
```

The helper builds an isolated one-operator graph from the caller's own
codes and scales. The adapter checks constant initializers and static
A/Y shapes. A full model integration uses that model's actual feed set,
output tolerances and end-to-end timing.

## Whole-call results

All three modes use the same original seed-0 M1 input and Q8 codes/scales.
The dense reference uses `dequant(B,S)`. Timing covers Python `Session.run`
and output allocation, after 30 warmup calls, across 200 timed calls.
Session construction and constant-weight prepacking are outside timing.

| Mode | p50 ms | p95 ms | Normalized max error | Initializer tensor bytes |
| :--- | ---: | ---: | ---: | ---: |
| Dense FP32 | 0.242462 | 0.271642 | 5.13e-7 | 9,437,184 |
| Stock two-Q4 Split8 | 0.978378 | 1.003880 | 3.24e-7 | 2,985,984 |
| **Native Q8M1** | **0.146056** | **0.155193** | **3.78e-7** | **2,654,208** |

`normalized max error = max(abs(Y-ref)) / (max(abs(ref)) + 1e-9)`,
where `ref = A @ dequant(B,S)` in NumPy FP32. This normalization applies
to the whole output tensor. The native schedule's actual rounding
difference is measured above.

The native operator retains an additional **2,654,208-byte prepack cache**
(signed codes plus reordered scales). Initializer and cache counts cover
those arrays; model metadata, allocator overhead, input/output tensors
and process RSS have their own accounting.

## Mechanism and controls

The constructor converts `q-128` to signed int8 and rearranges weights
into **16 output columns × 32 K entries**, ordered K-inner / output-lane.
Each call broadcasts FP32 activations, converts eight signed codes at a
time to FP32 registers, and uses **four independent FMA chains per
eight-lane half**. Each 32-entry block is then scaled in FP32 and added
to the output. No full FP32 weight matrix is created.

Controls through the actual ORT session:

- All **256** weight codes reproduce exactly with identity inputs and
  power-of-two scales; the table is duplicated to the native N16 tile.
- Deliberately wrong zero encoding yields the expected maximum error **16**.
- A non-power-of-two, two-block scale control has normalized error **1.02e-7**.
- Reversing the two block scales yields normalized error **0.0256**.

Separate small C++/adapter checks cover snapshot ownership after source
B/S changes, two live registered sessions, constant-weight requirements,
tensor shapes and unsupported M2 feeds. The executable reproduction
reruns the native controls and this M1 comparison:

```sh
python -m pip install onnx==1.23.2 onnxruntime==1.31.1 numpy==2.5.3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python research/ort-q8-m1-native/bench.py \
  --library /YOUR/temp/libq8_m1.so --out /YOUR/temp/q8-m1-results.json
```

Original need and seed-0 quantization reproduction: **n24q02m**,
[ONNX Runtime #34150](https://github.com/microsoft/onnxruntime/issues/34150).
Native kernel, adapter and independent measurements: **Blucca / Codex**.
The kernel, adapter, graph helper and builder are MIT licensed;
the benchmark reuses the existing repository's GPL reference helper.
ORT SDK code is used under Microsoft's MIT license.
