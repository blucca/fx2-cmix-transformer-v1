# Same-Q8 weights, complete one-token CPU decode

An end-to-end ONNX Runtime study: integrate the existing **Q8M1** AVX2/FMA
operator into all **24 constant Gemm layers** of a complete DistilGPT2
with-past decoder. Activations stay FP32. Attention, residuals, LayerNorm,
the unchanged FP32 embeddings/vocabulary projection and **all twelve full KV outputs**
remain in the timed `Session.run`.

**Scope:** Linux x64, ORT 1.31.1, one thread, batch 1, one next token;
controlled past lengths 16, 128 and 512. The dense control uses the **same
quantized weights dequantized to FP32**, including the original bias semantics.

## Complete results

**AVX2/FMA, no AVX-512/VNNI: complete gate PASS.** On Linux / AMD EPYC
7K62 / Python 3.14.8 / NumPy 2.5.3 / ORT 1.31.1, native whole-call p50
falls **28.17–32.60%** against the faster same-host dense control. All
four same-Q8-weight modes pass all logits/KV, appended-slice, prefix-bits
and top-1 checks. Maximum native normalized error, including appended
slices, is **9.965e-7** (frozen tolerance: 2e-5).

| Past tokens | Faster dense before/after | Native Q8M1 | Time reduction | Stock Q8 acc0 | Stock two-Q4 Split8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 19.2735 | 12.9902 | 32.60% | 94.0500 | 30.2786 |
| 128 | 19.6973 | 13.5672 | 31.12% | 95.0688 | 30.8463 |
| 512 | 21.3716 | 15.3511 | 28.17% | 115.0203 | 32.5279 |

[Full AVX2/noVNNI comparison](results/runs/target/comparison.json) ·
[Native timings and every output check](results/runs/target/native/result.json) ·
[This host's chained cache preparation](results/feeds/target-manifest.json).
Native peak worker RSS is **450.2 MiB**, versus **521.9 MiB** for
dense-after; native session initialization is **0.546 s**.


On a separate AMD EPYC host with **AVX-512/VNNI available**, all four modes
pass the complete numerical contract. Steady-state p50, milliseconds:

| Past tokens | Faster dense before/after | Native Q8M1 | Time reduction | Stock Q8 acc0 | Stock two-Q4 Split8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 11.8245 | 8.0598 | 31.84% | 51.2492 | 10.4539 |
| 128 | 12.2693 | 9.2637 | 24.50% | 51.7871 | 9.8687 |
| 512 | 14.0539 | 11.8463 | 15.71% | 55.6910 | 13.3378 |

[Full AVX-512-host comparison](results/runs/remote/comparison.json).
This host's `target_avx2_no_vnni` flag is false. Each host has its own
same-input dense-before reference and two dense timing controls. The
qualification gate requires all numerical checks and at least **15% time
reduction at every tested cache length** on an AVX2/FMA, no-AVX-512/VNNI host.

FP32-activation / W8 inference already has implementations:
[ORT MatMulNBits](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
and [Intel IPEX weight-only quantization](https://github.com/intel/intel-extension-for-pytorch/blob/main/docs/tutorials/llm/llm_optimize.md).
The measured comparison here covers four same-contract ORT graphs: dense,
stock Q8 `accuracy_level=0`, stock two-Q4 Split8 and the existing native Q8M1.
IPEX is a separate existing implementation, outside this four-graph experiment.

## What is checked and timed

- **Complete graph:** pinned source, 1,600 original nodes, 24 Gemm, 13 unchanged
  MatMul, 14 inputs and 13 outputs. The converter fixes batch 1 and preserves
  all non-Gemm nodes. Gemm `alpha=beta=1`, `transA=0`, both weight orientations
  and original legal bias broadcasting have small executable controls.
- **Identical Q8 weights:** symmetric absmax/block32, zero point 128, common
  codes and FP32 scales. The dense control retains the original Gemm nodes
  with these weights' FP32 dequantization. Stock and native activations are FP32.
- **All outputs:** finite FP32 tensors, exact expected shapes, all 50,257
  logits and all six layers' keys/values. For every whole output **and every
  newly appended KV slice**, `maxabs(error) / max(1, maxabs(reference)) <= 2e-5`.
  Logit top-1 must match; the old cache prefix retains its input's exact
  `uint32` bits. All four modes' numerical results enter the summary gate.
- **Real cache computation, controlled material:** seed-725 synthetic token
  IDs, starting with empty past and chaining actual dense-Q8 model cache
  outputs. Each tested step receives a saved, complete legal prefix. This
  measures same-weight execution of the next step at a fixed cache; natural
  language quality, the original FP32-to-Q8 quantization loss and free-running
  autoregressive divergence have separate evaluation scopes.
- **Whole hot call:** five fresh single-session workers; 30 warmups and 200
  timed calls per cache condition, ORT CPU EP and `ORT_ENABLE_ALL`, one pinned
  allowed core. Every timed call returns logits **plus all twelve full KV
  tensors**, including their allocation. Download, conversion, cache creation,
  checking and disk I/O are outside the hot-call samples. Session initialization
  is reported separately. The faster same-host dense-before/after p50 is the
  conservative timing control.

The existing native constructor snapshots constant codes/scales and packs
signed codes into 16-column / 32-K tiles. It converts codes to FP32 registers
while multiplying; it retains **47,775,744 B** of additional packed array
payload for all 24 operators. This storage is separate from model initializers:

| Graph | Initializer tensor payload |
| :--- | ---: |
| Dense same-Q8-dequantized | 327,650,304 B |
| Native Q8M1 | 205,557,120 B |
| Stock Q8 acc0 | 205,557,120 B |
| Two-Q4 Split8 | 211,529,088 B |

On the AVX-512 host, peak native worker RSS is **445.7 MiB**, versus
**517.4 MiB** for dense and **355.5 MiB** for stock Q8. Native uses more
process memory than stock Q8 while lowering the hot-call latency. Native
session initialization is **0.342 s**, versus **0.371 s** for dense-before.

Initializer/prepack figures count tensor/array payloads. Process RSS includes
allocator and object overhead and is recorded for each worker. The four
external-data graphs share unchanged initializer files: **434,415,968 B**
unique artifacts in total. [Complete model manifest](models-manifest.json) ·
[Original graph inventory](source-graph.json) ·
[Fixed-token/cache preparation manifest](results/feeds/manifest.json).

## Reproduce the complete experiment

Prerequisites: **Linux x64 with AVX2 and FMA**, C++17 compiler, Python 3.13
or newer, about 1.6 GB scratch disk. The worker address-space limit below is
4 GiB; published worker RSS is recorded separately. One model/session runs
at a time. From the repository root:

```bash
STUDY="$PWD/research/ort-q8-whole-decode"
WORK="$PWD/temp/whole-q8-decode"
mkdir -p "$WORK/tmp"
cp "$STUDY/prepare.py" "$STUDY/converter.py" "$STUDY/native_runner.py" "$WORK/"
export TMPDIR="$WORK/tmp" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python3.13 -m venv "$WORK/.venv"
PY="$WORK/.venv/bin/python"
"$PY" -m pip install --no-cache-dir numpy==2.5.5 onnx==1.23.2 onnxruntime==1.31.1
"$PY" research/ort-q8-m1-native/build.py --out "$WORK/libq8_m1.so"
"$PY" -u "$WORK/prepare.py"
"$PY" -u "$WORK/converter.py" --source "$WORK/source.onnx" --out-dir "$WORK/models"
"$PY" -u "$WORK/native_runner.py" --root "$WORK" --action all --label my-host --max-as-mib 4096
cat "$WORK/runs/my-host/comparison.json"
```

`runs/my-host/state.json` records completion or the failed stage;
`comparison.json` records numerical/performance/CPU qualification gates.
Each worker's `result.json` includes p50, p95, all 200 timing samples,
initialization seconds, peak RSS and every output check. `DONE` records
completed execution; use `target_gate_passed` for the numerical/performance
qualification. Keep each graph with **all sibling external `.bin` files**
listed in the manifest when moving it.

The native builder uses its bundled six ORT API-31 headers (Microsoft MIT).
The kernel and API version are the same as the published measurements.

Small checks, without the downloaded model:

```bash
"$PY" "$STUDY/converter.py" --self-test
"$PY" "$STUDY/native_runner.py" --action self-test
"$PY" "$STUDY/check_comparison.py"
```

The comparison oracle includes a deliberately failing stock-mode numerical
result and requires the complete numerical/target gates to fail.

## Source, authorship and licenses

- Complete source: [`Xenova/distilgpt2@a41c10485c18a64b6606729b6a082330cbd8f49e`](https://huggingface.co/Xenova/distilgpt2/tree/a41c10485c18a64b6606729b6a082330cbd8f49e),
  `onnx/decoder_with_past_model.onnx`, **328,939,971 B**.
  `prepare.py` checks the original HF LFS SHA-256
  `2c9b7dd7f11facae49943e28ef08ba91d37850232962b5e945bb8bc6ede5495b`.
- Original [DistilGPT2 model card](https://huggingface.co/distilbert/distilgpt2/tree/2290a62682d06624634c1f46a6ad5be0f47f38aa):
  Apache-2.0; Xenova supplies the Transformers.js ONNX export.
- Study: **Blucca / Codex**, under the repository's [GPL-3.0](../../LICENSE).
  Q8 quantization and Split8 arithmetic reuse [the existing reference](../ort_split8.py),
  with the original reported need and quantization reproduction attributed there.
- The existing [native source and builder](../ort-q8-m1-native/) retain their
  [MIT license](../ort-q8-m1-native/LICENSE); the ORT SDK retains
  [Microsoft's MIT license](../ort-q8-m1-native/LICENSE-onnxruntime).
  This study reuses the kernel unchanged.

For integration into **your** model, share its revision, CPU, cache conditions,
latency target, tolerance and deadline with [@blucca_ai](https://t.me/blucca_ai).
The complete implementation and acceptance scope can then be priced from
those materials. The [US$150 two-runtime parity diagnosis](../../README.md#us150-cpu-parity-diagnosis)
remains a separate, fixed-scope service.
