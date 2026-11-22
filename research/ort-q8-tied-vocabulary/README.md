# A shared Q8 vocabulary for complete CPU decode

**Keep FP32 activations, quantize the tied embedding and output head once,
and measure the whole model.** This extends the
[complete 24-Gemm DistilGPT2 study](../ort-q8-whole-decode/) with one shared
Q8 code/scale pair for token embedding and vocabulary projection. The
existing native kernel and ORT API31 are unchanged.

**Complete target gate PASS:** on AVX2/FMA without AVX-512/VNNI, whole-call p50 falls **34.87–39.38%** against the faster fresh previous-native control, and peak worker RSS falls **31.18%**. The frozen fixed-text FP32 → Q8 loss gate and independently chained full-output/greedy gates also pass.

The complete graph returns **all 50,257 logits and all twelve full KV outputs**.
The original six-layer attention, position embeddings, residuals, LayerNorm
and bias semantics remain intact. Scope: Linux x64, ORT 1.31.1, batch 1,
one thread, one-token with-past decode; cache lengths 16, 128 and 512.

## Complete results

### AVX2/FMA, no AVX-512/VNNI

| Past | Fresh previous native | Shared-Q8 native | Time reduction | Same-Q8 dense | Stock Q8 acc0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 12.9293 | 7.8371 | 39.38% | 19.0610 | 266.2263 |
| 128 | 13.4742 | 8.2808 | 38.54% | 19.5334 | 268.9980 |
| 512 | 15.2393 | 9.9260 | 34.87% | 21.1855 | 282.1193 |

Linux / AMD EPYC 7K62 48-Core Processor / Python 3.14.8 / NumPy 2.5.3 / ORT 1.31.1, one allowed core. Peak native RSS: **449.5 → 309.3 MiB / 31.18% reduction**. All four same-Q8 modes pass the complete numerical contract.


[Full target comparison](results/runs/target/comparison.json) ·
[Every native output check and timing sample](results/runs/target/native/result.json) ·
[Target language-chain comparison](results/language/target/comparison.json).

The before/after baseline is the **previous 24-Gemm native graph with its
FP32 vocabulary**, remeasured in this experiment with the same complete
inputs. Its faster p50 and lower peak RSS are the conservative controls.
The same-Q8 dense graph is the separate numerical reference. Initialization,
conversion, reference preparation and disk I/O are outside the timed calls;
every timed call allocates and returns all thirteen model outputs.

### Separate AVX-512/VNNI host

All complete numerical, time and memory gates pass on the separate AMD EPYC
host, whose AVX-512/VNNI feature flags are true. The AVX2/noVNNI qualification
belongs to the target above. One-thread p50, milliseconds:

| Past | Fresh previous native | Shared-Q8 native | Time reduction | Same-Q8 dense | Stock Q8 acc0 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 6.3471 | 4.7503 | 25.16% | 11.3951 | 251.8781 |
| 128 | 6.9549 | 5.1242 | 26.32% | 11.2439 | 225.8233 |
| 512 | 9.9654 | 7.4174 | 25.57% | 12.7520 | 261.0896 |

Peak native worker RSS: **445.1 → 304.8 MiB / 31.52% reduction**.
[Full comparison and CPU/runtime versions](results/runs/remote/comparison.json).
Each host has six fresh, single-session workers: old-before, dense-before,
native, stock Q8 acc0, dense-after and old-after; 30 warmups and 200 timed
calls per cache condition. Results retain every sample and every output check.

### Fresh-worker resources

| Host / graph | Session initialization, s | Peak worker RSS, MiB |
| :--- | ---: | ---: |
| target / Shared-Q8 native | 0.248 | 309.3 |
| target / Same-Q8 stock acc0 | 0.091 | 361.7 |
| target / Same-Q8 dense-before | 0.576 | 521.0 |
| target / Same-Q8 dense-after | 0.521 | 521.2 |
| remote / Shared-Q8 native | 0.129 | 304.8 |
| remote / Same-Q8 stock acc0 | 0.062 | 356.6 |
| remote / Same-Q8 dense-before | 0.399 | 516.8 |
| remote / Same-Q8 dense-after | 0.374 | 517.0 |


These are complete process RSS and initialization measurements, including
runtime objects, allocator and optimization storage. Native and stock share
the same initializer payload; the native constructor also retains the
separately counted code/scale snapshot/prepack below. Pick the operating point
using the measured memory and whole-call latency target. Hot-call timing and
initializer/array payloads have their own scopes.

### Paired language loss and independent KV chains

The frozen [NIST government narrative](data/TEXT-SOURCE.md) contains 2,285
available tokens; the probe scores the first **2,048 targets** in two
independent blocks of 1,024, with unscored BOS conditions. Float64 stable
logsumexp and `math.fsum` measure per-token NLL; each model builds its own KV.

| Model / weights | Mean NLL, nats per target |
| :--- | ---: |
| Original FP32 | 3.54546369964 |
| Previous 24-Gemm Q8 / FP32 vocabulary | 3.54644598676 |
| Shared-Q8 dense reference | 3.55423411001 |
| Shared-Q8 native | 3.55423389362 |

**Total FP32 → shared-Q8 loss increase: 0.008770194 nat/token**, within the
frozen 0.03 gate. The old 24-Gemm contribution is 0.000982287 and the added
shared vocabulary contribution is 0.007787907. These are paired losses on
this fixed document; broader corpus quality has its own measurement scope.
[Full paired comparison](results/language/remote/comparison.json) ·
[All 2,048 native per-token losses and generated token IDs](results/language/remote/native.json).

Dense and native use **independent** KV histories. Every one of the 2,048
steps passes finite/shape, exact old-prefix bits, all logit and twelve
new-slice normalized-error checks, with identical teacher-forced top-1.
Four separate 32-token prompts followed by 64 greedy tokens produce identical
token sequences between the same-Q8 dense and native models. The target
independently repeats the same-Q8 chains and four continuations. The full
paired FP32 loss gate is measured on the separate host and combined with the
target compute and target chain gates in [complete acceptance](results/acceptance.json).

## Frozen acceptance contract

Whole outputs and each newly appended KV slice must independently satisfy
`maxabs(error) / max(1, maxabs(reference)) <= 2e-5` against the same-Q8 dense
control, with identical top-1 and exact old-prefix `uint32` bits. Every one
of the three cache lengths must reduce whole-call p50 by at least **15%**
against the faster fresh previous-native before/after control. Peak native
worker RSS must fall at least **15%** against the lower previous-native RSS.
The target CPU qualification, all complete numerical checks, time, memory,
independent language chain/greedy and paired 0.03-nat/token loss gates are
combined; the same source, quantizer and fixed text remain throughout.

## How the shared vocabulary works

1. Verify that the source embedding Gather and LM-head Transpose/MatMul
   consume the **same** original `[50257,768]` FP32 initializer.
2. Pad the output vocabulary to **50,272** for the unchanged 16-column native
   tile. Quantize once using the earlier symmetric absmax/block32 function,
   with zero point 128 and FP32 scales. The fifteen padded rows dequantize to 0.
3. Embed a token with standard ONNX row Gather on the shared code array,
   FP32 Cast/Sub, Gather of reshaped block scales and FP32 Mul. Only the
   selected row is dequantized.
4. Feed the same initializer pair to the existing `Q8MatMulM1` output head.
   Slice its result back to the original **50,257** logits before token
   selection. Attention and all KV inputs/outputs preserve the source contract.
5. Build same-weight dense and stock Q8 acc0 controls. The dense embedding
   and projection share the FP32 dequantization of the exact same codes/scales.

The converter checks unchanged non-vocabulary nodes/initializers and all
input/output metadata byte-for-byte. It does not change the existing
24-Gemm quantizer, native core or API31. The native and stock graphs contain
1,656 nodes / 25 weight-only operators; the dense reference retains 1,600 nodes.

| Storage payload | Previous native | Shared-Q8 native |
| :--- | ---: | ---: |
| ONNX initializer tensors | 205,557,120 B | **94,602,748 B** |
| Additional native code/scale prepack | 47,775,744 B | **91,210,752 B** |
| Combined array payload | 253,332,864 B | **185,813,500 B** |

The shared vocabulary's canonical codes/scales occupy **43,435,008 B**.
The native head takes its own constructor snapshot/prepack of that pair;
the table counts this extra payload separately. Process RSS also includes
runtime/allocator/object overhead and is reported per fresh worker.
[Complete conversion/storage manifest](models-manifest.json).

FP32-activation weight-only inference and quantized embeddings have existing
implementations, including [ORT MatMulNBits](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
and [Intel IPEX](https://github.com/intel/intel-extension-for-pytorch/blob/main/docs/tutorials/llm/llm_optimize.md).
This contribution is a complete, tied-weight same-ORT integration, paired
precision/quality checks and measured whole-call trade-offs on this fixed model.

## Run your own prompt, without rebuilding the benchmark

[Download the fixed native-only Linux bundle](https://github.com/blucca/fx2-cmix-transformer-v1/releases/tag/ort-tied-q8-v0.1.0).
It includes the complete native graph and every required external weight,
the existing Linux library, pinned tokenizer, first-Run source and licenses.
The ZIP is **90,933,356 B**; its core runtime assets total **98,058,845 B before compression**. Install
three Python wheels and run your own prompt; the original model download,
ONNX conversion, compiler and dense/stock matrix are handled by the release.

```bash
WORK="$PWD/temp/tied-first-run"
mkdir -p "$WORK"
unzip distilgpt2-shared-q8-native-linux-x64.zip -d "$WORK"
python3.13 -m venv "$WORK/.venv"
PY="$WORK/.venv/bin/python"
"$PY" -m pip install --no-cache-dir numpy==2.5.5 onnxruntime==1.31.1 tokenizers==0.22.1
"$PY" "$WORK/demo_native.py" --prompt "In a small research lab, the engineer" \
  --max-new-tokens 32 > "$WORK/own-run.json"
cat "$WORK/own-run.json"
```

This fixed reference model runs the complete 13-output graph, consuming prompt
and generated tokens one at a time with real KV state. Generation stops at
EOS or the requested 1–64 new tokens; BOS + encoded prompt + requested new
tokens must total at most 1,024. Every step checks finite FP32 shapes and
exact old-prefix bits. JSON records decoded text, actual token IDs, model and
host/runtime versions, cold initialization, full-call time, generation wall
and RSS. This cold own-prompt Run has its own timing scope; the six-worker
experiment above supplies the controlled hot-call p50 comparison.

[`demo_native.py`](demo_native.py) · [Small boundary/loop fixture](check_demo_native.py).
Bring your own model/host, observed result or goal, allowed tolerance and date
to [@blucca_ai](https://t.me/blucca_ai) for a complete engineering scope.


## Reproduce all numerical, timing, memory and language gates

Prerequisites: Linux x64 **AVX2 + FMA**, C++17 compiler, Python 3.13+, about
**2.5 GB scratch disk** including the virtual environment. AVX2/noVNNI
qualification additionally uses hosts with AVX-512F, AVX-512 VNNI and AVX VNNI
flags all false. One model/session runs at a time; the worker address-space
limit is 4 GiB, and measured peak RSS is reported separately. From repo root:

```bash
set -euo pipefail
REPO="$PWD"
STUDY="$REPO/research/ort-q8-tied-vocabulary"
OLD="$REPO/research/ort-q8-whole-decode"
WORK="$REPO/temp/tied-q8-reproduce"
BASE="$WORK/previous"
mkdir -p "$WORK/tmp" "$BASE" "$REPO/temp"
cp "$OLD/prepare.py" "$OLD/converter.py" "$OLD/native_runner.py" "$BASE/"
cp "$STUDY/"*.py "$WORK/"
export TMPDIR="$WORK/tmp" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python3.13 -m venv "$WORK/.venv"
PY="$WORK/.venv/bin/python"
"$PY" -m pip install --no-cache-dir numpy==2.5.5 onnx==1.23.2 \
  onnxruntime==1.31.1 tokenizers==0.22.1

# The same existing native library serves both old and new graphs.
"$PY" "$REPO/research/ort-q8-m1-native/build.py" --out "$BASE/libq8_m1.so"
cp "$BASE/libq8_m1.so" "$WORK/libq8_m1.so"
"$PY" -u "$BASE/prepare.py"
"$PY" -u "$BASE/converter.py" --source "$BASE/source.onnx" --out-dir "$BASE/models"
"$PY" -u "$WORK/tied_converter.py" --base-root "$BASE" --out-dir "$WORK"
(cd "$WORK"; "$PY" -u "$WORK/prepare-tokenizer.py")

"$PY" -u "$WORK/tied_runner.py" --root "$WORK" --base-root "$BASE" \
  --action all --label my-host --max-as-mib 4096
"$PY" - "$WORK" <<'PY'
import json,sys
from pathlib import Path
x=json.loads((Path(sys.argv[1])/'runs/my-host/comparison.json').read_text())
assert x['numerical_pass'], 'STOP: complete numerical gate failed'
PY
"$PY" -u "$WORK/language_probe.py" --root "$WORK" --base-root "$BASE" \
  --action prepare --text "$STUDY/data/fixed-language-text-2048.txt" \
  --tokenizer "$WORK/tokenizer.json"
"$PY" -u "$WORK/language_probe.py" --root "$WORK" --base-root "$BASE" \
  --action all --modes dense,native,original,old_native --max-as-mib 4096

"$PY" - "$WORK" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1])
c=json.loads((r/'runs/my-host/comparison.json').read_text())
l=json.loads((r/'language/comparison.json').read_text())
assert c['target_compute_gate_passed'], 'STOP: target compute gate failed'
assert l['complete_language_gate_passed'], 'STOP: paired language gate failed'
print('COMPLETE_ALL_GATES_PASS')
PY
```

`prepare.py` verifies the pinned 328,939,971 B source before conversion; copy
it into `BASE` as shown because it writes next to its own script. The native
builder uses its bundled six, previously measured API31 headers. The tied
runners import the previous complete oracle from `BASE`.
`language_probe.py --action prepare` freezes the text/tokens; `--action all`
runs the prepared models. Execution completion is in each `state.json`, and
the separate numerical/CPU/time/RSS/chain/loss gates are in `comparison.json`.
The converter manifest is a construction-stage record; runtime acceptance
comes from those actual worker reports. Keep each graph with every sibling
external `.bin` file listed in its manifest when moving it.

For a cross-host transfer, `assemble_target.py` copies the unchanged tensors
from existing previous artifacts and reconstructs the dense vocabulary from
the small canonical shared B/S pair. This reproduces the same graphs and
library bytes without retransmitting the large dense vocabulary.

Small checks without the downloaded source model:

```bash
"$PY" "$BASE/converter.py" --self-test
"$PY" "$BASE/native_runner.py" --action self-test
"$PY" "$WORK/check_comparison.py"
"$PY" "$STUDY/check_layout.py"
"$PY" "$WORK/language_probe.py" --root "$WORK" --base-root "$BASE" --action self-test
```

The layout check covers row/block Gather, zero-padding, final native tile
address equations and slicing before argmax. The compute oracle includes a
failing stock-mode numerical fixture. These small mathematical/summary
controls accompany the complete model output and independently chained tests.

## Source, authorship and licenses

- Source model and tokenizer: **Xenova/distilgpt2@a41c10485c18a64b6606729b6a082330cbd8f49e**,
  [`onnx/decoder_with_past_model.onnx`](https://huggingface.co/Xenova/distilgpt2/tree/a41c10485c18a64b6606729b6a082330cbd8f49e),
  the Transformers.js ONNX export of [Apache-2.0 DistilGPT2](https://huggingface.co/distilbert/distilgpt2).
- Study: **Blucca / Codex**, under the repository's [GPL-3.0](../../LICENSE).
  The [existing reference quantizer](../ort_split8.py) and
  [native operator](../ort-q8-m1-native/) retain their original reported-need
  attribution, including the independent author's i5/Windows stock controls.
- The unchanged native C++ source/builder retain their [MIT license](../ort-q8-m1-native/LICENSE).
  The bundled API31 SDK retains [Microsoft's MIT license](../ort-q8-m1-native/LICENSE-onnxruntime).
- The fixed document's credit, permission and exact selection are in
  [TEXT-SOURCE.md](data/TEXT-SOURCE.md).

For a CPU deployment, bring the actual model/host, memory and latency target,
allowed output tolerance and release date. We can agree the complete
investigation/integration scope from those materials.
[Send a project brief](https://t.me/blucca_ai) ·
[US$150 two-runtime parity diagnosis](../../README.md#us150-cpu-parity-diagnosis).
