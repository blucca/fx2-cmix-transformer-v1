# Single-copy Q8 weights for complete CPU decode

**Keep one packed weight representation, including the tied vocabulary.**
On the fixed DistilGPT2 AVX2/FMA target without AVX-512/VNNI, this removes
**91,210,752 B of native-owned code/scale arrays**, reduces whole fresh-worker
peak RSS **25.08%**, and changes full-call p50 by **+0.43–1.15%**.
All frozen complete numerical, time, memory and fixed-language gates pass.

This extends the [shared-Q8 vocabulary study](../ort-q8-tied-vocabulary/).
The trained model, Q8/block32 quantizer, FP32 activations, old kernel
arithmetic and ORT **1.31.1 / API31** stay unchanged. The independent new
operator reads directly from packed ONNX initializer inputs on each Compute.
All timed calls return **50,257 logits and all twelve complete KV tensors**.
Scope: Linux x64, batch 1, one thread, one-token with-past CPU decode.

[Complete acceptance](results/acceptance.json) ·
[All 600 packed timing samples and thirteen-output checks](results/runs/target/packed/result.json) ·
[Run your own prompt](https://github.com/blucca/fx2-cmix-transformer-v1/releases/tag/ort-single-copy-packed-q8-v0.1.0).

## Full target results

Three **fresh, single-session workers**, old-before → packed → old-after,
use the same existing complete feeds at cache lengths 16, 128 and 512.
Each condition has 30 warmups and 200 timed calls. The faster old p50 and
lower old RSS are the conservative controls for this experiment.

| Past tokens | Faster fresh old p50, ms | Packed p50, ms | Time change |
| ---: | ---: | ---: | ---: |
| 16 | 7.620629 | 7.661386 | +0.535% |
| 128 | 8.161046 | 8.196322 | +0.432% |
| 512 | 10.024439 | 10.140036 | +1.153% |

Peak fresh-worker RSS: **298,112 → 223,336 KiB / 291.1 → 218.1 MiB**, a
**25.0832% reduction**. The frozen contract permits at most 5% longer p50
per condition in exchange for at least 20% lower peak RSS; every condition
passes. All thirteen complete outputs are **bitwise identical** to the
fresh old native outputs at all three cache lengths, including preserved
cached prefixes, new KV slices and top-1.

Host: AMD EPYC 7K62, AVX2/FMA true, AVX-512F/AVX-512 VNNI/AVX VNNI false;
Python 3.14.8, NumPy 2.5.3, ORT 1.31.1. Initialization, conversion, reference
preparation and disk I/O are outside timed calls. Each timed call includes
allocation and return of all thirteen outputs. RSS includes the whole fresh
worker and its runtime/allocator objects. Raw reports retain all samples,
output checks and initialization/resource measurements:
[comparison](results/runs/target/comparison.json),
[old-before](results/runs/target/old-before/result.json),
[packed](results/runs/target/packed/result.json),
[old-after](results/runs/target/old-after/result.json).

### Fixed language and independent KV

Both fresh target workers build independent KV histories for **2,048 fixed
NIST narrative targets**, in two 1,024-target blocks with unscored BOS.
Every step checks finite FP32 shapes, exact own-prefix bits, all logits and
twelve new KV slices against the preserved complete same-Q8 dense oracle,
and identical teacher-forced top-1. All normalized errors are at most 2e-5.
Four independent 32-token prompts followed by 64 greedy tokens match between
the old and packed workers.

| Weights / worker | Mean NLL, nat/token |
| --- | ---: |
| Fresh old native, target | 3.554233919678 |
| Packed native, target | 3.554233919678 |
| Reused original FP32, separate host | 3.545463699641 |

All **2,048 per-token NLL values are identical** between the fresh old and
packed workers; their NLL delta is zero. Total Q8 − original FP32 delta is
**0.008770220 nat/token**, inside the frozen 0.03 gate. The exact target IDs
and four prompt ID sequences match the reused original FP32 report.
The FP32 model and dense oracle are reused from the previous complete study;
no dense, stock or FP32 model is rerun in this experiment. The FP32 report's
separate AMD EPYC host has AVX-512/VNNI flags true. This fixed document and
these four continuations are the measured language scope.

[Complete language comparison](results/language/target/comparison.json) ·
[Every old loss / generated ID](results/language/target/old.json) ·
[Every packed loss / generated ID](results/language/target/packed.json) ·
[Original FP32 report and token IDs](results/language/reused-original/) ·
[Government text and rights](../ort-q8-tied-vocabulary/data/TEXT-SOURCE.md).

## One representation from embedding to projection

The old complete native graph owns **91,210,752 B** of signed/preordered
weights and scales in addition to ORT's **94,602,748 B** initializer payload.
This serializer stores the exact kernel layout directly:

```text
P  : INT8[N/16, K/32, 32, 16]
PS : FLOAT[N/16, K/32, 16]
P[t,b,i,l]  = int8(int(B[t*16+l,b,i]) - 128)
PS[t,b,l]   = S[(t*16+l)*(K/32) + b]
```

The native object keeps API/K/N/KB/T scalars. Each Compute borrows its current
context inputs; native persistent weight-array ownership is **0 B**. Its
`Run` and `FmaPair` preserve the old arithmetic and accumulation order.
The same head P/PS pair supplies embedding through dynamic tile/lane Gather,
with only 13,824 B of codes/scales selected for local dequantization. Extra
full-vocabulary embedding, dequantization and Transpose tables total **0 B**.

| Array payload | Old tied native | Single-copy packed |
| --- | ---: | ---: |
| ONNX initializers | 94,602,748 B | 94,602,744 B |
| Additional native weight arrays | 91,210,752 B | 0 B |
| Combined array payload | 185,813,500 B | 94,602,744 B |

The complete model retains all six attention layers and twelve KV outputs.
It has 1,660 nodes / 135 initializers / 25 new packed operators / 101 external
bins. The existing 50,272-padded head and 50,257-logit Slice are preserved.
[Complete construction manifest](models-manifest.json) ·
[Native interface / source](native/) · [Exact serializer / checks](graph/).
Construction-stage manifests keep their original pending labels; actual
runtime acceptance comes from the later worker reports above.

## Run your own prompt

[Download the complete Linux-x64 native-only bundle](https://github.com/blucca/fx2-cmix-transformer-v1/releases/tag/ort-single-copy-packed-q8-v0.1.0).
It includes the graph, every external weight, the measured 43,056-byte
library, pinned tokenizer, corresponding source and licenses. Core graph /
bins / library / tokenizer payload totals **98,064,033 B before compression**.
The full benchmark's dense oracle and other graphs are separate replay assets.

```bash
WORK="$PWD/temp/packed-first-run"
mkdir -p "$WORK"
unzip distilgpt2-single-copy-packed-q8-native-linux-x64.zip -d "$WORK"
python3.13 -m venv "$WORK/.venv"
PY="$WORK/.venv/bin/python"
"$PY" -m pip install --no-cache-dir numpy==2.5.5 onnxruntime==1.31.1 tokenizers==0.22.1
"$PY" "$WORK/demo_packed_prompt.py" \
  --prompt "In a small research lab, the engineer" --max-new-tokens 32 > "$WORK/own-run.json"
cat "$WORK/own-run.json"
```

The CLI consumes your prompt one token at a time with real KV state,
returns all thirteen outputs each step, and records text, token IDs,
host/runtime versions, cold initialization, full-call time, generation wall
and peak RSS. EOS or 1–64 requested new tokens ends generation; BOS + prompt
+ requested new tokens must fit 1,024. No compiler, model download, conversion
or dense/stock matrix is needed for first Run.
[`demo_packed_prompt.py`](demo_packed_prompt.py) ·
[Small boundary/loop fixture](check_demo_packed_prompt.py).

For a complete model-specific memory/latency scope, send your model/host,
observed result or goal, allowed tolerance and release date to
[@blucca_ai](https://t.me/blucca_ai).

## Reproduce the complete frozen gates

Use the [previous complete reproduction recipe](../ort-q8-tied-vocabulary/#reproduce-all-numerical-timing-memory-and-language-gates)
to create its pinned full `temp/tied-q8-reproduce/` artifacts. That supplies
`models/native.onnx` with 101 sibling bins, `libq8_m1.so`, three complete
`feeds/past-*.npz`, `language/tokens.json`, and both read-only dense-oracle
files: `reference-logits.npy` (411,705,472 B) and `reference-new-kv.npy`
(75,497,600 B). The oracle files total **487,203,072 B** and are rebuilt by
that recipe's dense language worker. The native-only OwnPrompt ZIP supplies
the immediate prompt runtime; the linked source recipe supplies these full
comparison prerequisites and the local original FP32 report.

From repository root, using the previous recipe's existing environment:

```bash
set -euo pipefail
REPO="$PWD"
STUDY="$REPO/research/ort-q8-single-copy-packed-input"
BASE="$REPO/temp/tied-q8-reproduce"
PY="$BASE/.venv/bin/python"
OUT="$REPO/temp/single-copy-packed-reproduce"  # fresh destination
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$OUT"
bash "$STUDY/native/build.sh" "$OUT/libblucca_q8_m1_packed.so"
"$PY" "$STUDY/graph/convert.py" --source "$BASE/models/native.onnx" \
  --output_dir "$OUT/models"
"$PY" "$STUDY/packed_runner.py" --out "$OUT/paired" --base-root "$BASE" \
  --model "$OUT/models/native.onnx" --library "$OUT/libblucca_q8_m1_packed.so"
"$PY" "$STUDY/packed_language.py" --out "$OUT/language" --base-root "$BASE" \
  --model "$OUT/models/native.onnx" --library "$OUT/libblucca_q8_m1_packed.so" \
  --fp32-report "$BASE/language/original.json"

"$PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
assert json.loads((out/'paired/runs/target/comparison.json').read_text())['target_compute_gate_passed']
assert json.loads((out/'language/comparison.json').read_text())['complete_language_gate_passed']
print('COMPLETE_ALL_GATES_PASS')
PY
```

A target qualification requires AVX2/FMA and all three AVX-512/VNNI flags
false. Fresh workers use one allowed core and a 4 GiB address-space bound;
per-stage state/logs survive a disconnected controller. Keep every graph
with all of its declared sibling external bins. The earlier full recipe
uses about 2.5 GB scratch; this mechanism adds the packed model and reports.
`--action self-test` on either new runner checks the small frozen gate
fixtures without loading a model; other required path arguments stay explicit.

Study / Python code: GPL-3.0. Native operator and original ORT SDK: MIT.
Model / tokenizer: Apache-2.0. Bundles carry corresponding source and all
licenses. Original source-stage failures are retained beside successful
source and runtime receipts.
