# Serialize the complete kernel layout once

The fixed serializer rewrites the existing tied-Q8 native graph into
**25 `org.blucca::Q8MatMulM1Packed` nodes**. Every code is re-centered and
reordered once offline; every scale preserves its original FP32 bits.
The quantizer, trained weights and native arithmetic stay unchanged.

```text
T=N/16, KB=K/32
P[t,b,i,l]  = int8(int(B[t*16+l,b,i])-128)
PS[t,b,l]   = S[(t*16+l)*KB+b]
P shape    = INT8[T,KB,32,16]
PS shape   = FLOAT[T,KB,16]
```

The same head P/PS pair directly supplies standard ONNX embedding through
`input_id // 16` and `input_id % 16`, tile Gather then lane Gather. Only
13,824 B of tile codes/scales are selected before local row dequantization.
The graph adds **0 B** of full-vocabulary embedding/dequant/Transpose tables.
The original head Slice preserves 50,257 logits; all fifteen padding rows
remain zero.

## Convert and optional complete file check

With the previous full tied-Q8 native graph and all of its sibling bins:

```bash
python research/ort-q8-single-copy-packed-input/graph/convert.py \
  --source /path/to/tied-q8/models/native.onnx --output_dir /path/to/fresh-packed-model
python research/ort-q8-single-copy-packed-input/graph/check.py \
  --source /path/to/tied-q8/models/native.onnx \
  --model /path/to/fresh-packed-model/native.onnx --receipt /path/to/check.json
```

NumPy and ONNX are the only conversion dependencies. The destination must be
fresh; source files remain read-only. The converter preflights every external
file before parsing, writes a progress receipt after each complete P/PS pair,
and records a failure at its actual stage. A failed construction restarts in
a fresh destination, preserving its existing failure receipt.

Complete checks read all **81,076,224 code bytes** and **10,134,528 scale bytes**,
verify 51 unchanged external files bytewise, and preserve the 1,624 remaining
nodes, 82 other initializer metadata, 76 value_info entries, graph headers
and I/O bytes. The serialized graph has **1,660 nodes / 135 initializers**,
**94,602,744 B initializer payload**, and **101 external bin files**.
P/PS payload is **91,210,752 B**; the net initializer change is -4 B.
`native.onnx` is **1,311,084 B**; graph plus every external bin totals
**95,913,324 B**. Runtime ownership and RSS are measured separately.

Local one-core conversion took **1.544 s**, peak RSS **262,552 KiB** within
700 MiB address space. The complete file check took **0.790 s**, peak RSS
**149,180 KiB**. Original source-stage failures and all successful receipts
are preserved in [`../results/source/graph/`](../results/source/graph/).

The construction manifest retains its original runtime-pending stage.
Actual complete runtime qualification is in
[`../results/acceptance.json`](../results/acceptance.json), which joins the
subsequent fresh target and fixed-language reports.
