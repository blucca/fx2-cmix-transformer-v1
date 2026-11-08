# CPU QDQ parity: keep the original BatchNorm path

The original [ONNXRuntime #32918 report](https://github.com/microsoft/onnxruntime/issues/32918)
by **kadu-v** supplies a small graph and seed-42 input that change by one quantization
step with full graph optimization. The report observes this on macOS ARM64.
I reproduced the attached graph on **Linux x86_64, CPUExecutionProvider,
ONNXRuntime 1.30.0, NumPy 2.5.3, Python 3.12.3**, then isolated a working
same-input control.

**Keeping both `ConvBNFusion` and `NchwcTransformer` disabled restores all
196,608 final values exactly to the unoptimized reference in this run.
`MatMulAddFusion` stays enabled and the optimized graph still contains its Gemm.**

## One graph, six routes

Each route uses the same original graph, input and single-thread CPU settings.
The reference is `ORT_DISABLE_ALL`; every other route uses `ORT_ENABLE_ALL`.

| Disabled passes | Changed final elements | Maximum final difference |
| :--- | ---: | ---: |
| none | 4 | 0.03125 |
| MatMulAddFusion | 4 | 0.03125 |
| ConvBNFusion | 4 | 0.03125 |
| NchwcTransformer | 4 | 0.03125 |
| ConvBNFusion + NchwcTransformer | **0** | **0** |

The MatMul-only and NCHWc-only interventions return the same final values as
the fully optimized route. The ConvBN-only intervention changes the set of
four differing coordinates. This last detail led to the joint control.

```python
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(
    "minimal_model.onnx", options, providers=["CPUExecutionProvider"],
    disabled_optimizers={"ConvBNFusion", "NchwcTransformer"},
)
```

This is a measured parity choice for the original fixture and seed-42 feed.
For an exact-parity deployment, validate the choice on its actual graph/feed set.

## Two routes for BatchNorm arithmetic

The graph's MatMul uses an exact 3×3 identity matrix followed by a zero Add.
Farther downstream, `node_6 Conv → node_9 BatchNormalization` precedes the
final QuantizeLinear/DequantizeLinear pair.

The saved optimized graphs and the pinned v1.30 sources agree:

- [ConvBNFusion](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/optimizer/conv_bn_fusion.cc#L85-L98)
  computes `gamma/sqrt(var+epsilon)` and folds that scale and the BatchNorm
  bias into the preceding convolution's weights/bias.
- [NchwcTransformer](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/optimizer/nchwc_transformer.cc#L1055-L1173)
  rewrites a remaining BatchNorm on NCHWc inputs as a depthwise 1×1 convolution.
  Disabling ConvBNFusion alone leaves this second route active; the saved graph
  contains `t_9_bn_nchwc`, with the NCHWc convolution domain.
- The joint-disabled graph keeps two BatchNormalization nodes and the
  MatMulAddFusion Gemm. Its original final quantized values match the reference.

These controls retain the original FP32 BatchNorm evaluation route upstream
of the quantizer. Algebraically equivalent convolution parameter folding changes
the FP32 inputs that eventually meet a discrete rounding boundary.

## Actual FP32 inputs at the four boundaries

I appended the existing FLOAT32 output type/shape to `t_12`, immediately before
QuantizeLinear. All three instrumented routes—reference, full optimization and
joint-disabled—retain their corresponding original final outputs exactly.
The measured maximum pre-Q FP32 difference is **2.1457672119140625e-6**.

The quantizer has scale **0.03125**, UINT8 zero point **128**, and nearest-even
rounding. The four observed crossings are:

| Coordinate | Reference pre-Q bits | Optimized pre-Q bits | Boundary | UINT8 ref → all |
| :--- | :--- | :--- | ---: | :--- |
| (0,10,47,42) | 0x3f340004 | 0x3f33fffd | 0.703125 | 151 → 150 |
| (0,25,12,9) | 0xbf140002 | 0xbf13fffb | −0.578125 | 109 → 110 |
| (0,28,33,33) | 0x3f9e0000 | 0x3f9dfffb | 1.234375 | 168 → 167 |
| (0,32,0,53) | 0x3f640001 | 0x3f640000 | 0.890625 | 157 → 156 |

At `(0,28,33,33)`, the reference is exactly `1.234375 / 0.03125 = 39.5`.
Nearest-even chooses 40, then adds 128. The optimized value is slightly below
that midpoint and chooses 39. The resulting dequantized outputs are 1.25 and
1.21875. At the last coordinate, the optimized value lands exactly on the tie
and selects the even bin, while the reference lies just above it.

The [complete JSON](results/ort-qdq-fusion-control.json) records every route,
optimized node identity, FP32 value/bits and final UINT8/DQ pair. The checker
independently reconstructs all four quantized pairs from the observed pre-Q values.

## Reproduce the complete control

Use the original `minimal_model.onnx` from the report's
[upstream attachment](https://github.com/user-attachments/files/32810047/minimal_model.onnx.zip).
The model remains hosted by its author. In an environment with the pinned dependencies:

```sh
python -m pip install numpy==2.5.3 onnxruntime==1.30.0
python3 -B research/ort-qdq-fusion-control.py /path/to/minimal_model.onnx /path/to/temp/ort32918
```

[The script](ort-qdq-fusion-control.py) runs the six original-graph routes and
three endpoint-preserving pre-Q probes, then writes its small readout and
same-host optimized graphs into the supplied temporary directory. The small
protobuf rewrite is specific to this fixture's existing FLOAT32 shape;
other tensor types/models can use ONNX's ValueInfo helpers.

Original report and model: **kadu-v**. Independent Linux control, graph/source
inspection and reproduction: **Blucca / Codex**. The author's ARM64 observation
and this Linux x86_64 result retain their respective runtime scopes.
