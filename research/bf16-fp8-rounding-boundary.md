# A BF16 rounding boundary survives the final FP8 cast

An intermediate dtype is part of an inference transformation's numerical
contract. With round-to-nearest-even, this finite example produces:

| Conversion path | Final value |
| :--- | ---: |
| FP32 → BF16 → E4M3FN FP8 | **1.0** |
| FP32 → E4M3FN FP8 | **1.125** |

The input is the FP32 representation of `1.0626`, exactly
`1.062600016593933`. BF16 first rounds it to **1.0625**. In E4M3FN, the
adjacent values are 1.0 (`0x38`) and 1.125 (`0x39`), with midpoint 1.0625.
The midpoint rounds to the even code `0x38`; the original FP32 input is
above the midpoint and rounds to `0x39`.

Run the complete, standard-library check:

```sh
python3 -B research/check-bf16-fp8-boundary.py
```

Its scope is this finite positive rounding cell and round-to-nearest-even.
The local converter explicitly accepts `[1.0, 1.125]`. Device conversion,
other cells, overflow and whole-model equivalence have their own validation
scope.

## Applying the contract to a fusion

For a sequence `normalize in FP32 → store BF16 → cast FP8`, a fused kernel
can keep the BF16 value in registers and perform both conversions. The
FP32 → BF16 step stays observable in the final FP8 tensor.

RoPE adds another such boundary when it reads the BF16 normalized value,
rotates in FP32 and writes BF16 before the cache cast. Each intermediate
rounding belongs in the fused expression.

A concrete implementation to inspect is SGLang's pinned
[`FusedQKNorm`](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py):
it carries source-dtype round points through norm, optional RoPE and its
FP8 output path. Backend conversion policy also includes saturation,
scale and cache layout. Those contracts travel with a backend port.

Evidence here is the runnable numerical counterexample and the referenced
implementation. A performance claim requires the device path's measured
work and model validation.
