# Keep 1.5 as 1.5

**A BF16 OpenVINO example input becomes FP16 while passing through NumPy.**
This self-contained native check reproduces [Somay-kousis’s OpenVINO #39490](https://github.com/openvinotoolkit/openvino/issues/39490), preserves the declared type and byte payload, and tests the resulting CPU graph.

| Read the same OV BF16 tensor | Torch values | Model input | CPU `x * 2`, output read as f32 |
|---|---|---|---|
| Stock helper | `[1.9375, -2]` | f16 | — |
| Numeric cast to BF16 | `[1.9375, -2]` | bf16 | `[3.875, -4]` |
| Copy int16 bytes, view as BF16 | `[1.5, -2]` | bf16 | `[3, -4]` |

**Check both the logical dtype and expected values.** BF16 `1.5` carries `0x3fc0`; the NumPy f16 view reads that payload as `1.9375`. A numeric cast to BF16 stores the changed value (`0x3ff8`). A typed-byte copy keeps `0x3fc0`.

## Small fix, native evidence

The OV Tensor branch in [`helper-bf16.patch`](helper-bf16.patch) adds:

```python
if tensor.element_type == Type.bf16:
    return torch.tensor(tensor.data.view(np.int16)).view(torch.bfloat16)
```

`torch.tensor` retains the existing copy semantics. The same shared helper handles tuple/list/dict leaves. Direct Torch BF16, ordinary OV FP16, source mutation and one seven-pattern byte check (finite values, signed zeros, infinities and NaN) pass in the [recorded result](results/result.json).

Measured: **Linux x86-64 / EPYC 9254 VM, Python 3.12.3, OpenVINO 2026.4.1-22982, Torch 2.9.0+cpu, NumPy 2.5.4**; two CPU threads and one stream. This is the example-input Python boundary. CPU outputs use an explicit graph f32 Convert, separating it from the BF16 output NumPy-view boundary.

## Run the complete check

Download [`repro_bf16.py`](repro_bf16.py) into a writable scratch directory, then:

```sh
python3 -m venv venv
venv/bin/python -m pip install --only-binary=:all: \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.0'
venv/bin/python -m pip install --only-binary=:all: \
  --index-url https://pypi.org/simple 'openvino==2026.4.1' 'numpy==2.5.4'
venv/bin/python -u repro_bf16.py
```

The script checks the reported stock behavior, a numeric-cast counterexample and an in-process helper overlay; it writes `result.json`. It runs the tiny `M(x) = x * 2` graph with copied typed inputs. The installed package file remains unchanged. Model weights and C++ builds are outside this reproduction.

## Sources and recorded attempts

The installed-wheel and fetched-master helper functions have identical ASTs, as do the tested overlay and source patch ([comparison](source-comparison.json), [source](source-provenance.json)). Logs preserve the [final PASS](logs/final.stdout.log) and the first attempt: its CPU-output adapter connected a nested Result incorrectly; using `Result.input_value(0)` completes the native check. Numeric outputs are original; the environment path in the public JSON is normalized.

Original fault/reproducer: **Somay-kousis**. Independent Linux test, controls and helper patch: **Blucca / Codex**. This directory and the OpenVINO patch use [Apache-2.0](LICENSE).

Have a CPU runtime mismatch with a deployment deadline? [US$150 two-configuration diagnosis](../../#us150-cpu-parity-diagnosis).
