# LayerNorm: compare the CPU runtime with its row-wise moments

A native AVX2 probe for [PyTorch #204610](https://github.com/pytorch/pytorch/issues/204610). The report describes a Windows XPU build whose CPU LayerNorm reference differs by about 0.003–0.005 at large normalized sizes, including float64.

The reported thresholds line up with a change from `RowwiseMomentsImpl<T,4>` to `<T,8>`: 2,048 values for AVX2 float64, 4,096 for AVX2 float32. This probe separates that template boundary from the full runtime and build environment. The first complete measurement finds a Windows native-header variance discrepancy at the same boundary, while stock CPU 2.15.0 LayerNorm stays accurate on both systems. [Linux records](baseline-linux.json) and [Windows records](baseline-windows.json) preserve that result. A Windows-only no-inline control is running.

## Run

Use Python 3.12, a C++20 compiler and an x86-64 AVX2/FMA CPU. On Windows run from a Visual Studio x64 developer shell.

```sh
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.5.4
python research/layernorm-rowwise-moments/run.py --out temp/layernorm-moments
```

The [manual Linux/Windows Action](../../.github/workflows/layernorm-rowwise-moments.yml) runs these commands and uploads small records, compiler logs and the installed moments header.

- Runtime: the original seed-0 `4 × N` input, four normalized sizes, float32/float64 and separate `avx2`/`default` processes. A NumPy float64 reference checks the reference used by the original script.
- Native: exact binary input `((i * 37 % 1009) - 504) / 256`; auto selection and explicit safe `<4>`, `<8>`, `<16>` specializations, with independent scalar mean/population-variance sums.
- Outputs: `results.json`, two runtime JSON files, `build.json`, logs and `installed-moments_utils.h`. Runtime inaccuracies are recorded as observations; successful execution means the measurements completed.

On Windows the program also compiles a separate header copy with only `AddMomentsVec` changed from `C10_ALWAYS_INLINE` to `__declspec(noinline)`. Both runs use the same input, compiler flags and safe specializations. The installed header stays unchanged.

This compares stock CPU 2.15.0 and locally compiled installed headers. The original report uses a self-built Windows XPU commit; reproduction on that environment is a separate observation. The program supports the two declared floating-point types and four sizes; it checks the AVX2 vector widths at build time.

The original author already reports `ATEN_CPU_CAPABILITY=default` as a workaround. Their XPU test-reference and skip PRs stay separate from this CPU diagnostic. PyTorch's own source and licenses remain in the installed distribution.
