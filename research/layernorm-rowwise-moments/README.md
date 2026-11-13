# LayerNorm: compare the CPU runtime with its row-wise moments

A native AVX2 probe for [PyTorch #204610](https://github.com/pytorch/pytorch/issues/204610). The report describes a Windows XPU build whose CPU LayerNorm reference differs by about 0.003–0.005 at large normalized sizes, including float64.

The reported thresholds line up with a change from `RowwiseMomentsImpl<T,4>` to `<T,8>`: 2,048 values for AVX2 float64, 4,096 for AVX2 float32. This probe separates that template boundary from the full runtime and build environment. The first complete measurement finds a Windows native-header variance discrepancy at the same boundary, while stock CPU 2.15.0 LayerNorm stays accurate on both systems. [Linux records](baseline-linux.json) and [Windows records](baseline-windows.json) preserve that result. The [Windows no-inline control](noinline-windows.json) restores all eight rows and every safe specialization to the declared reference tolerance.

## Observed result

[Complete Linux/Windows run](https://github.com/blucca/fx2-cmix-transformer-v1/actions/runs/43482499624) and [single Windows control run](https://github.com/blucca/fx2-cmix-transformer-v1/actions/runs/43484480470), both completed successfully. Native Windows uses the runner's MSVC 14.51 toolchain with `/O2 /arch:AVX2 /std:c++20`; Linux uses `-O3 -mavx2 -mfma -std=c++20`.

| Type / N | Original mean / variance error | No-inline mean / variance error |
| :--- | ---: | ---: |
| float32 / 1024 | 0 / 9.07e-08 | 0 / 9.07e-08 |
| float64 / 1024 | 3.47e-18 / 0 | 3.47e-18 / 0 |
| float32 / 2048 | 2.33e-10 / 6.09e-08 | 2.33e-10 / 6.09e-08 |
| float64 / 2048 | 2.6e-18 / 0.000108 | 2.6e-18 / 0 |
| float32 / 4096 | 0 / 3.64e-05 | 0 / 4.36e-08 |
| float64 / 4096 | 4.34e-19 / 0.000102 | 4.34e-19 / 0 |
| float32 / 175616 | 1.68e-05 / 5.21e-05 | 5.68e-10 / 8.55e-08 |
| float64 / 175616 | 0.000108 / 0.000106 | 3.33e-18 / 2.22e-16 |

The same explicit `<8>` and `<16>` specializations work at the small sizes and fail once depth reaches five on Windows. Changing only `AddMomentsVec` from forced-inline to no-inline restores the native results: mean/variance error at most about `5.7e-10 / 9.1e-8` for float32 and `3.5e-18 / 2.2e-16` for float64. This is an isolated build-control result. The original author's self-built XPU runtime can run this probe with its own headers and compiler to check the corresponding environment.

Stock **CPU 2.15.0** stays accurate in all **32 runtime observations** on these Windows/Linux hosts: the original seed-0 LayerNorm cases, two capability settings and two dtypes. Its Windows binary records MSVC 194234444; the standalone build is a different toolchain. The installed moments algorithm matches the reported commit after removing the package's outer stable-API guard and normalizing line endings.

```sh
python research/layernorm-rowwise-moments/verify.py
```

## Run

[Download the small source-and-record bundle](https://github.com/blucca/fx2-cmix-transformer-v1/releases/download/layernorm-moments-v0.1.0/layernorm-moments-v0.1.0.zip). Extract it and run `python run.py --out temp/layernorm-moments` from an existing PyTorch environment and C++20 developer shell. The program reads that environment's headers and leaves the installation unchanged.


Use Python 3.12, a C++20 compiler and an x86-64 AVX2/FMA CPU. On Windows run from a Visual Studio x64 developer shell.

```sh
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.5.4
python research/layernorm-rowwise-moments/run.py --out temp/layernorm-moments
```

The [manual Linux/Windows Action](../../.github/workflows/layernorm-rowwise-moments.yml) runs these commands and uploads small records, compiler logs and the installed moments header.

- Runtime: the original seed-0 `4 × N` input, four normalized sizes, float32/float64 and separate `avx2`/`default` processes. A NumPy float64 reference checks the reference used by the original script.
- Native: exact binary input `((i * 37 % 1009) - 504) / 256`; auto selection and explicit safe `<4>`, `<8>`, `<16>` specializations, with independent scalar mean/population-variance sums.
- Outputs: `results.json`, two runtime JSON files, original/no-inline build records and logs, and `installed-moments_utils.h`. Runtime inaccuracies are recorded as observations; successful execution means the measurements completed.

On Windows the program also compiles a separate header copy with only `AddMomentsVec` changed from `C10_ALWAYS_INLINE` to `__declspec(noinline)`. Both runs use the same input, compiler flags and safe specializations. The installed header stays unchanged.

This compares stock CPU 2.15.0 and locally compiled installed headers. The original report uses a self-built Windows XPU commit; reproduction on that environment is a separate observation. The program supports the two declared floating-point types and four sizes; it checks the AVX2 vector widths at build time.

The original author already reports `ATEN_CPU_CAPABILITY=default` as a workaround. Their XPU test-reference and skip PRs stay separate from this CPU diagnostic. PyTorch's own source and licenses remain in the installed distribution.

License: GPL-3.0 for this probe; PyTorch headers retain their original license in the installed distribution.
