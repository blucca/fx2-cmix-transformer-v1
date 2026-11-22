# Direct packed-initializer Q8 M1 operator

`org.blucca::Q8MatMulM1Packed(A, P, PS) -> Y`, CPUExecutionProvider,
ORT **1.31.1 / API31**, Linux x64 AVX2 + FMA. The independent library is
`libblucca_q8_m1_packed.so`; the earlier owner kernel stays unchanged.

| Item | Fixed contract |
| --- | --- |
| Attributes | lowercase INT `k`, `n`; `k % 32 == 0`, `n % 16 == 0` |
| A | FP32 singleton leading dimensions, K trailing; M=1 |
| P | constant INT8 `[N/16, K/32, 32, 16]` |
| PS | constant FP32 `[N/16, K/32, 16]`, finite positive scales |
| Y | FP32, A leading dimensions with N trailing |

The fixed graph supplies `[1,K] -> [1,N]` for 25 operators, including the
padded 50,272-wide output head and its original 50,257-logit Slice.

```text
P[t,b,i,l] = int8(int(B[t*16+l,b,i]) - 128)
PS[t,b,l]  = S[(t*16+l)*(K/32) + b]
```

Kernel state keeps the API reference and K/N/KB/T scalars. Constructor checks
borrowed constant inputs and releases those local views. Each Compute
obtains A/P/PS from its current context and creates a stack-local borrowed
view. Native code/scale array ownership is **0 B**, down from **91,210,752 B**
for the old complete fixed graph. ORT retains its initializer tensors.
`Run` and `FmaPair` have the exact original operation/accumulation order after
substituting the two vector `.data()` expressions with borrowed pointers.

## Build and optional small fixture

From repository root:

```bash
bash research/ort-q8-single-copy-packed-input/native/build.sh
bash research/ort-q8-single-copy-packed-input/native/check_packed.sh
```

Both scripts use the adjacent unchanged `ort-q8-m1-native` SDK/core by
default. A standalone corresponding-source directory has the same layout;
`SDK=/path/to/sdk-1.31.1`, `OLD_CORE=/path/to/old/core`, `CXX=g++` and the
builder's first output argument provide explicit path overrides.
Flags stay `-O3 -mavx2 -mfma -shared -fPIC -std=c++17`; compilers and small
fixtures run on one allowed CPU with 512 MiB address-space limits.
The public wrappers change staging paths and receipt metadata only;
C++ sources and compilation flags match the measured target binary.

The preserved small fixture compares twelve runs against the original
owner bit-for-bit: signed codes, distinct tile/block/lane scales, offset
storage and padded output lanes. Saved source/build receipts live in
[`../results/source/native/`](../results/source/native/). The actual target
GCC16 build produced **43,056 B** in **2.750 s**, peak child RSS **343,184 KiB**.

## Diagnostics and measured scope

```c
uint64_t BluccaQ8M1PackedCtorCount(void);
uint64_t BluccaQ8M1PackedOwnedPackedBytes(void);
```

Constructor count is cumulative in the loaded library; the complete model
adds 25. OwnedPackedBytes reports native code/scale payload, always zero for
this implementation. Whole fresh-process RSS is measured separately with
`getrusage(RUSAGE_SELF).ru_maxrss`.

The complete target three-cache comparison passes all thirteen outputs,
whole-call latency and peak RSS gates. Fixed NIST2048 independent chains and
four 64-token greedy continuations also pass. See the
[full study and acceptance](../README.md); this small fixture has its own
source/arithmetic scope.

New native source is MIT, with the same six MIT-licensed ORT API31 headers
supplied by the adjacent original source. The surrounding study is GPL-3.0.
