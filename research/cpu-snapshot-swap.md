# Capture the values before the writes

A small CPU swap exposes two separate parts of a compiled program's contract:
the values it returns and the storage it changes. Both deserve a reference check.

[Peter Sharpe reported a real `torch._foreach_copy_` compiler fault](https://github.com/pytorch/pytorch/issues/203806)
while working on TensorDict. His application uses a
[materialized stack buffer](https://github.com/pytorch/tensordict/pull/3521) as a
workaround. A [proposed upstream repair](https://github.com/pytorch/pytorch/pull/203874)
realizes cross-read sources before mutating their destinations.

I measured the original fault and two application-level controls on the released
**PyTorch 2.15.0+cpu**, commit `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`.
The CPU reports **AVX512**. All runtime and compiler thread counts are one;
compilation uses `backend="inductor", fullgraph=True, dynamic=False`.

## The snapshot is part of the operation

The intended operation is a simultaneous value swap of two independent,
contiguous FP32 CPU tensors. Capture both old values before the first write:

```python
def explicit_snapshot(a, b):
    sources = (b.clone(), a.clone())
    for dest, source in zip((a, b), sources):
        dest.copy_(source)
    return a, b
```

The other two controls replace the loop:

```python
# Original reported fault
sources = (b.clone(), a.clone())
torch._foreach_copy_([a, b], list(sources))

# Standalone stack-buffer control
sources = torch.stack((b, a))
torch._foreach_copy_([a, b], list(sources.unbind()))
```

For each call I preserve independent expected values and pre-call views of the
original inputs. The check covers six value coordinates: two returns, two inputs,
and two external views. It also checks that the original storage addresses survive.

Inputs are exactly representable FP32 integers: `a = arange(N)` and
`b = a + 2**21`. The two tested lengths are **3** and **1,048,576**. This is a
single-leaf, two-tensor, contiguous, no-grad swap experiment.

## Actual values and actual writes

| Compiled control | N=3 | N=1,048,576 |
| :--- | :--- | :--- |
| Foreach + upfront clones | Three wrong elements in return `b`, input `b` and its external view | Stopped at the small correctness gate |
| Explicit copies + upfront clones | All six value coordinates exact; storage preserved | All six exact; storage preserved |
| Stack + foreach copies | All six value coordinates exact; storage preserved | All six exact; storage preserved |

The small foreach kernel loads a single source and stores it to both destinations:

```cpp
auto tmp0 = in_ptr0[x0];
out_ptr0[x0] = tmp0;
out_ptr1[x0] = tmp0;
```

The generated wrapper binds this source to the original `b` buffer. The first
input gets its expected value; the second loses its old counterpart. The eager
versions of all three controls pass the same small snapshot contract.

## Correctness has a measurable cost

At **N=1,048,576**, each FP32 tensor occupies 4 MiB. Generated wrappers and C++
show exactly what the two correct controls keep:

| Correct control | Runtime scratch allocation | Full-length loops in generated C++ | Compiled entrypoint calls | Median warm call |
| :--- | ---: | ---: | ---: | ---: |
| Explicit copies | **4 MiB** | **3** | 1 | **1.132860 ms** |
| Stack buffer | **8 MiB** | **4** | 1 | **1.647245 ms** |

The explicit-copy wrapper allocates one N-element buffer. Its compiled body
saves one old input, copies the other over it, then restores the saved values
into the other destination. The stack wrapper allocates a 2N-element buffer,
builds both snapshots, then copies each half back.

Both paths have one compiled entrypoint. Their scratch allocations and full-length
loops remain visible inside the generated code. This is the useful unit of comparison
for a memory-bound repair.

The explicit-copy control used **half the scratch** and had **31.23% lower
median latency** in this run. Timing is a synchronous, warm, one-thread CPU call:
two warmups followed by 20 samples. Input resets, verification and compilation
are outside the timed interval. The raw samples and generated wrappers are kept
with the result. Application metadata, other layouts, devices and real TensorDict
end-to-end performance have their own acceptance contracts.

## Reproduce the measurement

Use the official CPU `torch==2.15.0+cpu` package and a working C++ compiler with
matching Torch build headers. The script asserts the version and the eager
reference, stops failed controls before larger compilation/timing, and saves
compiled source:

```sh
SNAPSHOT_OUTPUT=temp/snapshot-swap \
TORCHINDUCTOR_CACHE_DIR=temp/snapshot-swap/cache \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
python3 -B research/check-inductor-snapshot-swap.py
```

- [Complete script](check-inductor-snapshot-swap.py)
- [Five measured cases and every timing sample](results/snapshot-swap-cpu/result.json)
- [Wrapper allocation and loop-count proof](results/snapshot-swap-cpu/allocation-proof.json)
- Generated [fault wrapper](results/snapshot-swap-cpu/kernels/foreach_snapshot-3/1.py),
  [explicit-copy wrapper](results/snapshot-swap-cpu/kernels/explicit_snapshot-1048576/0.py)
  and [stack wrapper](results/snapshot-swap-cpu/kernels/stack_snapshot-1048576/0.py)

The original report, application workaround and upstream fix belong to their
linked authors. This is an independent CPU acceptance and cost measurement,
recorded November 8, 2026 by Blucca / Codex. It complements our
[compiler rounding checks](bf16-fp8-inductor-cpu.md): preserve the agreed values,
then measure the physical implementation that preserves them.
