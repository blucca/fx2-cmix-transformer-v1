# Full-model CPU backward: a bounded feasibility result

The unchanged 6M FP32 QAT checkpoint can backpropagate on CPU using the repository's existing PyTorch KDA, causal-convolution and gated-norm references, plus its SDPA implementation. The production model and inference code are unchanged.

## Measured result

One complete prepared-stream article, `[11700, 13047)`, provides **1,346 next-token targets** and exercises the original **1,024-token window**. Priors retain the original alignment: row `g` predicts token `g+1`. The configuration comes from `training_recipes/cooldown.py`, with SDPA replacing the GPU-specific attention implementation.

| Measurement | Result |
| --- | ---: |
| CPU / runtime | One thread, PyTorch 2.6.0+debian |
| Trainable parameters / tensors | 5,923,228 / 429 |
| Forward CPU time | 9.961 s |
| Backward CPU time | 16.533 s |
| Forward + backward throughput | 50.80 targets / CPU s |
| Peak RSS | 1396.6 MiB |
| Finite, nonzero parameter gradients | 429 / 429 |
| Loss / targets | 0.992707614 nats |
| Optimizer updates | 0 |

[Exact result](results/cpu-backward-probe.json). Gradient checkpointing is active across all 12 blocks; the nine KDA blocks are recomputed during backward. The 429 checkpoint tensors and 434 serialized inference payloads use different formats: inference also carries its derived tables.

This establishes a CPU gradient path. Model compression and rate–distortion improvements require actual training and exported-weight evaluation.

## Reproduce

Use a CPU PyTorch environment with NumPy and `jaxtyping`. No FLA or FlashAttention installation is needed. Provide existing captured token IDs, matching 205-way float16 priors, and int32 article boundaries from the prepared stream.

```sh
python -B research/cpu-backward-probe.py \
  --checkpoint models/6m-q4-fp32.tch \
  --tokens /path/to/tokens.u8 --priors /path/to/priors.f16 \
  --boundaries /path/to/boundaries.i32 \
  --output /path/to/temp/cpu-backward.json
```

The script imports the original three references through local adapters, extracts the existing cooldown configuration, checks the full gradient set, and records bounded CPU/memory use. The first setup attempt exposed an AST-selection error: the later `raw_model.to(device)` assignment replaced the constructor expression. Keeping the first assignment fixes that selection; the measured run above includes the corrected script.

## Reference-only scale decision

The frozen two-slice training halves contain 1,044,469 valid targets. Extrapolating this short-article rate gives **5.71 CPU hours per epoch**, before optimizer/entropy overhead and long-article effects. The longest q2 training piece has 35,163 tokens; only 22.9% of q2 training tokens belong to pieces no longer than 4,096 tokens.

The naive reference retains per-token KDA state and constructs dense attention masks. Full-data training is deferred. The compact backend below now passes the full-article resource and numerical gates; entropy-constrained learning is the next experiment.


## Compact CPU training backend

`cpu-kda.cpp` and `cpu_ops.py` provide an opt-in CPU training path. The
production model, hard-QAT forward, article resets and 1,024-token context
stay intact. The [paired measurements](results/cpu-compact-backward.json)
include the original checkpoint, all parameter gradients and a complete
35,162-target article.

| Measurement | 1,346 targets | 35,162 targets |
| --- | ---: | ---: |
| Forward + backward, CPU s | 1.828 | 61.871 |
| Peak RSS, GiB | 0.643 | 5.277 |
| Finite, nonzero gradients | 429 / 429 | 429 / 429 |

The small paired run has identical NLL and **3.66e-7 relative L2 gradient
error** against the same-runtime reference. It is **6.57× faster** than that
reference and **14.49× faster** than the original reference-BLAS measurement.
Both runs enforce a 6 GiB address-space limit. Optimizer updates remain zero.

- **KDA:** save the state before every 128th token and recompute one block
  during backward. The state adjoint crosses every block boundary. This
  retains full-article BPTT with a small checkpoint array and a 128-state
  work buffer. Forward reductions follow the reference's groups of 16.
- **Attention:** tiled queries and local key ranges retain the exact causal,
  document and window masks. A 3-D mask keeps the original CPU math-SDPA
  scaling order. Changing to CPU flash produced small operator differences
  that crossed subsequent hard-QAT thresholds in the full model.
- **QAT backward:** reuse the original function over bounded groups of
  examples, restoring its global square-root scale-gradient normalization.
  Quantized forward values and the input adjoint are preserved.

An initial profile found 75.6% of compact-backend CPU time in matrix
multiplication with Debian's reference BLAS. The paired compact measurements
use single-thread OpenBLAS; their reference comparison uses the same BLAS.
The original production C++ coder remains unchanged.

```sh
mkdir -p /YOUR/temp/cpu-kda
clang++-17 -O3 -mavx2 -ffp-contract=off -shared -fPIC \
  research/cpu-kda.cpp -o /YOUR/temp/cpu-kda/cpu-kda.so
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -B research/cpu_ops.py /YOUR/temp/cpu-kda/cpu-kda.so
# Add this option to the probe command above:
# --kernel /YOUR/temp/cpu-kda/cpu-kda.so
# Save a reference gradient set with --save-gradients FILE.pt;
# compare it in the compact run with --compare-gradients FILE.pt.
```

The operator check covers article resets, 128-token recomputation boundaries,
attention window/tile boundaries and the large-activation quantizer adjoint.
Full-model comparisons use the same checkpoint, input and CPU runtime.
