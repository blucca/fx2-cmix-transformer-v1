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

## Scale decision

The frozen two-slice training halves contain 1,044,469 valid targets. Extrapolating this short-article rate gives **5.71 CPU hours per epoch**, before optimizer/entropy overhead and long-article effects. The longest q2 training piece has 35,163 tokens; only 22.9% of q2 training tokens belong to pieces no longer than 4,096 tokens.

The naive reference retains per-token KDA state and constructs dense attention masks. Full-data training is deferred. The next implementation target is a compact CPU KDA adjoint and exact bounded-window attention, keeping full article state and gradients; its usefulness is decided by measured full-model speed and memory before an entropy-constrained training run.
