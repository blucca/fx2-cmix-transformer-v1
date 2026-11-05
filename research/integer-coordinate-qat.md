# Fixed-scale integer-coordinate CPU QAT

[`integer-coordinate-qat.py`](integer-coordinate-qat.py) learns only the
111 integer-coordinate matrices. It reuses [`cpu_model.py`](cpu_model.py),
the original whole-piece batching, [`cpu_weights_export.py`](cpu_weights_export.py),
the static15 recoder and the native evaluator. The original weights remain
the production baseline.

## Frozen method

Initialize FP32 coordinates with the original hard symbols:

```
z0 = round(original_weight / original_signed_BF16_scale).clamp(-7, 7)
forward_weight = STE(round(z)) * original_signed_BF16_scale
J = CE_mean + KL(original_W_distribution || student_distribution)_mean
    + 2 * sum(per_matrix_soft_entropy_nats(z)) / 586459020
```

The original BF16 weight scales, activation scales and every other model
parameter stay fixed, including the original signed 6–8e-15 Q/K scales.
Inference retains the original −7…7 symbols, signed scales, quantized
activations, 12 layers and 1024-token window. Entropy acts directly on bounded
`z`, using centers −7…7 and fixed temperature **0.5**.

Projected AdamW updates only `z`: **lr .01**, betas **(.9, .99)**, epsilon
**1e-8**, zero decay; each update clips coordinates to **[-7,7]**. The original
W teacher is frozen. Its FP32 logits are cached per complete piece with the
same original PPMd priors and position reset; cache storage is about
**820 B per training target**. CE and KL sums accumulate across whole pieces
until at least **32,768 targets**, then divide by the actual batch target count.
Each piece retains its complete gradient.

The original 429-key model state stays fixed for direct tensor and BF16-bit
audits. Export uses `export_state(model,z,scales)` to place hard integer-times-
signed-scale weights into the existing payload exporter. The recoder retains
all five original derived tensors. Checkpoints store `z`, AdamW, RNG and the
epoch/piece cursor; `--resume` reloads the original checkpoint for the fixed
parameters.

## Actual preflight, 2026-11-06

- Full zero-step static15: **2,904,325 B**, byte-identical to the original.
- Original complete piece **[11700,13047), 1346 targets**: student/teacher
  logits identical; CE **0.992835488397**, teacher KL **0**.
- **111/111** finite, nonzero gradient tensors; norm **0.041474457253**.
- One update changes **5,610,626** fractional coordinates, max |Δz|
  **0.00999999046**. Original **429** tensors and every scale's BF16 bits stay
  identical. All eight tiny scale rows retain their signed values.
- Hard symbols stay unchanged on this first .01 update; hard range
  **[-7,7]** and the post-update static15 bytes equal the zero-step file.
- New check CPU **14.58904 s**, peak RSS **705,432 KiB**. The historical
  original native baseline cost **82.945466 CPU s** is recorded separately.
  The check restores initial `z`/Adam/RNG and leaves a ready step-zero checkpoint.

The first raw receipt displayed post-update entropy under the same field as
the objective's pre-update entropy. The final source records these separately:
`soft_entropy_nats` for the objective, `current_soft_entropy_nats` for geometry.
The original raw preflight retains its measured values: **15,413,912.224609375**
before the update, **15,414,186.32421875** after it.

## Complete budget and coverage

One fixed trajectory: **64 epochs / 48 CPU hours maximum**, one CPU thread,
**7 GiB address-space limit**, including export/native scoring. Checkpoints
occur every eight updates and at epoch boundaries. `STOP` requests a clean
batch-boundary save; failures retain the last saved checkpoint and stop.
Every epoch measures real static15 bytes and native held-out NLL.

The initial learning manifest covers **807 complete pieces / 1,044,469 targets**
from the first halves of the prefix and independent q2 1 MiB samples. The
full method also includes mature PPMd sample evaluation, real mixed-coder
propagation and complete-package adjudication. These remain the quality and
compression decision stages following learning.

`--check-only` runs the full zero-step export and one real small-piece update,
then restores zero. `--zero-only` exports and saves zero. The CLI accepts the
existing checkpoint/kernel/manifest/native baseline/v2 template/recoder/evaluator
paths and an output directory; `python3 -B research/integer-coordinate-qat.py --help`
lists the interface.

## Mature validation and first committed hard change

[The mature-prior measurement](mature-ppmd.md) is complete. The same text
with causal PPMd history improves the original NN by .0400393 nats/target
versus fresh priors. The first epoch retains every hard symbol and all native
scores; epoch 2 changes 3,575 symbols, grows W by 12 B, and changes the two
cold held-out NLLs by +.00412299 / −.00169225. Its independent mature NLL
increases by .00590677. All original scales remain fixed. The single 64-epoch /
48-CPU-hour trajectory continues; the online mixed pair follows its exit.
[Full machine-readable receipts](results/integer-coordinate-mature-ppmd.json).
