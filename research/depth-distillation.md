# Eight-block causal distillation

The fixed eight-block native family has compiled successfully with the original
v22 production/trace flags: 20 commands, 19.289341 CPU seconds, single CPU,
1 GiB build address-space limit, and four native executables. The
[complete build receipt](results/depth-native-build.json) records the actual
sources, flags and terminal result. Actual student construction, tensor export,
gradient updates and codec experiments are **0** at this publication.

## One student, one complete method

Keep original blocks **0,1,2,3,8,9,10,11** and renumber continuously. This gives
six KDA blocks, two attention blocks, 192 hidden dimensions, 768 MLP dimensions,
three 64-wide heads and the original 1024 attention window. Four retained skip
pairs use the original destination weights at indices **2,3,4,5**.

The genuine student schema is 289 state tensors: 75 INT4 matrices, 75 weight
scales, 80 activation scales and 59 raw F32 tensors. Its complete weight file
adds the three original RoPE tensors and two eight-block configuration tensors.
[CPU construction/export](depth_student.py) strictly maps original keys/shapes,
keeps the three original RoPE arrays, writes a genuine eight-block v2 template,
and passes it through the existing lossless static15 recoder. The original
12-block template/replacement path stays fixed.

The [original histogram estimate](depth-distillation-contract.md) suggests about
1.816 MB reduction in two weight copies. Training supplies the actual weight
file and prediction-loss tradeoff. Final score uses each side's executable and
weight size: `S=P+2B+2D+O+2W+69`.

## Frozen learning recipe

Reuse the complete-piece, article-cluster-separated causal inputs from the
[whole-stream prior method](causal-priors.md). Teacher inference uses the
original twelve-block checkpoint with identical priors/history and per-piece
state resets. Cache all 205 logits and the final 192-dimensional RMS feature.
`compute_last_activation` and `last_activation_to_logits` provide both values
through the original forward path.

The [trainer](depth-distillation.py) learns 75 bounded integer-coordinate
matrices and the 59 retained raw F32 tensors. Every signed BF16 row scale and
activation scale stays fixed. Epochs 1–8 use continuous `z*scale`; epochs 9–32
use the original straight-through rounded/clamped INT4 coordinates. Coordinate
learning rates are 0.03 then 0.01; raw F32 rates are 0.0003 then 0.0001.
AdamW uses beta=(0.9,0.99), epsilon=1e-8 and weight decay=0.

For a target-weighted complete batch, the objective is

```
CE + KL(teacher || student) + 0.1 * mean_d((h_student-h_teacher)^2)
   + 2 * soft_H(z) / 586459020
```

Whole-piece sums are divided by the batch target count; the short final batch
merges into its predecessor. Gradient graphs are freed piece by piece. Hard
static15 exports and full held/long/native-length scores are produced every
four epochs. One single trajectory uses at most 48 CPU hours, including its
teacher cache, exports, native scoring and completed preparation costs. Each
process has one CPU, numerical-library threads=1 and a 7 GiB training AS limit.
Checkpoints retain coordinates, raw tensors, optimizer, RNG and epoch cursor.

## Native identity and actual measurement

`FX2_MODEL_LAYERS` accepts the two fixed families, 8 and 12; its default is 12.
Both native implementations share the new shape contract. Existing matrix,
KDA, attention, PPMd and range arithmetic stays fixed. Build every new eight-block
object in its own directory:

```sh
python3 research/depth-native-build.py --out "$SCRATCH" --depth 8 \
  --flags-from "$ORIGINAL_PRODUCTION_BUILD_JSON"
```

The existing RAW `test_xcheck` compares reference and optimized inference. The
student's actual tensor export, a complete-piece gradient check and the genuine
causal training trajectory are the next measurements. Uniform short inputs
serve the shape/gradient contract; causal inputs supply quality evidence.

The baseline uses its original twelve-block probe; the student uses the fixed
eight-block probe. Select one hard checkpoint from the complete held/export
results, then measure the actual mature interval, fresh transfer interval and
full payload. Charge `2*(B0-B1)` and `2*(W0-W1)` once per paired score. A useful
actual result proceeds to the complete native archive path and restored 1 GB
original; actual package sizes and the latest record determine the final score.
