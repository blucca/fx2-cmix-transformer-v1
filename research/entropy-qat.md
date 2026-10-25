# Fixed entropy-constrained CPU QAT

This continuation trains the original 5,923,228 FP32 parameters, retaining
15 weight values, BF16 scales, 256 activation levels, 12 layers and the
1024-token attention window. Production inference is unchanged.

## Frozen objective and data

For each of the 111 quantized matrices, assign `w / bf16(scale)` softly to
integer centers −7…7 with `softmax(-distance² / tau)`. Average assignments
within that matrix to obtain its symbol probabilities and entropy `H_t`.
The implementation cancels the shared `-z²` term and evaluates
`softmax((2*z*center - center²)/tau)`. Original learned scales reach 6e-15,
so some saturated coordinates have |z| above 1e12. Direct FP32 squared
distances lose the center offsets at that scale. The loss is:

```
CE_sum / actual_batch_targets + 2 * sum(H_t in nats) / 586459020
```

The factor of two charges the two packaged copies of the weights. Scale
gradients use the original weight-QAT LSQ gain `1/sqrt(numel*7)`, with an
FP32 straight-through gradient through the BF16-rounded scale.

- Training: **807 complete pieces / 1,044,469 targets**, taken from the first
  halves of the existing prefix and independent q2 1 MiB samples. Pieces
  crossing the split and final incomplete pieces are excluded.
- Each piece retains its full gradient, original PPMd priors and position
  reset. Shuffle seed is `245 + epoch`; accumulate whole pieces until at
  least 32,768 targets, then normalize by the actual accumulated count.
- AdamW: learning rate **5e-5**, betas **(.9, .99)**, epsilon **1e-8**, zero
  weight decay, global gradient-norm cap **1.0**.
- Eight epochs maximum, with `tau = .5 * (.15/.5)**(epoch/7)`.
  Total CPU budget is **12 hours**, including preparation, exports and
  native evaluation; the address-space cap is **6 GiB**, using one CPU thread.

The fixed held-out pieces contain 523,194 / 516,391 targets. Original native
raw NLL is **0.485906618297 / 0.939429783007 nats/target**. At each epoch end,
the real static15 package is measured and the native evaluator scores both
halves. The next-stage gate is **at least 1,500,000 B saved across both weight
copies**, with each raw NLL increase at most **0.004**. Passing this screen
triggers a separate real mixed-coder decision.

## Export and continuation

[`cpu_model.py`](cpu_model.py) shares the exact CPU loading path with the
[bounded backward probe](cpu-backward.md). [`cpu_weights_export.py`](cpu_weights_export.py)
uses the original INT15/BF16/FP32 export arithmetic for all 429 learned
payloads. `weights-recode --replace-payloads` preserves the original five
derived tensors, metadata order and encoding choices, then checks all 434
decoded tensors against their expected payloads.

The real CPU **zero-step export is byte-identical to the original static15
file: 2,904,325 B**. A native 64 KiB pair covers 65,439 targets with identical
NLL totals, samples and output counts; only the load-time weight decoder was
recompiled, with original neural-kernel objects reused.

Build the recoder with assertions enabled:

```sh
c++ -O2 -std=c++17 research/weights-recode.cpp cpp_infer/src/weights_io.cpp \
  -o "$SCRATCH/recode"
python3 -B research/entropy-qat.py --help
```

The training CLI accepts the original checkpoint, compiled CPU kernel,
frozen piece manifest, completed native baseline receipt, original v2
weight template, recoder and native evaluator. It checks zero-step export
before any update. `--zero-only` stops after that check; `--resume` continues
from the atomically replaced `latest.pt` (model, optimizer, RNG, epoch and
piece offset). Checkpoints occur every eight updates and around epoch scoring.
Creating `STOP` in the output directory requests a clean batch-boundary stop.
After an unclean process exit, `--recovery-cpu-seconds` explicitly accounts
for CPU since the last status write. Finite gradients, valid scales and
successful export are required throughout; failures retain the last good
checkpoint and stop the process.

The first batch stopped before its optimizer update: the direct squared-distance
implementation produced spurious scale gradients up to 4.68e22 and overflowed
the FP32 gradient norm. Cancelling the common term preserves the mathematical
objective and restores saturated endpoint assignments; the original failure is
retained in the [preflight result](results/entropy-qat-preflight.json). A regression check covers both ordinary
coordinates and the actual tiny-scale regime.

This is one trajectory with fixed learning rate, temperature and quantization.
Its measured learning and native-score results follow the run; the zero-step
check itself leaves the model and compression result unchanged.

## First 16 updates

All 429 gradient tensors and scale checks passed through **595,718 targets**
in **804.47 CPU seconds** (740.5 targets/CPU-second). Peak RSS was
**5.190 GiB**, within the 6 GiB address-space limit;
the model, AdamW moments, RNG and next piece were saved at updates 8 and 16.

**1,372,455 / 5,868,864 integer weight symbols changed.**
Hard weight entropy rose by **27,411.67 nats**
(0.178%). This is the current adverse
rate signal; new compressed weight sizes and held-out native NLL arrive at
the epoch boundary. The original production weights remain retained.
[First-16 measurements](results/entropy-qat-first16.json).

## First epoch: rate and quality gates fail

After 30 updates / 1,044,469 targets, the real static15 file grows from
**2,904,325 to 2,911,289 B**: the two packaged copies cost 13,928 B more.

| Fixed native scoring sample | Original raw NLL | Epoch 1 | Change |
|---|---:|---:|---:|
| Prefix, 523,194 targets | 0.4859066183 | 0.5446639363 | +0.0587573180 |
| Independent q2, 516,391 targets | 0.9394297830 | 1.0201699989 | +0.0807402159 |

Both increases exceed the +0.004 gate. Hard symbol entropy also rises;
1,449,283 of 5,868,864 integer weights change. Original production weights
remain retained. The existing bounded trajectory continues to measure its
recovery curve, with the same optimizer, temperature schedule and budget.

A complete 1,346-target training piece deteriorates in both backends:
CPU QAT NLL rises by 0.04398 and native NLL by 0.03775. Their absolute
NLL differs by about 0.003 on this piece; native scores remain authoritative.
Eight original weight scales below 1e-10 become 4.51e-5–2.33e-4 after epoch 1.
Cold Adam's first update is `-lr*g/(abs(g)+eps)`: LSQ gradient scaling does
little to reduce that absolute step when `abs(g) >> eps`. This is a concrete
perturbation mechanism; its share of the aggregate loss is unmeasured.
[Measured package, scores and scale diagnostics](results/entropy-qat-epoch1.json).
