# What the frozen NN adds to causal PPMd priors

One scan of the existing complete-piece inputs measures the probability
baseline beneath both ongoing learning methods. PPMd has already consumed the
real preceding bytes of the full prepared stream. The original neural weights,
model execution and original training queues stay fixed.

## Measured incremental prediction value

The prior column uses the captured 205-column binary16 row, a float32 `1e-6`
probability floor, and row normalization. The NN column is the original native
`probs_mean`. Positive prior-minus-NN loss is the NN's improvement.

| Scope | Targets | Guarded PPMd nats/target | Original NN nats/target | NN improvement |
| --- | ---: | ---: | ---: | ---: |
| held | 1,082,505 | 1.555349406 | 0.966248314 | 0.589101092 |
| held-short | 974,858 | 1.515268521 | 0.933964311 | 0.581304211 |
| held-long | 107,647 | 1.918324421 | 1.258614281 | 0.659710140 |
| max-audit | 131,071 | 1.218899976 | 0.038986375 | 1.179913602 |

**All 473 held pieces have lower average NLL under the original NN.** Its
largest measured contribution occurs in the independent 131,071-target segment
of the giant article: the NN removes about 96.8% of the prior's NLL there.
The independent audit is reported separately from the sampled held total.
Held/audit roles refer to the two new learning trajectories. The original
checkpoint is the upstream enwik9-trained predictor described in the
[training writeup](../writeup.md#training).
This favors preserving the learned predictor and measuring compressed students.
The original causal and eight-block methods continue to their selected-weight
mixed-codec measurement.

## Alignment and rounding

Within each original piece, prior row `i` scores token `i + 1`. The piece's
last row is unused and its first byte has no NN target. Explicit bounds preserve
all 1,082,505 original held targets. Train has 2,941,307 targets in 1,313 pieces;
its original receipt supplies prior scores and a missing NN-baseline field.

The unguarded binary16 distribution contains one zero target in train and one
in held. The raw column records that event and a null whole-scope NLL; the
finite comparison above uses the probability guard. All four metric definitions
share the same target mask.

The result also reports the production predictor's scalar-tail conversion:
columns 200–204 halve binary16 subnormal values before the guard. That carrier
metric changes the held prior NLL by about `+0.000008464` nats/target and the
audit by `-0.000020141`. NN raw-logit and fp32-probability scores retain their
original names. Full payload transmission is evaluated by the real FXCM,
argmax-context, online mixers, SSE and range coder.

## Reproduce

```sh
python3 research/prior-value.py --self-test
python3 research/prior-value.py \
  --job /ABS/original-causal-job \
  --out /ABS/existing-output-directory/prior-value.json
```

The job is the original `causal-whole-piece-v1`, seed 414, with captured
`train/validation.u8`, `train/validation.f16`, explicit bounds, `data/plan.json`,
`data-manifest.json`, original `baseline.json`, the reserved long-audit inputs,
and the parent's `prepared/vocab.bin`. The probe writes all piece-level scores
to the output file and prints a small scope summary. Its one six-row fixture
covers target alignment, piece resets, zeros and guards.

[Complete measured result](results/causal-prior-value.json) preserves inputs,
counts, original native loss pairs and all four prior definitions. The scan
used **7.915115 CPU seconds**, **18.689813 wall seconds** and **59,308 KiB RSS**
with NumPy 2.2.4, one CPU and bounded chunks. Additional model inference,
PPMd capture and codec execution were zero.
