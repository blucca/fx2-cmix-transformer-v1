# Teacher-mean agreement, label-specific path gradients

The equal-per-visited-bit objective has the same teacher-label expected loss as
full categorical KL. Its **one-label gradient follows the visited prefixes**.
This saved-row measurement makes that distinction quantitative before any new
learning run. The current full teacher-only trajectory retains its original 2KL.

## The effective-row derivative

Let T and S be positive normalized rows, with `softmax(z) = S`. At one visited
prefix, let O and Z be its nonempty one/zero children, with conditional one
probabilities t and s. The Bernoulli-KL contribution has logit derivative:

```text
g_i = (s - t) × S_i / S(O),       i in O
g_i = -(s - t) × S_i / S(Z),      i in Z
g_i = 0,                         outside the prefix
```

Sum this over the target byte's eight visited nodes to obtain the gradient of
`U(y)`. A deterministic vocabulary branch contributes zero. Enumerating all
teacher-drawn labels gives:

```text
E_T U(y) = KL(T || S)
E_T g_U(y) = S - T
E_T ||g_U(y)||² = ||S - T||² + E_T ||g_U(y) - (S - T)||²
```

The coordinate here is **effective guarded-row logits**. Original-model logits,
the guard/half transform, the declared CPU shadow and the parameter VJP retain
their own chain rules. The native-index forward/shadow contract is
[specified separately](native-observer-training-contract.md).

## Twelve saved initial-student rows

The input is the existing four saved rows from each of three selected complete
held pieces: **12 examples**, each enumerating **205 labels**. The full
3,934-target geometry has its [separate target-weighted result](byte-path-quotient.md).
These twelve-example statistics give every saved row equal weight.

| Statistic | Measured value |
| :--- | ---: |
| Actual-label mean U | 1.356487901 |
| Mean full KL | 1.339574979 |
| Pooled actual U/full-KL gradient norm ratio | 1.271789955 |
| Pooled actual U/full-KL gradient cosine | 0.848784828 |
| Teacher-label RMS U/full-KL gradient norm ratio | 1.246241191 |
| Teacher-label variance / expected squared U-gradient norm | 35.6134% |
| Maximum enumerated mean-gradient identity error | 1.814e-16 |

The pooled norms sum squared norms across the twelve examples before taking
ratios. Per-row actual-label norm ratios range from **0.9689 to 3.0311**.
The smallest U/full-KL cosine is **0.1907**. At hard/piece14/row0, the actual
U/label-CE gradient cosine is **−0.4777**, while full-KL/CE cosine is **+0.1228**.
Changing the visited-path weighting changes the sample's output-gradient
direction, including its relationship to the observed label.

This replay checks an output-space loss and its label-dependent estimator.
Training convergence, held/long/giant generalization and the selected weight's
actual mixer/u16/range bytes belong to the complete learned-method measurement.

## Replay

```sh
python3 research/byte-path-gradient-check.py \
  research/results/byte-path-initial8-geometry.json
```

The standard-library script enumerates the saved rows, verifies the mean loss
and gradient identities and the variance decomposition, and checks the analytic
gradient against one four-leaf finite difference. Model forwards, parameter
VJPs, optimizer steps and new native calls are zero.

[Full per-example result](results/byte-path-gradient-saved-rows.json).
