# Fixed scale changes the units of an Adam step

The admitted q7 method increases each signed weight scale from `s15` to
`s7 = BF16_RNE((7/3) * s15)`, then optimizes integer coordinates with the
same Adam learning rate, 0.01. The scale stays fixed throughout the run.

The pre-round latent coordinate is `theta = s*z`. The archive uses
`s*round(z)`, with the method's projected integer range. These are useful
separate quantities: the admitted 1,199-target update changed 5,443,182
floating coordinates while retaining every hard symbol and the initial archive.

## A controlled local calculation

Hold a scalar physical-gradient history `h_t = dL_t/dtheta` fixed between
two coordinate systems. Let `r = s7/s15 > 0`. Their coordinate gradients then
satisfy `g7_t = r*g15_t`. With identical Adam parameters, zero weight decay,
and correspondingly scaled moment states:

```
mhat7 = r*mhat15
vhat7 = r*r*vhat15

delta_theta15 = -lr*s15*mhat15 / (sqrt(vhat15) + eps)
delta_theta7  = -lr*r*s15*mhat15 / (sqrt(vhat15) + eps/r)
```

For a nonzero old displacement, the new-to-old ratio before projection is

```
R = r * (sqrt(vhat15) + eps) / (sqrt(vhat15) + eps/r)
```

When the second-moment root dominates epsilon, `R` approaches `r`. As a
nonzero gradient history shrinks into the epsilon-dominated regime, it
approaches `r*r`. Bias correction preserves both scaling identities.

The [standard-library control](q7-coordinate-step.py) enumerates all 128 BF16
mantissas with both signs. A successful normal-BF16 scale map has the same
ratio across exponents; representative tiny and large exponents are checked.
It gives

```
2.324786324786325 <= r <= 2.3423423423423424
5.404631455913508 <= r*r <= 5.486567648729811
```

The sign of the scale is preserved, so the ratio stays positive for signed
rows. These bounds cover the representable scale map, with its finite-normal
input and successful finite output conditions.

## Scalar Adam histories

Prescribed linear-loss gradients `[a, -a/2, 2*a]` supply an independent
three-step control. Both signed scales, both moment recurrences and their
bias corrections are evaluated directly. With `s15=1`, the mapped BF16
scale is exactly `s7=2.328125`.

| Physical gradient magnitude a | Last scaled-latent displacement ratio |
|---:|---:|
| 1e-4 | 2.328225106 |
| 1e-8 | 3.084370610 |
| 1e-12 | 5.419211280 |

Every directly computed ratio matches the formula. Each trajectory stays
within half an integer-coordinate unit, making the distinction between
latent movement and hard-symbol changes explicit.

## Apply the result at its stated scope

The calculation uses a shared complete gradient history up to factor `r`,
including any rate term transformed covariantly. It describes the scaled
latent coordinate before rounding and the post-step box projection.

The real q15 and q7 runs have distinct output objectives, alphabet/rate
centers and CPU forward math. The older causal student/teacher compute
scaled FP32 linear products; q7 and its cached teacher compute integer-unit
products followed by the native BF16-scale fold. Their native evaluations
share the original full12 family and holdout contracts.

The actual q7 recipe remains fixed: same learning rate, scales, optimizer,
projection, complete32 target and sole selected-weight codec audit. The
[observed e4 native result](results/q7-native-e4.json) retains its empirical
scope; this calculation supplies an explicit optimizer-unit control.

```sh
python3 -B research/q7-coordinate-step.py
```

It writes [the scalar result](results/q7-coordinate-step.json), checks the
scale-ratio bounds and direct Adam arithmetic, and performs zero model,
native, real-training or network calls.
