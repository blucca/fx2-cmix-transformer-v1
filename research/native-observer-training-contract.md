# Native observer values, CPU shadow gradients

This is a design contract and a scalar replay of existing observations.
It adds zero model forwards, optimizer steps or new native captures.
The [saved native capsule](native-byte-path/findings.md) still has one changed
lookup among its 96 teacher/oracle target-path pairs.

## Three distinct inputs

1. A CPU model produces logits. A chosen softmax, half encoder, decoder and
   floor produce a row. The saved-row experiments use this boundary.
2. The original `ByteModel`, `MixerInput` and `Sigmoid` consume a supplied row
   and target byte, returning eight actual lookup indices and overrides.
   The capsule directly links these source files under the fast/O3 contract.
3. The deployed native model produces that row from the current hard-exported
   weights and the causal piece. `fx2::opt::TransformerOpt` is the existing
   full forward implementation, with `begin_article` and `step`. The coder
   then applies its actual half conversion, decoder tail, guard and consumer.

An exported-native observation objective needs **3 → 2** for its forward value.
A CPU row connected to the native consumer is a **CPU-row native-consumer shadow**.
Both are useful experiments with their respective input identities.

The full native forward is already available in
[`activation-probe.cpp`](activation-probe.cpp) and
[`model_opt.cpp`](../cpp_infer/src/opt/model_opt.cpp). Its loader accepts RAW
`FX2TFW01` as well as compressed weights
([`arena_build.cpp`](../cpp_infer/src/opt/arena_build.cpp)).
[`depth_student.py`](depth_student.py) already exports the hard eight-layer
state to RAW. A future training bridge can load one current hard W per
optimizer batch, reset each piece and consume its actual rows.
The existing `eval-pieces` interface requires compact token/prior input with
explicit piece bounds; a batch bridge must preserve that input contract.

## The scalar loss

For normal observations use `r=(lookup_index + 0.5)/100001`; use 0 or 1 for
actual endpoint overrides. This coordinate identifies the observed
lookup/override pair. For a target byte y, sum Bernoulli KL over its eight
visited nodes:

`U_obs(y) = Σ_visited KL(Bern(r_teacher) || Bern(r_student))`.

The scalar loss is zero exactly when those eight observed pairs match.
This is the output-observation term; hidden-state and weight-entropy
auxiliaries have their own constraints. A complete new recipe must decide
which auxiliaries it keeps. The current teacher-only trajectory stays unchanged.

For a normal student coordinate the relaxed derivative is

`g = (r_student - r_teacher) / (r_student * (1 - r_student))`.

In the saved native counterexample, indices 52721 and 52722 give
`KL = 2.0059032845e-10` and `g = 4.0118497362e-5`.
The other 95 pairs give zero loss, including three shared zero overrides.
The native lookup and clamp are piecewise constant. Their ordinary chain
derivative is zero on plateaus. A controlled clipped case with teacher index20,
student index10 and shadow q=1e-5 gives positive loss and shadow g≈−.952481: a
usable direction for leaving the plateau. This case is a scalar control.

## Preserve the actual forward value

For finite tensors, an explicit shadow estimator can be written as:

```python
# r_obs is computed from the actual native index/override.
# q_shadow is the declared differentiable CPU coordinate.
r_train = r_obs + (q_shadow - q_shadow.detach())
```

The forward adds an exact zero; the backward follows q_shadow.
The construction `q_shadow + (r_obs - q_shadow).detach()` reconstructs the
forward through subtraction and addition. Our FP32 control uses
`r_obs=0.00010499895142856985` and `q_shadow=0.9900000095367432`: that
reconstruction returns `0.00010502338409423828`. Adding the identical-shadow
difference preserves the original center bit pattern.

This declares an estimator, with actual native values and a CPU relaxation
for derivatives. A [native log-odds residual carrier](native-log-odds-shadow.md)
expresses the observed backward directly as `(r_student-r_teacher) × eta_shadow`. Learning, complete native held/long/giant evaluation and
the selected W's real mixer/u16/range bytes remain the full-method acceptance.

## Replay

```sh
python3 research/native-observer-vjp-check.py
```

The standard-library checker reads all 192 frozen native observations, checks
zero/mismatched pairs and the relaxed scalar derivative, and runs the two
explicitly labelled controls. It invokes zero native builds or model runs.
Saved output: [native-observer-vjp-check.json](results/native-observer-vjp-check.json).
