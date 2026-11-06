# Extracting the original-v22 local neural slope

2026-11-07. Source-specific continuation of the [signed byte objective](codec-aware-byte-surrogate.md).
The current causal sixteen-epoch and eight-block 32-epoch trajectories, their
selectors and single mixed measurements stay fixed. New capture, tracing,
training and full-stream implementation allocation is zero. This result gives
the actual same-bit path and a compact sufficient-statistic representation for
a later complete method, conditional on the existing G→M deployment evidence.

## 1. The explicit FXCM skip is neutral padding

The frozen original [v22 source](https://github.com/astOwOlfo/fx2-cmix-transformer-v1/blob/83f2603f3f7751da5f429fd32669807eb8510494/src/models/fxcmv1.cpp)
allocates `439+1-2-7 = 431` probabilities, initialized to `.5f` (lines 93–106).
The 31 context-map arrays have at most 81 active contexts. Both populated and
disabled branches expose `4+skip2` outputs per context, including their run
prediction. The original producer bound is:

| Producer | Maximum exposed outputs |
|---|---:|
| Context maps | 394 |
| Seven stationary maps | 7 |
| Match model: `1+2*nST`, `nST=3` | 7 |
| Sparse match / run / bias | 2 / 1 / 1 |
| Ten internal first-level mixers | 10 |
| Base, five exposed APM values and final blend | 7 |
| **Total written prefix** | **429** |

The two NN `Inputs.add` calls each decrement the output cursor, so their
temporary exposed slots are overwritten by the subsequent predictions. Slots
**429 and 430 stay .5**; shorter initialization prefixes preserve the same
bound. `FXCM::Predict` returns the complete 431-slot array. The outer predictor
uses its last slot, absolute input index 431, as the explicit FXCM skip and as
one auxiliary-context component. Its stretched value is exactly zero:

```
FXCM_last = .5                 d FXCM_last / dx = 0
auxiliary = floor(15 * (.5 + sigmoid(NN_lookup)) / 2)
```

Consequently the trace's `fxcm` field is that neutral skip probability. The
NN-sensitive **ten internal mixer outputs and seven APM outputs** remain in the
full first-level expert row. Existing G/M/u16/range accounting is unchanged.
The earlier 430-output prose is superseded by this 431-slot source count.

## 2. Compute the FXCM derivative when its cache is produced

Original `Predictor::Perceive` first updates the preceding bit's outer models,
then computes the next prefix's NN probability and argmax, writes
`lstmpr=1+4094*p` and `lstmex`, and calls `FXCM::Perceive` (716–719).
Inside `update1`, the twelve internal mixers update the preceding prediction,
then `modelPrediction` and the APM chain produce the next cached output
(4689–4765). `ResetPredictions` subsequently resets only the write cursor;
the next `FXCM::Predict` returns the cache (4812–4833).

A later collector computes and retains the derivative **during this cache
production**, then pairs it with the same next bit's outer `Z0`. The initial fixed FXCM
cache has zero local FXCM derivative. Freeze the
reference online history, selected context rows, APM state after its preceding
update, argmax and clamp/lookup branches. The local slope concerns this current
bit; changed contexts and later online adaptation are evaluated by a fresh
coupled codec trajectory.

## 3. The internal 10+2 path, with its real scales

On an ordinary prefix, let `x=logit(p_N)`, `r=native_uint(1+4094*p_N)`,
`T=stretch(r)` in native 8-bit logit units.
For the chosen smooth/straight-through branch, integer casts, shifts and lookup
rounding keep the native forward value. Casts and lookup rounding use identity
backward; a right shift by m multiplies the backward value by `2^-m`. The local derivative uses

```
T'(x) = gate_T * 256 * 4094 * p_N*(1-p_N)
                     * (1/r + 1/(4096-r)).
```

The evaluation point uses the native reference count and the active stretch
clamp. Each internal `Mixer1` pairwise dot shifts by 8, then multiplies by
`shift1` and shifts by 11 (488–568). Its smooth coefficient is therefore
`a_i=shift1_i/2^19`, with `p1` clipped to [-2047,2047]. For the ten selected rows:

```
d_i = gate_i * a_i * w_i,NN * T'                         i=0..9
v_j = gate_j * (6/2^19) * (sum_i w_j,i*d_i + w_j,10*T'/2) j=10,11
a   = native((7*L10+L11+4)>>3)
R'  = S'(a) * (7*v10+v11)/8
```

Here `S(d)=4096*sigmoid(d/256)` is the smooth count-domain squash; native
rounding, the +4 averaging offset and endpoint clamps remain in the forward
branch. The ten exposed probabilities have derivatives
`S'(L_i)*d_i/4095` because `AddPrediction` uses **1/4095**. Native first-level
shifts are 237,204,70,54,55,55,70,55,55,55; both final shifts are 6
(3255–3285). The two NN inputs are `T` and integer `T/2` (4674–4686).

For an APM with current frozen neighboring U16 nodes `t_j,t_j+1`, the chosen
native interpolation cell gives the smooth count derivative

```
A'(q) = (t_j+1-t_j)/2048 * stretch'(q).
```

Its state update precedes selecting/interpolating the current cell (1558–1567).
Evaluate `stretch'` at the actual native count; count zero follows the native
zero-to-one saturated stretch branch with derivative zero.
Starting with base count R, the seven exposed count values are

```
R
U = A3(native((A0(R)+7*R+4)>>3))
V1 = A1(R)
V = A4(V1)
Q = A2(R)
K = A5(U)
F = native((6*Q+U+11*V+14*K+31)>>5)     if fails&255
    native((4*Q+5*U+12*V+11*K+31)>>5)  otherwise.
```

Apply ordinary chain rule and divide each count derivative by 4095. Freeze the
current fail/context branch. This includes the 7:1 APM0 blend and both actual
final coefficient sets (4739–4765). The active ordinary producer prefix places
the ten mixer values at 412–421 and these seven values at 422–428.

## 4. Feed every live path into the triangular outer mixer

For each exposed probability `P_j`, put `C_j=clamp(P_j,1e-4,1-1e-4)`.
Its outer stretched derivative is `D_j=gate_j*P'_j/[C_j*(1-C_j)]` under
the specified smooth logit law; a saturated probability clamp has gate zero. The direct
NN input has `D_N=gate_N`; statistical inputs frozen at this bit have zero
local derivative. The 23 first-level outer mixers give

```
b_i = w_i,N*D_N + sum_FXCM_j w_i,j*D_j
                   + sum_l<i e_i,l*gate_l*b_l
s   = a_N*D_N + sum_i a_i*gate_i*b_i.
```

The explicit FXCM-last skip contributes zero. The clamp gates match both
`SetExtraInput` and `SetStretchedInput`. Omitting the FXCM derivatives defines
a useful direct-NN conditional slope with frozen expert outputs. Including
these seventeen paths gives the complete **same-bit, frozen-history smooth
slope**. Argmax/context switches, integer staircases and future weight updates
retain their native behavior in real mixed evaluation.

## 5. Two FP32 fields, calibrated in the training reference coordinate

At capture time, the predictor supplies the actual guarded bit probability
`p_native0`, its `x_native0=logit(p_native0)`, and pre-SSE `Z0`. Evaluate the
slope in sections 2–4 at these native values. A native-coordinate intercept
`Z0-s*x_native0` anchors a learner whose reference coordinate equals
`x_native0`. The current CPU/native execution difference makes that identity
an explicit preparation requirement.

On the existing 1346-target uniform-prior piece, the eight-block production
CPU route has CE **3.043249167092** and the same hard export has native logit
NLL **3.045512204858**: a **0.002263037766 nats/target** difference
([completed folded-GEMM receipt](results/depth-native-scaled-gemm.json)).
The earlier CPU route scored 3.040956131594 on this input
([native preflight](results/depth-native-preflight.json)). These are measured
aggregate losses; the reference correction uses each bit's actual coordinate.

Define `theta_ref` as the frozen hard-export initialization used by the captured
codec. Let `x_train(theta)` be the chosen CPU/guard/STE bit coordinate, and
compute `x_train_ref=x_train(theta_ref)` with the same model family, kernel,
priors, byte map and piece resets as the learner. The reference-centered law is

```
eta(theta) = Z0 + s * (x_train(theta) - x_train_ref)
b_train   = Z0 - s*x_train_ref
store (b_train, s)
eta(theta) = b_train + s*x_train(theta)
```

**Recommendation:** collect `(Z0,s)`, then fold the first field in one frozen
CPU reference pass, complete piece by complete piece. At `theta_ref`, the
recomposed prediction recovers `Z0` within the stored FP32 precision. The
chosen CPU/STE map supplies the parameter tangent; native hard-export and
same-weight mixed measurements evaluate its transferred changes. With a
native-only intercept, the reference drift is exactly
`s*(x_train_ref-x_native0)` before field rounding.

The twelve-block causal learner already caches **FP32 logits**, and its
zero-step check asserts elementwise equality with the training initialization
(`integer-coordinate-qat.py`, `teacher_logits` and `exact_teacher`). Reuse those
rows when their initialization and execution identities match `theta_ref`.
The eight-block distillation cache contains the **original twelve-block**
teacher's FP32 logits and features; it serves KL/MSE. Its affine anchor uses a
separate frozen **eight-block** reference with `native_scaled_gemm=True`.
The 205-column `.f16` inputs are causal PPMd priors. Cache dtype and prior dtype
have separate roles.

For a deterministic vocabulary branch, store `(Z0,0)` and use its existing
zero task mask; skip endpoint-logit evaluation. Target offsets, byte labels
and stream identity supply alignment. The coupled baseline still advances
through the intervening bytes.

At 4,023,812 selected targets and eight bits, the final two-field array remains
**257,523,968 B**; coordinates, headers and masks are separate. Fold and release
each reference piece's logits, keeping additional persistent reference-logit
storage at zero. An eight-block reference forward pass belongs in the complete
method's preparation budget. The focused check retains the native folding
case and adds a synthetic CPU-coordinate shift, its corrected FP32 anchor and
the reference BCE gradient. Its shift is a mathematical fixture.

Run `python3 -B research/check-v22-codec-local-slope.py` in this checkout.
It reads the frozen git object, replays the producer bound and affine-field
contract, and reports the [compact source result](results/v22-local-slope-source.json).
A different native family reviews its own producer counts before reusing the
padding or indexed-path conclusion. Actual G→M loss and M→u16 calibration still
select the next complete method and its resource budget.
