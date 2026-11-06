# A signed local coding objective for byte-logit models

2026-11-07. Mathematical research for the frozen original-v22 context mixer.
The current sixteen-epoch causal trajectory, eight-block 32-epoch trajectory,
selectors and one mixed measurement per method stay fixed. New capture, model
training and codec runs for this objective have an implementation allocation of
zero. [Native-domain evidence](native-loss-regimes.md) motivates the question;
the same-weight G/M/u16/actual-byte measurements locate the deployment bottleneck.

## Recommendation

Use **affine-logit cross entropy on the eight observed-prefix bits**, with signed
local slopes and intercepts anchored to the reference codec. Retain the existing
teacher distribution as the trust reference. Admit this complete next method
when the existing paired measurement shows material guarded-NN gain eroding at
the pre-SSE mixer. When actual mixed bytes already support a worthwhile complete
archive, deliver that candidate first.

A slope-weighted neural CE and a signed affine CE have different optimization
geometry. A negative CE multiplier gives a loss unbounded below as the neural
prediction becomes confidently wrong. Affine CE remains nonnegative for positive,
zero and negative slopes; its residual changes with the current prediction.
A direct reference residual times slope is its gradient at the reference point.
This makes affine CE the useful finite-step continuation of that gradient.

## 1. The native byte partition

`ByteMixer::SetProbs` maps the 205 model columns to the ascending valid byte
values in a 256-entry array. `ByteModel::Predict/Perceive` bisect the current byte
interval [L,U] at m=L+floor((U-L)/2), traversing eight MSB-first decisions. Partitions
use **actual byte values**, including the vocabulary holes.

For logits z, valid bytes V and the observed-prefix branches B1 and B0:

    x_k(z) = logsumexp(z on B1) - logsumexp(z on B0)
    y_k    = the observed byte's bit

Each branching prefix has both nonempty sets. A vocabulary-deterministic prefix
has zero task gradient; retain its native override flag for deployed accounting.
The eight ordinary conditional-bit CEs telescope to byte CE. The current native
codec additionally applies half rounding, a 1e-6 probability floor, feature
clamps and a 100001-cell logit lookup before mixing.

Source: retained original-v22 `src/mixer/byte-mixer.cpp:20–32`,
`src/models/byte-model.cpp:9–37`, `src/predictor.cpp:255–257,428–509`;
the byte mapping and byte-model arithmetic also appear in the current source.

## 2. The complete local objective

Freeze current expert outputs, context keys, mixer weights, earlier online
history and selected clamp branches. The triangular 23+1 outer mixer gives the
local slope s_k=dZ_k/dx_k, including the explicit second-level NN skip. For the
reference logit x0 and observed pre-SSE mixer Z0, define

    eta_k(z) = Z0_k + s_k * (x_k(z) - x0_k)
    L_k(z)   = softplus(eta_k(z)) - y_k * eta_k(z)
    r_k(z)   = s_k * (sigmoid(eta_k(z)) - y_k)
    gradient_z L = sum_k r_k(z) * gradient_z x_k(z).

At the reference the loss, probability and local first derivative match the
frozen smooth mixer. The [source-specific v22 continuation](v22-codec-local-slope.md)
adds its live internal FXCM paths and pairs the cached derivative with the
correct bit; its explicit last-output skip is neutral padding. s=0 gives a constant loss and zero task gradient. s<0
reverses the useful NN-logit direction while preserving a proper binary CE.

For each scalar bit logit x, curvature is

    d²L_k/dx² = s_k² * q_k * (1-q_k) >= 0.

The complete objective in transformer byte logits has Hessian

    sum_k [s_k²*q_k*(1-q_k)*g_k*g_k^T + r_k*Hessian(x_k)].

The first term is positive semidefinite. The second contains the difference of
the two branch softmax covariances and can have negative curvature. The ordinary
unmodified eight-bit CE recovers the usual convex byte softmax CE by telescoping;
custom affine bit objectives generally have the composite geometry above.
The existing gradient optimizer and teacher reference are sufficient for an
initial complete method; new second-order optimizer engineering has zero budget.

For unfloored branches, g_k equals the B1 softmax vector minus the B0 softmax
vector, embedded in the complete 205-column output. All other columns have zero
component at that prefix. The gradient sums to zero under a common logit shift.

## 3. Guarded probabilities need the guarded derivative

For a smooth probability-floor surrogate, let p=softmax(z), q_i=max(p_i,epsilon),
Q1=sum(B1 q_i), Q0=sum(B0 q_i), and

    a_i = 1[p_i>epsilon] * (1[i in B1]/Q1 - 1[i in B0]/Q0)
    g_j = p_j * (a_j - sum_i p_i*a_i).

This is the exact derivative of log(Q1/Q0) away from floor boundaries. With no
active floors, the mean term cancels and gives the two-LSE formula. With floors,
the common softmax normalization interacts with fixed floor mass: a column
outside the current prefix can receive a gradient through that mean term.
The included fixture exercises this behavior and the all-floored zero-gradient
case. Its floor of .05 is a deliberate mathematical fixture, separate from the
native 1e-6 deployment constant.

Actual half conversion and logit lookup are staircase functions. A future QAT
implementation specifies their straight-through derivative explicitly and uses
the native rounded/guarded forward values. Reference x0 is obtained in that same
surrogate coordinate. Anchoring eta at the stored actual Z0 preserves the
reference probability despite the forward rounding offset. The current native
scalar half decoder also has the documented subnormal-tail behavior; reuse the
existing conversion contract when implementing that forward boundary.

## 4. Mixer and final-u16 objectives have separate local slopes

The existing trace stores guarded neural probability G, pre-SSE mixer logit M,
final u16 and bit/override flags. It provides each layer's observed loss and the
paired differences, with the reducer reproducing the range-coder byte count.
Those values supply residuals and reference intercepts. Local mixer weights,
clamp derivatives and SSE table state are additional sufficient statistics for
local slopes. A paired finite difference between two whole W trajectories gives
their secant contrast, including changed contexts and online adaptation.

The original SSE starts from p=sigmoid(Z), converts to a 15-bit P(0), uses two
seven-node interpolators and two learned logit mixers, then returns P(1). Context
selection includes probability bins and preceding-byte history. On one frozen
smooth branch E, define q=E(p). Its logit Jacobian is

    tau = E'(p) * p*(1-p) / (q*(1-q)).

For an explicit smooth count-domain chain, set S=32768, H=S/2 and
c=(H-1)/ln(S-1). Replace the frozen lookup cells by their smooth counterparts:

    T(r)=H+c*ln((S-r)/r);  Q(t)=S*sigmoid(-(t-H)/c)
    X_C(t)=H+(C/8192)*(t-H)
    I(r)=C_j+(6*r/S-j)*(C_{j+1}-C_j)-8192
    m_alpha(a,b)=b+alpha*(a-b), alpha=(mixer_weight-H)/S.

Here j and adjacent SSE nodes are held at the reference branch; active clamps
use their native branch. With p0=1+(1-p)*32766, the original chain is

    p1=I6(Q(X_f0(T(p0)))); s0=X_f1(T(p0)); s1=X_f2(T(p1))
    s2=X_sm6(m_alpha1(s0,s1))
    p2=I7(Q(X_f3(T(p0)))); s4=X_f4(T(p2))
    s5=X_sm7(m_alpha2(s2,s4)); E(p)=1-(Q(s5)-1)/32766.

Ordinary scalar chain rule yields E'. The minimal check uses the original
extrapolation coefficients and mixer initialization, with synthetic adjacent
SSE nodes, to verify this entire branch by finite difference.

A final-probability local objective uses effective slope **tau*s**, anchored to
logit(u16/65536). Multiplying a final-u16 residual by the pre-SSE slope uses a
unit-SSE-Jacobian assumption. Read the existing M→u16 loss difference first:
start with the M objective for a G→M bottleneck; budget SSE-state capture only
when the measured final calibration makes it relevant. On a vocabulary override
bit the final prediction follows its deterministic branch, giving zero local
NN task derivative while the online model updates retain their native behavior.

FXCM receives NN-dependent stretch features and argmax context; the outer mixer
also has an NN-dependent auxiliary context. The current local slope definition
freezes those coupled paths. A complete experiment evaluates their resulting
changes in the same-weight deployed codec. Full-history differentiation is a
separate mechanism with its own complete cost.

Source: retained original-v22 `src/predictor.cpp:488–509,716–719`,
`src/mixer/mixer.cpp:43–61`, `src/mixer/mixer-input.cpp:15–35`,
`src/mixer/sse.cpp:242–288,320–324`, `src/coder/encoder.cpp:24–48` and
`research/trace.h`. SSE arithmetic and the trace layout are unchanged locally.

## 5. Complete admission and resource scope

A future complete method needs: native-preserving slope capture; selected full
causal pieces and held articles; matching priors/teacher/reference coordinates;
a fixed training trajectory; native hard-export curves and the independent long
audit; one actual mixed candidate; and complete archive/resource validation when
its net size supports the record path. Price these phases together after the
current mixed run provides actual same-machine CPU throughput.

Faithful dispersed mixer-state samples require advancing the coupled baseline
through the entire 586459020-byte prepared stream. The existing dense trace is
16 bytes per bit, so that scan would write **75,066,754,560 bytes**. Collect selected
rows during the scan instead. Two FP32 scalars (slope and reference intercept)
per bit cost 64 bytes per selected NN target: the current 2941307 training plus
1082505 held targets would occupy **257,523,968 bytes**, with offsets/masks separate.
The paired trace remains useful for the currently promised small intervals.

The original 32 MiB measurement has a whole-stream projection of 43.66 elapsed
hours; its CPU-specific field and machine T are outside that old receipt. The
PPMd-only full causal scan measured about 4.09 CPU hours. A new coupled scan uses
the actual mixed CPU/time/RSS and remaining owner capacity to set its budget.
Current native/training throughput and the existing complete-piece machinery
calibrate the learning and export phases. The same Hutter archive/award path is
counted once; publication of this design carries measured coding gain zero.

## Minimal replay

    python3 -B research/check-codec-aware-byte-surrogate.py

The seven-byte synthetic fixture preserves the native ascending-byte/MSB
partition. Six cases check signed/zero slopes, floored-softmax gradients,
common-logit-shift invariance and eight-bit telescoping; one scalar direction
exhibits negative byte-logit curvature. A second scalar fixture checks the
complete two-interpolator smooth SSE chain. Byte-gradient finite-difference
maximum error is 2.28e-11; all results are in
[the mathematical receipt](results/codec-aware-byte-surrogate-check.json).
NN/codec calls are zero.
