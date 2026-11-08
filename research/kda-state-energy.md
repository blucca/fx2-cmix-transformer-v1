# KDA state energy: value fitting and conditioned keys

For one KDA head, the actual CPU/native order is decay → retrieve → delta update:

\[
A=DS,\quad r=A^\top k,\quad
S'=A+\beta k(v-r)^\top.
\]

Here `S` is a key-by-value state matrix, `D` is the diagonal row decay,
`k` is the normalized key, `v` the incoming value, and `beta` the update gate.
The following certificate applies to exact arithmetic with
`0 ≤ Dii ≤ 1`, `0 ≤ beta ≤ 1`, and `c = ||k||² ≤ 1`.

The deployed persistent state is **FP32**. The Python check below uses
standard-library binary64 algebra controls. Native holdout losses and codec
bytes keep their measured runtime scopes.

## 1. A sharp one-step energy certificate

The rank-one update gives

\[
\Delta=\|S'\|_F^2-\|DS\|_F^2
=-\beta(2-\beta c)\|r\|^2
 +2\beta(1-\beta c)\langle r,v\rangle
 +\beta^2c\|v\|^2.
\]

Set `alpha = (1-beta*c)/(2-beta*c)`. Completing the square yields

\[
\boxed{\Delta=-\beta(2-\beta c)\|r-\alpha v\|^2
                  +\frac{\beta}{2-\beta c}\|v\|^2.}
\]

Thus, for `lambda_t = max_i D_t[i,i]`,

\[
\|S_T\|_F^2\le
\left(\prod_{j=1}^T\lambda_j^2\right)\|S_0\|_F^2+
\sum_{t=1}^T\left(\prod_{j=t+1}^T\lambda_j^2\right)
\frac{\beta_t}{2-\beta_tc_t}\|v_t\|^2.
\]

The injection coefficient lies between `beta/2` and `beta`. The one-step
coefficient is sharp whenever `beta > 0` and `Dk != 0`: choose
`S = (Dk)(alpha*v)^T / ||Dk||²`, which makes `r = alpha*v`.
For `beta = 0` or `k = 0`, the actual update is `S' = DS`.
The completed-square identity extends to `beta > 0, beta*c < 2`.

Uniform forgetting `lambda_t ≤ rho < 1` and `||v_t|| ≤ V` give a bounded
geometric energy sum. With `lambda_t ≤ 1` and bounded values, the squared-energy bound grows at most
linearly with sequence length.

## 2. A concrete key-conditioning mechanism

Take a two-dimensional state, scalar values, identity decay and unit beta.
Alternate

\[
k_+=[\cos\theta,\sin\theta]^\top,\quad v_+=1,
\qquad k_-=[\cos\theta,-\sin\theta]^\top,\quad v_-=-1.
\]

For `0 < theta < pi/4`, both updates fit the same state

\[
S_*=[0,1/\sin\theta]^\top,
\qquad k_+^\top S_*=1,\quad k_-^\top S_*=-1.
\]

Every homogeneous step `M_± = I-k_±k_±^T` is an orthogonal projection with
operator norm **1**. The two-step period has spectral radius `cos²(2*theta)`.
Starting at zero, it converges to the fixed point, whose norm grows as `1/theta`.
The key-pair condition number is `cot(theta)`.

| Mathematical control | State norm at the common fixed point | Period spectral radius |
| :--- | ---: | ---: |
| theta = 0.2 | 5.03348955 | 0.84835335 |
| theta = 0.1 | 10.01668613 | 0.96053050 |

The values have the same unit amplitude in both controls. Nearby keys with
opposite requested values build a large perpendicular state component through
repeated fitting. The identity-decay/unit-beta cases are mathematical boundary
controls; the current model uses its own learned qkv, decay and beta values.

### Forgetting: a stream that reaches the geometric energy bound

Keep the alternating unit keys and values, allow `0 ≤ theta ≤ pi/2`, and take
constant `D = d*I`, `0 < d < 1`, `0 < beta ≤ 1`. The stream has a unique
attracting periodic state pair:

\[
S_+=(a,b)^\top,\quad S_-=(-a,b)^\top,
\qquad
Z=(1-d)(1+d-d\beta)+2d\beta\sin^2\theta,
\]
\[
\boxed{a=\frac{\beta(1-d)\cos\theta}{Z},\qquad
b=\frac{\beta(1+d)\sin\theta}{Z}.}
\]

The plus update maps `S_-` to `S_+`; the minus update maps it back. Each
homogeneous step has norm `d`, so the cycle attracts every initial state.
Its squared norm, with `y = sin²(theta)`, is

\[
\|S_\pm\|^2=
\frac{\beta^2[(1-d)^2+4dy]}
{[(1-d)(1+d-d\beta)+2d\beta y]^2}.
\]

For `beta ≥ 1-d`, choose

\[
y_* = \frac{(1-d)(1+d-\beta)}{2d\beta}\in[0,1].
\]

At this angle, the retrieval before each update is
`r = ((1-beta)/(2-beta))*v`. The completed-square residual is zero at both
states, and the cycle reaches the stationary geometric energy ceiling exactly:

\[
\boxed{\|S_\pm\|^2=\frac{\beta}{(2-\beta)(1-d^2)}.}
\]

Thus the energy scale from Section 1 is attainable by a fixed input stream,
even with forgetting and unit-amplitude values. The displayed angle is also
the maximum of this cycle's squared norm over `y ∈ [0,1]` under the stated
gate condition.
At `beta = 1-d`, the angle is `pi/2` and the two states coincide. The near-key
range `theta_* < pi/4` corresponds to `beta > 1-d²`.

For `beta = 1`, `sin²(theta_*) = (1-d)/2`. If
`P_± = I-k_±*k_±^T`, the period matrix is `d²*P_-*P_+` with spectral radius
`d²*cos²(2*theta_*) = d⁴`. Starting from zero, after `m ≥ 1` periods the
exact-arithmetic state error is

\[
\|S_{2m}-S_-\|=d^{4m-1}\|S_-\|.
\]

| Mathematical control, beta = 1 | theta_* | Cycle state norm | Period spectral radius | Spectral half-life, updates |
| :--- | ---: | ---: | ---: | ---: |
| d = 0.99 | 0.07076974 | 7.08881205 | 0.96059601 | 34.4838 |
| d = 0.999 | 0.02236254 | 22.36627204 | 0.99600600 | 346.4003 |

The half-life is `2*log(0.5)/log(d⁴)` for the slow period eigenmode. It measures
asymptotic error decay; the finite-time norm formula includes its displayed
prefactor. These controls connect state amplitude, key geometry and history
length in one stream. The deployed recurrence has learned anisotropic decay,
epsilon-normalized keys and varying gates; its model-specific state amplitude
and forcing stay with its own native measurements.

## 3. Fixed-control propagation and cross-model forcing

Write `F_t(S) = M_t*S + b_t`, with

\[
M_t=(I-\beta_tk_tk_t^\top)D_t,
\qquad b_t=\beta_tk_tv_t^\top.
\]

For the same controls, a state perturbation follows `E_t = M_t*E_(t-1)` and
`||E_t||_F ≤ lambda_t*||E_(t-1)||_F`. The matrix `I-beta*k*k^T` has eigenvalues
`1-beta*c` in the key direction and 1 in orthogonal directions. Uniform
`lambda < 1` is a sufficient condition for strict contraction.

For two models with their own controls, the exact finite difference is

\[
\boxed{E_t=\widetilde M_tE_{t-1}
 +(\widetilde M_t-M_t)S_{t-1}
 +(\widetilde b_t-b_t).}
\]

The decay/key changes act on the accumulated state, and value changes add
fresh forcing. Before gated normalization, readout `o=S^T*q` obeys

\[
\delta o=E^\top\widetilde q+S^\top(\widetilde q-q),
\quad
\|\delta o\|\le\|\widetilde q\|\|E\|_F+\|\delta q\|\|S\|_F.
\]

This provides a specific long-history interpretation: key geometry, value
compatibility, forgetting and accumulated state jointly determine the response
to changed weights. The actual q7/teacher-only runs retain their fixed training,
selection and native/same-weight measurement contracts. Model-specific forcing
and key conditioning require those models' own signals.

FP32 writeback fits the same framework by adding an actual rounding residual
`e_t` to each update. Its one-step energy contribution is
`2*<S',e_t>_F + ||e_t||_F²`.

## Source and rerun

- [CPU recurrence](cpu-kda.cpp): FP32 decay, retrieve, delta update and q readout.
- [CPU input normalization](cpu_ops.py): `k` uses L2norm epsilon `1e-6`;
  `q` additionally uses `d^(-1/2)` scaling.
- [Gate reference](../pysrc/kda_reference.py):
  `D = exp(-exp(A_log)*softplus(g_raw+dt_bias))`.
- [Native recurrence](../cpp_infer/src/opt/kda.cpp) and
  [FP32 state](../cpp_infer/src/opt/kda.h): the same state-update structure,
  followed by gated RMSNorm and the output projection.

```sh
python3 -B research/kda-state-energy.py
```

[Result](results/kda-state-energy.json): eight fixed 4×3 energy cases, sharpness
and zero-boundary controls, fixed-control state differences, and two 2×1
identity-decay conditioned-key sequences of 4,000 updates. Three decayed cycles
check the fixed-point formula and energy equality, with a finite-period error
check for one unit-beta stream. The check compares the expanded and
completed-square energies with tolerance `1e-11`; both final toy errors are below
`1e-7`. The result names its exact-arithmetic error bounds separately from the
observed binary64 residuals.
