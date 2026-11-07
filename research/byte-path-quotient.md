# A byte coder's nine-group distillation target

**A smaller neural predictor has a precise observation target:** the eight
conditional probabilities consumed while coding the next byte. This audit
connects that interface to a coarse teacher KL, measures its output geometry,
and checks the existing half/guard boundaries. The ongoing complete
[teacher-only trajectory](teacher-only-distillation.md) retains its original 2KL.

## Follow the probability interface

The network emits 205 vocabulary probabilities. Vocabulary indices map to
ascending **raw byte IDs**; `ByteModel::Predict()` splits the raw `[0,255]`
range MSB first and returns right-subtree mass divided by remaining mass.
After each observed bit it selects that bit's subtree.

The normal mixed coder consumes that scalar through:

```text
NN prefix probability → clamp + logit lookup → 23 recursive mixers
→ NN/FXCM skip inputs → final mixer → sigmoid → integer SSE → u16 → range bytes
```

The recursive mixers and their auxiliary context use the stretched NN scalar.
The current source's `ex`, `lstmex` and `lstmpr` end at assignments; the other
NN full-row read computes diagnostic loss. Relevant sources are
[byte-model.cpp](../src/models/byte-model.cpp),
[predictor.cpp](../src/predictor.cpp),
[mixer-input.cpp](../src/mixer/mixer-input.cpp) and
[sigmoid.cpp](../src/mixer/sigmoid.cpp).

**Exact payload contract:** with the same input bytes, vocabulary, non-neural
source/initial state and floating-point execution contract, matching each
actual bit's NN **lookup index and normal/0/1 override** gives identical mixer
inputs, context selection, online updates, SSE and range-coded payload.
This follows by induction over the fixed observed bits. Weight and binary
charges remain separate archive components.

## The partition induced by one target byte

For target byte `y`, form its singleton leaf and the eight sibling subtrees
that first diverge from its MSB path. Empty vocabulary groups disappear.
These at most **nine groups** partition all 205 entries.

```python
group_id = 8 - (raw_byte ^ target_raw_byte).bit_length()
```

Let `T_g` and `S_g` be the teacher/student normalized effective group masses.
The teacher-prefix mass at depth `d` is `A_d`, and its conditional bit
probability is `t_d`; the student's is `s_d`. Then:

```text
D_path = KL(T_groups || S_groups)
       = Σ_d A_d × KL(Bernoulli(t_d) || Bernoulli(s_d))

KL(T || S) = D_path + Σ_g T_g × KL(T(.|g) || S(.|g))
```

Matching the group masses fixes every real-arithmetic path conditional and the
target leaf probability. Inside each unvisited sibling, mass can redistribute.
The unconstrained minimum `KL(R || S)` projection with teacher group masses is:

```text
R_i = S_i × T_g / S_g,  i in group g
R_y = T_y
KL(T || R) = KL(T || S) − D_path
∇_z D_path = S − R,  when softmax(z) = S
```

At the teacher point, the path Fisher has rank at most **8**, versus **204**
for full categorical KL. For a logit direction `v`, its quadratic form is
`Σ_g T_g (E_T[v|g] − E_T[v])²`; the omitted within-group conditional variances
complete the full Fisher. These statements use effective-probability simplex
coordinates. A model's half/guard transform adds its own VJP.

`y` selects the partition during **offline learning and this output-space
oracle**. The deployed predictor continues to receive the original causal
inputs and emit a complete probability row.

### Count each actually visited bit once

For a learning objective, use `U(y) = Σ_d KL(Bern(t_d) || Bern(s_d))` along the
actual target path. Deterministic vocabulary branches contribute zero. The
nine-group certificate weights these terms by teacher prefix mass `A_d`.

Enumerating a target drawn from the frozen teacher gives:

```text
E_{y~T} U(y) = KL(T || S)
E_{y~T} D_path(y) = Σ_all_nodes A_node² × KL(Bern(t_node) || Bern(s_node))
```

The unweighted path preserves the expected full-KL scale under teacher-drawn
targets and follows actual bit visits under the data distribution. **Recommendation:
use equal-per-visited-bit teacher KL for a future learning comparison; use the
nine-group KL as its observation-equivalence and capacity certificate.** Both
have zero teacher-point gradient and the same exact-path-match zero set.

[The saved-row check](results/byte-path-weighting-saved-rows.json) enumerates all
205 targets on each of 12 saved rows to verify the expectation identity. These
are objective identities; the complete compressed-model experiment supplies
its learned native/coded performance.

## Initial-eight measurement: three complete causal held pieces

The existing full-piece CPU loader ran the original 12-block teacher and the
**untrained mapped eight-block initialization**, with the original signed
scales and native-scaled GEMM. Three previously scored held pieces were selected
by declared length/teacher-loss regimes. FP32 logits were analyzed with FP64
softmax, the `1e-6` guard and normalization; native half controls follow below.

| Regime | Targets | Full KL | Path KL | Within-sibling KL |
| :--- | ---: | ---: | ---: | ---: |
| Short | 576 | 0.437391 | 0.394144 | 0.043246 |
| Middle | 1,606 | 0.438772 | 0.310986 | 0.127786 |
| Hard | 1,752 | 0.756991 | 0.488118 | 0.268872 |
| **Target-weighted** | **3,934** | **0.580288** | **0.402047** | **0.178240** |

Thus **30.716%** of this measured full KL lies within unvisited siblings.
The real-arithmetic projection's maximum path error was `2.22e-16`.
The three pieces' effective-output gradient cosines were 0.988/0.940/0.905.
This table describes the initial student and these rows; the learned model's
ability to fit the coarser target is the next research question.

The measurement used **six forwards, 6.0304 CPU seconds and 510,764 KiB peak RSS**
under a 1 GiB memory cgroup. Parameter VJPs, optimizer updates and codec runs
were zero. [Full result and 12 saved logit examples](results/byte-path-initial8-geometry.json).

## Keep the physical probability boundaries

The normal native path rounds outputs through half precision, preserves the
last-five scalar subnormal decoder rule, applies `1e-6`, then sums float
subtrees. The mixer uses a **100,001-entry** probability-to-logit table.
These are part of the existing codec contract.

The simple projection can place entries below the guard. The floor-feasible
probability-space projection minimizes each sibling's relative entropy using
`R_i = max(floor, λ_g S_i)` and the teacher's absolute group mass.
It preserves the group sums while respecting the measured floor.

On the **12 saved rows**, applying the source half-tail/guard rule gave:

- Simple projection: 1,175 entries below the guard.
- Floor-feasible projection: **0** below it; maximum group error `1.11e-16`.
- Serial-FP32 source-order lookup comparison: **0/96** changed indices.
- Rounding that feasible oracle through half again: **31/96** changed indices.

The last result exposes the physical half-representability question for any
learning experiment. Production fast/O3 accumulation and actual payload bytes
retain their native validation. [Saved-row controls](results/byte-path-half-guard-control.json).

## Replay and continuation

Standard-library checks and saved-logit replay:

```sh
python3 research/byte_path_kl.py \
  --evidence research/results/byte-path-initial8-geometry.json
python3 research/byte-path-half-control.py \
  research/results/byte-path-initial8-geometry.json
python3 research/byte-path-weighting-check.py \
  research/results/byte-path-initial8-geometry.json
```

The [measurement source](byte-path-measure.py) accepts the existing causal data
manifest, original checkpoint/kernel and [frozen three-piece selection](results/byte-path-piece-selection.json):

```sh
python3 research/byte-path-measure.py --source . \
  --data /PATH/data-manifest.json --checkpoint /PATH/original.tch \
  --kernel /PATH/cpu-kda.so \
  --selection research/results/byte-path-piece-selection.json \
  --output temp/byte-path
```

It enforces 120 CPU seconds and a 4 GiB virtual-address limit; the original
run's 1 GiB systemd memory limit is separate. Setup and complete-piece data
provenance follow [the existing causal method](causal-priors.md).

The complete 2KL control and its same-weight mixed result come first. A later
unweighted bit-path learning comparison would explicitly target the half/guard effective row,
retain causal inputs at deployment, and measure learned held/long/giant behavior
plus its own G→M→u16→bytes. The current training and codec stay frozen.

**Related work:** [Decoupled Knowledge Distillation](https://arxiv.org/abs/2203.08679)
and [Generalized DKD](https://arxiv.org/abs/2512.04625) study grouped logit KL and
valuable within-group knowledge for classification. Here the groups follow the
literal bit-coder observation path; compression and learned generalization
supply the performance criteria.
