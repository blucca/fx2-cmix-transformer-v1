# Gated recurrence: computation slots, unique weights, and saved bytes

**Repeated computation can enlarge a residual model’s function family while
reusing its parameters.** This note gives an exact small witness and follows
the existing transformer’s CPU, export and native contracts. The current
[teacher-only eight-layer experiment](teacher-only-distillation.md) continues;
gated recurrence is a source-reviewed proposal.

## A small, exact function witness

Use a scalar ReLU-squared residual on positive inputs:

```text
F_a(x) = x + a x²
H_a,g(x) = (1−g)F_a(x) + g F_a(F_a(x))
         = F_a(x) + g a F_a(x)²
```

Setting `g=0` preserves every original `F_a`. With `a=g=1`, inputs `1,2`
give outputs `6,42`. The single original block fitting the first output has
`a=5` and gives `6,22`. The recurrent family therefore contains a function
outside this original one-parameter family.

At `a=1,g=0`, the output derivatives across these two inputs are:

| Input | ∂H/∂a | ∂H/∂g |
|---|---:|---:|
| 1 | 1 | 4 |
| 2 | 4 | 36 |

Their determinant is **20**: the new gate supplies an independent local
output direction in this example. At `a=g=1`, the two occurrences contribute
`5+4=9` and `52+36=88` to the single shared parameter’s derivatives.

These numbers describe this scalar family. Transformer fidelity, trained
weights and compressed-byte results require their own measurements.

Run the dependency-free check:

```sh
python3 -B research/check-gated-recurrence.py
```

[Saved output](results/gated-recurrence-contract.json) also checks the
source-derived skip mapping and a two-slot state witness. Measurement scope:
exact rational algebra and control flow; transformer targets **0**, native
observations **0**, trained models **0**, coded-byte measurements **0**.

## 1. Preserve the complete block function

[The Python constructor](../pysrc/model.py#L1092-L1107) already accepts
attention/MLP sharing patterns and reuses the same module objects. The
[original cooldown recipe](../training_recipes/cooldown.py#L55-L60) and
[current eight-layer configuration](depth_student.py#L55-L85) each use
distinct modules at every computation position.

A block first mixes the stream and token embedding, then adds attention and
MLP residuals ([source](../pysrc/model.py#L994-L1021)):

```text
u = r*x + t*E
v = u + Attention(RMS(u))
B(x) = v + MLP(RMS(v))
```

The existing `rsc` controls the input stream. A zero value removes that carry
term; the attention and MLP branches remain active. A proposed identity
position needs a **whole-block outer gate**, `T_g(x)=(1−g)x+gB(x)`, covering
this complete calculation.

A type-compatible 12-position / 8-bank proposal is:

```text
bank:  0 1 2 3 | 0 1 2 3 | 4 5 6 7
gate:  1 1 1 1 | 0 0 0 0 | 1 1 1 1
```

Banks retain the eight-layer parameters, scales and KDA/ordinary-attention
types. Active positions retain its token/prior, causal, RoPE and window
semantics. Block-local stream/token coefficients need explicit ownership;
the existing sharing patterns cover attention and MLP modules.

The transformer also uses reverse-paired long skips
([source](../pysrc/model.py#L1238-L1276)). For an original four-value skip
vector `[s0,s1,s2,s3]`, the twelve-position embedding requires
**`[0,0,s0,s1,s2,s3]`**. The two additional paths vanish and the original
four sources remain aligned. The check verifies this control flow with
affine block stubs and three exact inputs.

Physical zero-gate fidelity additionally follows the implemented F32/BF16
blend, quantization scales and native operation order. A deployed candidate
needs its own same-weight native comparison.

## 2. Share gradients and export canonical values

[The coordinate installer](depth-distillation.py#L39-L72) collects canonical
`named_parameters()` and attaches external integer-coordinate closures. A
reused module calls the same coordinate, so reverse mode sums its occurrence
contributions. [The entropy update](depth-distillation.py#L354-L392) charges
each unique stored coordinate once.

`state_dict()` can expose multiple paths to a shared module. The current
[hard export](depth-distillation.py#L327-L336) replaces canonical entries
from those external coordinates. A shared exporter must materialize **every
execution path from the canonical updated values**, or store one canonical
bank plus an execution map. Copying old aliased state entries would export
stale values.

## 3. Share weights while retaining each slot’s history

Both native implementations load/repack per-layer weights
([reference](../cpp_infer/src/model.cpp#L219-L298),
[optimized](../cpp_infer/src/opt/arena_build.cpp#L133-L204)). Their current
KDA/KV indexing couples a weight index to its streaming-state index.

A recurrent architecture needs **one parameter-bank index and a separate
computation-slot state index**. Each slot has a different input history. The
small check uses one coefficient with two accumulating slots: independent
slot 0 reaches `2` after two steps; an aliased state reaches `5`. Sharing the
parameters preserves two histories only when their state slots remain distinct.

The [C3 loader](../cpp_infer/src/weights_io_compressed.cpp#L301-L417) stores
material tensors and codes each I8 tensor’s symbols. Its current encoding
set has no tensor-reference opcode. A minimal representation can use its
existing I32/F32 support: eight canonical material banks, an I32 execution
map and F32 gates, consumed by a new model loader. The container and static15
coder can remain in place; canonical export, native bank/slot scheduling and
complete model identity need implementation.

**Archive accounting remains `payload + 2B + 2D + O + 2W + 69`.** Gate/map
metadata, actual packed W and the new native binary enter the deployed
measurement. Unique parameter count is a training coordinate; the actual
archive pays for its complete representation.

## Current decision

Retain the complete eight-layer teacher-only comparison. Gated recurrence
changes the architecture’s capacity and deployment contract; the next
implementation decision requires a credible learned-fidelity gain and a
complete native/packed-weight validation plan. This audit adds source and
mechanism evidence; the current training, native model and codec stay intact.
