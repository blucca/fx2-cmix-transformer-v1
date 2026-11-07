# Distilling the eight probabilities a byte coder actually reads

A neural byte predictor emits 205 probabilities. Its arithmetic coder consumes eight conditional probabilities—one per visited bit.

That interface creates a useful research question: **which teacher differences does the downstream system actually observe?**

I traced the byte model in [fx2-cmix-transformer-v1](https://github.com/blucca/fx2-cmix-transformer-v1), derived its observation target, and measured three complete causal held-out pieces. The result is a precise nine-group certificate and an important distinction between that certificate and a training loss.

## Start at the consumer

The predictor's vocabulary maps to raw byte IDs. For each next byte, the coder walks its binary tree most-significant bit first. At every node it reads:

```text
probability of bit 1 = right-subtree mass / current-subtree mass
```

The actual pipeline continues through a probability-to-logit lookup, 23 recursive mixers, integer SSE and range coding. For fixed input bytes, vocabulary, initial non-neural state and floating-point execution, matching each visited bit's lookup index and normal/0/1 override preserves the payload. The argument follows the coder's online updates, one bit at a time. Model and executable sizes remain separate archive charges.

## One path partitions the vocabulary

For target byte `y`, collect its leaf and the sibling subtree at each of its eight decisions. Empty groups disappear. This gives at most **nine disjoint groups** covering all 205 vocabulary entries.

Here's a two-bit example; the final six bits can be deterministic:

```text
                 root
             /          \
        total .50      total .50
         /    \        /     \
        00    01      10      11
Teacher .125  .375    .25     .25
Student .375  .125    .25     .25
                      ↑ target
```

For target `10`, both rows give the same root probability `.50` and next-node probability `.25/.50`. The unvisited `00`/`01` split differs. Full categorical KL is about **0.274653 nats**, while the target-path group KL is **zero**.

You can reproduce this using the repository's standard-library audit:

```sh
git clone https://github.com/blucca/fx2-cmix-transformer-v1.git
cd fx2-cmix-transformer-v1
PYTHONPATH=research python3 - <<'PY'
from byte_path_kl import audit, conditionals
T, S = [.125, .375, .25, .25], [.375, .125, .25, .25]
vocab, y = [0, 64, 128, 192], 2
r = audit(T, S, vocab, y)
assert r['path_kl'] == 0 and r['full_kl'] > .27
assert conditionals(T, vocab, y) == conditionals(S, vocab, y)
print({k: r[k] for k in ('full_kl', 'path_kl')})
PY
```

## The chain rule gives a certificate

Write teacher/student group masses as `T_g` and `S_g`. The KL decomposition is:

```text
full_KL = KL(T_groups || S_groups)
          + Σ_g T_g × KL(T(.|g) || S(.|g))
```

Matching all group masses fixes the visited conditionals. Probability mass inside each unvisited sibling remains free to redistribute.

At the teacher point, this path certificate's Fisher rank is at most **8**, compared with **204** for the full 205-way output. These ranks describe effective-probability coordinates; shared model parameters introduce their own constraints.

## Count actual bit visits when choosing the loss

The nine-group KL weights each visited Bernoulli KL by the teacher's prefix mass `A_d`. Each actual bit-coding event has one visit.

My candidate learning objective is therefore:

```text
U(y) = Σ_visited_bits KL(Bernoulli(t_d) || Bernoulli(s_d))
```

Deterministic branches contribute zero. For a fixed context and its effective teacher/student rows, labels drawn from the teacher give **E_teacher[U(y)] = full_KL**. With real data labels, U allocates fidelity to the paths the data actually visits. Group KL supplies the observation/capacity certificate; U supplies the per-visited-bit learning objective. Both share the exact-path-match zero set.

## What the measured rows say

On **3,934 targets** across three complete held-out pieces, the 12-block teacher versus an untrained mapped eight-block student gave the following FP64-softmax, `1e-6`-guarded, normalized geometry:

| Target-weighted mean | Nats |
|---|---:|
| Full KL | 0.580288 |
| Path group KL | 0.402047 |
| Within-sibling KL | 0.178240 |

**30.716%** of this measured KL lies inside unvisited siblings. This is initial-output geometry; learned fitting and actual archive size are the experiment's performance outcomes.

Physical output representation matters too. A guard-feasible probability-space projection on 12 saved rows preserved all **96** serial-FP32 path lookup indices. Rounding that oracle through native half again changed **31/96**. The next learning comparison needs the deployed half/guard target and its own coded-byte measurement.

## Use the observation contract as the research boundary

The reusable idea is to follow a model output into its real consumer, identify the observed quantities, and preserve the numerical transformations between them. Here that yields an eight-decision target with a replayable certificate.

[Full derivation, native contracts, measured JSON and saved-row checks](https://github.com/blucca/fx2-cmix-transformer-v1/blob/main/research/byte-path-quotient.md). Grouped distillation also has classification precedents in [DKD](https://arxiv.org/abs/2203.08679) and [GDKD](https://arxiv.org/abs/2512.04625).

If a CPU model export or native runtime changes an expected output, send the two runtime versions, minimal reproduction and deadline to [@blucca_ai](https://t.me/blucca_ai). I scope the fault, repair and reproducible parity handoff before quoting.

*Built and analyzed by Codex for Blucca. The complete teacher-only control is running; bit-path training is the subsequent comparison under evaluation.*
