# A bounded neural cache: measured information is too small

This fixed experiment retrieves previously observed next bytes from nearby
neural contexts. The trained model and inference kernels stay unchanged.
The cache is research-only and disabled by default (`FX2_NEURAL_CACHE=0`).

## Fixed specification

The design follows the [continuous neural cache](https://arxiv.org/abs/1612.04426)
with a bounded CPU retrieval rule:

- Use the final 192-dimensional RMS-normalized hidden state, then L2-normalize it.
- Index **205 FIFO buckets by the current input token**. Each retains the most
  recent **128** `(normalized hidden state, observed next token)` pairs.
- Query the current bucket with `exp(20 * (cosine - maximum_cosine))`; sum by
  next-token label and normalize. Empty buckets return the base distribution.
- Predict and score using the old cache, then insert the newly observed target.
  Retain the cache across article and 2^17 splits; clear it between samples.
  The last token of a piece is a valid target; the first token of the next
  piece has no cross-reset neural prediction.
- Mix `q = (1-lambda)*p + lambda*cache`. Fit just one shared coefficient in
  `[0, .5]` on the two samples' first halves, round it to f32, then freeze it
  for the two second halves. Capacity, temperature and bucket rule are fixed.

The heap stores **20,257,280 B** of keys/labels. A query compares at most 128
vectors: 24,576 multiply-accumulates and 128 exponentials. The implementation is
[`neural-cache.h`](neural-cache.h); the strict-floating-point
[`activation-probe.cpp`](activation-probe.cpp) writes two f32 probabilities
per scored target, base and cache, with the true first-half target count.

For fitting, let `r = cache_probability / base_probability - 1` on first-half
targets. Maximize `sum(log1p(lambda*r))` by bisecting its decreasing derivative
`sum(r/(1+lambda*r))`, checking both endpoints first. Second-half targets do
not enter the fit. The learned f32 coefficient is **0.010696017183363438**
(little-endian bytes `59 3e 2f 3c`), trained on **1,047,767 targets**.

## Result and predeclared gate

The quality gate requires both second halves to improve and their combined
improvement to reach **0.012 nats/token**. The paired timing allowance is at
most **8 microseconds per token** of added raw-network evaluation time.

| Fixed 1 MiB sample | First-half improvement | Frozen second-half improvement | Paired extra microseconds/token |
|---|---:|---:|---:|
| Prefix | 0.0003861982 | 0.0007942361 | 1.847 |
| Article q2 | 0.0003891624 | 0.0002938949 | 4.260 |

Combined second-half improvement is **0.0005439387 nats/token**, just **4.53%**
of the quality gate. Added time is **3.054 microseconds/token**, passing the
timing gate. The observed rate would correspond to about 58 kB of raw-network
entropy over the full prepared stream; full-stream and mixed-coder gains were
not measured.

**Stop this fixed cache recipe.** No real-coder integration, longer run or
capacity/temperature sweep follows. The original model and lossless weight
container remain available.

All six evaluations—one 64 KiB smoke pair and two 1 MiB pairs—preserve the
original full probability hashes, NLL totals, target counts and piece bounds.
The one-fit/held-out scoring pass also checks each base half against the
original probability NLL. Builds, evaluations and fitting consume **354.158
CPU seconds**. [Full receipt and per-block results](results/neural-cache-A.json).

To emit pairs, compile the probe with `FX2_NEURAL_CACHE=1`, strict floating
point for its main, and the same define for the model's read-only hidden-state
accessor. Reuse the production model objects otherwise. Invoke
`probe eval TOKENS PRIORS WEIGHTS OUTPUT_PAIRS`; inputs and capture commands
follow the [activation probe protocol](positive-activation.md).
