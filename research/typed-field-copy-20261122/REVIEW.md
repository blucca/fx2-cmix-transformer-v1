# Causal boundary and implementation review

2026-11-22. **PASS for the fixed frozen-frequency mechanism screen.**
The subsequent charged-cost screen fails on the two measured samples;
live native integration stops at that gate.

## Original native state: source-level integration contract

The original [`Encoder::Encode`](../../src/coder/encoder.cpp) obtains
`Discretize(Predict())`, changes its private interval, calls `Perceive(bit)`
and normalizes. [`Decoder::Decode`](../../src/coder/decoder.cpp) mirrors it.
Range registers and stream I/O have no feedback into
[`Predictor`](../../src/predictor.cpp).

Predict refreshes model/mixer/SSE prediction caches and input rows. Perceive
trains from those values, advances bit/byte contexts and feeds each completed
byte to PPMd and the neural state/reset schedule. A copied-bit Perceive call
with stale Predict caches diverges. The required schedule is exactly one
**Predict → Perceive pair per original reconstructed bit**, in original order,
on both RAW and COPY; event/rank symbols update only the range/control model.

Let H(k) be the full fixed native state before original bit k. At an identical
reconstructed prefix and H(k), Predict yields the same discretized frequency
and refreshed caches. RAW decodes its bit with that frequency; COPY obtains
the bit from a causally completed exact spelling. Perceive on that bit gives
the same H(k+1). Preserve original quantization, dictionary pretraining,
seeds, byte feedback and reset schedules throughout this induction.

An encoder may buffer one bounded field's original u16 probabilities by
Predicting/Perceiving each original bit while retaining the prefix3 table
group. It transmits saved frequencies on RAW or omits suffix range symbols
on COPY, with zero repeated predictor advancement. The live decoder would
Predict/Perceive the reconstructed suffix. Such integration requires extracting
frequency-only range operations from the existing coupled coder API.

The implemented screen's `FrozenFixture` Predict/Perceive bookkeeping consumes
a fixed observation fixture. Its counters and bit-order digests validate that
fixture schedule; direct source equality is checked after decode. Executed
live native model-state equality and full archive qualification belong to a
separate integration measurement, stopped after this negative cost result.

## Focused source checks

- Encoder group lookup precedes lookahead, and lookahead changes no parser/
  table state. The current field completes only after its physical RAW
  delimiter. Exact spelling and its causal MRU rank stay fixed for the event.
- Prefix3 has a one-shot latch. Every nonempty group pays once, including an
  abort, overlength field or exact slice EOF. Empty groups pay zero events.
- COPY accepts only a complete bounded legal target spelling and expands
  its suffix; caption/external paths remain RAW. Eligible title insertion
  waits for the entire close marker and applies the `[]Q\n` exclusion.
- Encoder/counter/decoder retain inclusive-u16 intervals, normalization,
  terminal byte and native EOF zero padding. Both peers code event/rank
  symbols identically and update/halve event counters after the actual mode.
- Stable table entries, full equality, global completed-recency eviction
  and per-prefix MRU ordering preserve the entry/spelling/heap bounds.
- Tail checkpoints use original reconstructed byte positions without a live
  range reset. A copy crossing that boundary separates trigger fees from
  bytewise avoided suffix cost.
- The replay verifies trace length/records, all original labels, original
  golden count, closed decode/source equality, count/cost/table equality,
  fixture call counts and bit order.

## Exact-EOF correction before the measured replays

The synthetic case `title("ABC") + "[ABC"` completes a prior title and ends
with a partial TARGET at prefix3. Its nonempty ABC group owes one RAW event.
An early implementation skipped that event. The final source removed both
EOF bypasses before either measured replay. The existing test now checks
RAW1, partial1, completed targets0 and prior title completions1. Prefix,
table bounds, prior, rank code and selection policy remain fixed.

Final independently reviewed SHA256:

| Source | SHA256 |
|---|---|
| field-copy.h | `31d113a2aeaa8c6abb8b9b8e95e7ea19c179128d1d7542a8a1542539e9248bd3` |
| field-event-replay.cpp | `19cf7a2707f095e5b0c472759e99c1291052c990423a655de6f5924867011d31` |
| field-copy-test.cpp | `e8bacc35995ea26c920057bf3f04bbc44aa9bae41eaa54989cb1d8c739aca54c` |

The publication retains these exact source bytes. Its one minimal synthetic
check is recorded in [test-check.json](test-check.json), alongside the two
complete scientific sample results linked in [README.md](README.md).
