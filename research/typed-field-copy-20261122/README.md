# Typed exact field references: complete negative mechanism screen

2026-11-22. **The fixed charged-cost gate failed.** A complete COPY/RAW range
stream and causal decoder were tested on two existing same-family v26
probability fixtures. The 8 MiB prefix grows by **1 B**; the interior 1 MiB
ties. Both specified tails tie. Stop this prefix3/table/prior/index recipe
before native integration, full-stream/archive work, parameter changes or
field expansion.

The encoder, closed decoder, source equality, original golden counts, event
and cost accounts, and every original-bit fixture consumption all pass.
These are results for the stated recipe and two observed samples. Existing
q7, entity-mixture and other native result matrices remain unchanged.

## What was measured

This is a **frozen-frequency event/decoder mechanism screen**. It uses exact
u16 probabilities from the unchanged v26 observation stream. It physically
encodes every COPY/RAW event and reference index, decodes the complete range
stream, and reconstructs all original prepared bytes.

The decoder receives coded bytes, the explicitly fixed probability fixture
and original length. Original source bytes and trace labels are kept on the
encoder/validation side. `FrozenFixture` bookkeeping advances one Predict /
Perceive pair per reconstructed original bit, including copied suffixes.
Live native Predictor integration, full
enwik9 restoration, archive size S/S0, decoder/model package cost and official
entrant qualification were outside this completed first-stage measurement.
New native Predictor runs and old-matrix reruns are **0**.

Both samples start with an empty reference table and retain the original
prepared spelling and order. The interior is an independent slice, with its
own empty table and existing same-family trace.

| v26 fixture | Prepared offset B | Length B | Whole original → candidate B | Saved B | Specified tail | Tail original → candidate B | Saved B |
|---|---:|---:|---:|---:|---|---:|---:|
| Prefix 8 MiB | 0 | 8,388,608 | 1,068,581 → 1,068,582 | -1 | final 1 MiB | 178,714 → 178,714 | 0 |
| Interior 1 MiB | 293,232,482 | 1,048,576 | 208,696 → 208,696 | 0 | final 0.5 MiB | 87,871 → 87,871 | 0 |

Both actual processes complete with exit0. Total measured replay CPU is
**2.103425 s** (1.845731 / 0.257694); peak process RSS is **166,204 KiB**.
The fixed execution envelope allocates one CPU, 1 GiB address space,
120 CPU seconds and 240 wall seconds per sample. The probability fixture
occupies 128 MiB / 16 MiB in memory; it is a research input, outside the
reported range-payload byte count.

[Summary and fixed scope](summary.json) ·
[Prefix complete scientific result](results/prefix8-v26.json) ·
[Interior complete scientific result](results/interior1-v26.json).

## Exact reference availability and its charged price

| Account | Prefix 8 MiB | Interior 1 MiB |
|---|---:|---:|
| Prefix3 triggers | 236,699 | 34,449 |
| Deterministic empty-group skips | 6,960 | 2,607 |
| Charged events | 229,739 | 31,842 |
| RAW / COPY events | 229,737 / 2 | 31,842 / 0 |
| Exact complete matches at charged triggers | 162,932 | 20,855 |
| Copied suffix bytes | 27 | 0 |
| Zero-suffix matches / copies | 29,162 / 0 | 1,655 / 0 |
| Maximum prefix-group size | 3,080 | 411 |

All counts belong to the causal bounded table. A completed exact match is
selected only when the COPY mode plus its actual rank index costs less than
RAW mode plus the original suffix's measured u16 ideal cost.

| Whole ideal account, bits | Prefix 8 MiB | Interior 1 MiB |
|---|---:|---:|
| RAW-event bill | 17.670092 | 7.343124 |
| COPY-event bill | 29.000000 | 0 |
| Actual index bill | 7.000000 | 0 |
| Avoided original suffix cost | 39.964739 | 0 |
| Net ideal saving | -13.705353 | -7.343124 |

`net = avoided suffix - RAW events - COPY events - index`. The prefix's two
references avoid 39.964739 bits while its total control bill is 53.670092
bits. The interior pays RAW events and selects no COPY. Tail ideal savings
are -0.738816 / -1.925014 bits, alongside the physical byte ties. Ideal cost
and flushed integer-range byte counts are retained separately.

The table reaches 65,536 / 13,860 entries, with peak spelling bytes
864,043 / 151,768 and tracked table-heap peaks 10,969,640 / 2,373,957 B.
The prefix performs 9,747 global completed-recency evictions; the interior
performs 0. Tracked allocator-requested table heap and whole-process RSS are
separate measurements.

## Fixed causal mechanism

- Only internal TARGET fields can transmit an event. After exactly three
  decoded legal name bytes, consult previously completed spellings sharing
  that prefix. An empty group deterministically stays RAW with zero event
  symbols; every nonempty group pays one COPY/RAW event, including exact EOF.
- COPY requires full spelling equality, a field of at most 1,024 prepared
  bytes and a physical `]` or `Q` delimiter. It expands only the suffix after
  prefix3. The delimiter retains the original RAW bit path; completion and
  recency updates happen after it. Captions, external links, aborts,
  overlength fields and incomplete fields use RAW.
- Eligible titles and completed targets share one table: at most 65,536
  distinct entries, 8 MiB aggregate spelling bytes and 64 MiB tracked heap.
  Titles containing `[`, `]`, `Q` or newline are excluded. Global replacement
  evicts the least recently completed entry; each prefix group uses the same
  completion order, most recent first. The current/future field stays out
  until completion. The rank group is fixed before encoder lookahead.
- One global event prior starts COPY=1, RAW=256. The u16 frequency is
  `1 + floor(65534*one/(one+zero))`; update after coding the actual mode,
  halving both counters at total≥65,536, each at least 1.
- MRU ranks are encoded with standard truncated binary; binary symbols use
  p=32,768. Ranks have floor(log2 n) or floor(log2 n)+1 symbols according to
  the cutoff. The actual length is charged; a singleton sends zero index
  bits. Original u16 arithmetic, termination and EOF padding are kept.
- Both RAW and COPY perform `FrozenFixture` Predict then Perceive bookkeeping
  for every original reconstructed bit. Event/index symbols touch only transmission state.
  A live implementation would preserve that schedule and native prediction
  caches; [source-level boundary and review](REVIEW.md) record the contract.

**Tail accounting:** checkpoint a copy of the live interval at the original
reconstructed-byte boundary, without resetting or flushing the live stream.
Tail bytes are whole closed size minus the closed-prefix size; the common
termination byte cancels. Event/index fees belong to their prefix3 trigger,
including a copy crossing the tail boundary. Avoided suffix ideal costs are
attributed by their original byte positions. Complete JSON retains both this
closed-prefix difference and terminal-inclusive suffix counts.

## Minimal test and reproduction

From the repository root, using C++17 and a caller-owned scratch directory:

```sh
SCRATCH="$PWD/temp/typed-field-copy"
mkdir -p "$SCRATCH"
c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic -I research \
  research/typed-field-copy-20261122/field-copy-test.cpp \
  -o "$SCRATCH/field-copy-test"
"$SCRATCH/field-copy-test"
c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic -I research \
  research/typed-field-copy-20261122/field-event-replay.cpp \
  -o "$SCRATCH/field-event-replay"

# Supply original prepared data and its already matched v26 trace fixtures.
"$SCRATCH/field-event-replay" PREFIX8_V26_TRACE.bin PREPARED_STREAM.bin \
  0 8388608 7340032 1068581 "$SCRATCH/prefix8-v26"
"$SCRATCH/field-event-replay" INTERIOR1_V26_TRACE.bin PREPARED_STREAM.bin \
  293232482 1048576 524288 208696 "$SCRATCH/interior1-v26"
```

The one runnable synthetic test covers inclusive encode/counter/decode,
short/long/singleton truncated-binary ranks, causal completion, title filtering,
entry/byte-cap eviction, COPY and paid RAW paths, a tail inside COPY,
charged prefix3 EOF, zero-suffix one-shot behavior and event-count halving.
It checks source equality and every original-bit `FrozenFixture` call/order
in both branches.
[Publication check receipt](test-check.json).

For original fixture generation, follow the existing
[prepared-source recipe](../README.md#reproduce),
[observation layout](../README.md#small-causal-calibration-traces), and
[source-preserving v26 observer](../README.md#ppmd-reliability-observe-first-calibrate-causally).
The input trace contains one 16-byte `<HHfff>` record per original bit,
before Perceive and excluding dictionary pretraining. It must match the
corresponding original range golden. The replay validates every record,
every original bit label and the golden before charged coding. The older
v22 interior golden 208,769 belongs to a different arithmetic family.

Replays write `.coded.bin` and `.result.json` under the caller's scratch.
This publication includes code, review and scientific JSON, with generic
fixture labels replacing private absolute locations. Prepared bytes, traces
and coded binary artifacts remain outside the public append.

## Attribution and license

GPL-3.0-only, retaining the repository's [GPLv3 license](../../LICENSE).
Range arithmetic derives from [`src/coder/`](../../src/coder/) and
[`research/replay.cpp`](../replay.cpp); parser syntax and WRT title markers
derive from [`entity-prefix-replay.cpp`](../entity-prefix-replay.cpp).
The replay reuses the existing [`trace.h`](../trace.h).
This research builds on Vladimir Ivanov's transformer compressor and
Kaido Orav / Ibrahim Marcouch's statistical experts, with original source
authorship preserved. The new isolated field-reference mechanism and
measurement are maintained by Blucca / Codex.
