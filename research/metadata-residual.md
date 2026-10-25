# Two causal clocks in Wikipedia metadata

The prepared header carries two different revision-ID processes. Below
`2^24`, observed revision IDs follow page order; newer IDs follow time.
Separating these regimes repairs a previously poor single-clock predictor.
The resulting numeric rewrite uses **zero offline weights or lookup tables**.

## Complete-header observation

The original PHDA9 already delta-codes page IDs, normalizes timestamps and
moves metadata behind article text. It leaves these two cross-field relations
for the general-purpose predictor to discover.

| Exact same 243,425 records | Original | Residual rewrite |
|---|---:|---:|
| Prepared header bytes | 15,717,602 | 14,060,752 |
| zlib level-9 bytes | 4,165,276 | 3,342,195 |

The **823,081 B / 19.76% zlib saving** includes every residual and all 243,425
`@` markers. The entire header restores byte for byte. This is a fast-codec
direction measurement; the original v26 coder and the full-file score require
their own measurements. [Observation and frozen screen](results/metadata-residual-observation.json).

There are 87,813 old and 155,612 new revisions. The old page-offset predictor
is exact 77.26% of the time and within one ID 89.29%. The new-only clock has
median error two seconds, with 94.79% of errors within 60 seconds. Its late
records retain a two-second median. One mixed clock had large late errors.

## Fixed reversible transform

[metadata-residual.py](metadata-residual.py) accepts complete canonical
PHDA9-WRT header records. Each call starts with empty maps and a signed
relative page coordinate of zero. Page-delta bytes and all other fields stay
unchanged. Coordinate translation preserves old-ID predictions, so a cold
mid-header slice needs no previous-page seed.

- **Old revision ID `<2^24`:** choose the closest previously observed old page
  key, breaking equal distances toward the smaller key. Predict
  `r_neighbor + page - page_neighbor`; the empty-map prediction is zero.
  Replace the revision ID with `@` followed by its signed decimal residual.
- **New revision ID:** transmit the original ID. Predict seconds since
  2001-01-01 using the two previously observed new revision IDs bracketing it;
  outside the range use the two end points. A singleton predicts its time,
  and an empty map predicts zero. Interpolation is integer nearest rounding,
  with exact halves away from zero. Replace the timestamp by `@` plus its
  signed residual.
- Insert the current true value **after** prediction and reconstruction.
  The revision marker selects the old branch during decoding; the new branch
  expects its timestamp marker. Canonical numbers and dates restore the exact
  original representation.

```sh
python3 -B research/metadata-residual.py --self-test
python3 -B research/metadata-residual.py encode header.bin residual.bin
python3 -B research/metadata-residual.py decode residual.bin restored.bin
```

The research helper intentionally handles this header grammar. It leaves the
production preprocessor unchanged. A successful coder screen would justify a
small C++ integration and an actual double-binary size measurement.

## Real-coder screen

Use the unchanged v26 probe, 205-byte vocabulary, dictionary and Transformer
weights. The two fixed cases each contain 16,384 complete original records:

| Case | Prepared offset | Original bytes | Candidate bytes | Old/new records |
|---|---:|---:|---:|---:|
| First clock-dominated records | 554,867,462 | 1,077,456 | 966,396 | 225 / 16,159 |
| Last page-dominated records | 569,560,117 | 1,024,947 | 930,865 | 14,412 / 1,972 |

Both maps start empty for each case. Every original and rewritten stream gets
a real encode/decode. Rewritten output is inverse-transformed and compared
with the exact original slice. Charge each candidate **16 additional bytes**
for region-position/length framing. Use original→candidate order for the
first case and the reverse for the second, keeping each pair on one machine.

The fixed continuation gate is nonnegative savings on both cases and at
least **32,768 B combined**, after framing. That is one byte per sampled
record, a roughly 243 KB header-component scenario at the full record count.
Actual complete-header savings, C++ packaging cost and mature model behavior
are later measurements. Record coding/CPU time and peak RSS alongside sizes.
The queue stops after these two cases.

A longer test must include the following language tail: shortening the
header shifts forced Transformer piece boundaries, even where those later
bytes themselves stay unchanged. Article-body bytes and the original model
weights remain fixed.
