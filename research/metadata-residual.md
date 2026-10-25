# Two causal clocks in Wikipedia metadata

The prepared header carries two different revision-ID processes. Below
`2^24`, observed revision IDs follow page order; newer IDs follow time.
Separating these regimes repairs a previously poor single-clock predictor.
The resulting numeric rewrite uses **zero offline weights or lookup tables**.

## Real-v26 result: stop this rewrite

Both original and candidate streams completed real encode/decode, followed
by exact restoration of the same source records. Each candidate pays 16 B
for framing.

| 16,384-record case | Original coded B | Candidate coded B | Net saved B |
|---|---:|---:|---:|
| New-clock-heavy first records | 134,517 | 129,905 | **4,596** |
| Old-page-heavy last records | 72,595 | 74,095 | **−1,516** |
| Combined | 207,112 | 204,000 + 32 framing | **3,080** |

The fixed gate required each case to save bytes and a combined saving of at
least 32,768 B. It failed both requirements. **The online ASCII recipe stops;
no full-tail run or production integration was started.** Same-host coding
time fell about 10% in each case as the input became shorter.

The whole-header zlib result below remains a useful structural observation.
The stronger model retains almost all of its advantage on the original
format; further work starts with its actual field-level coding costs.
[Complete paired result](results/metadata-coder-screen.json).

The old probe predates the CPU field in newer probe reports. A controller
aggregation error was repaired after the first variant on each host; all
completed phases were reused, then the missing variants ran within their
original one-hour envelopes. Total child CPU was 2,952.39 seconds.

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
direction measurement; the real-v26 result is reported above. [Observation and frozen screen](results/metadata-residual-observation.json).

There are 87,813 old and 155,612 new revisions. The old page-offset predictor
is exact 77.26% of the time and within one ID 89.29%. The new-only clock has
median error two seconds, with 94.79% of errors within 60 seconds. Its late
records retain a two-second median. One mixed clock had large late errors.

## Fixed reversible transform

[metadata-residual.py](metadata-residual.py) and its
[C++ port](metadata-residual.cpp) accept complete canonical PHDA9-WRT header
records. Each call starts with empty maps and a signed
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

The research helpers handle this header grammar and leave the production
preprocessor unchanged. The completed coder screen stops this recipe before
runner integration or a double-binary package measurement.

### C++ parity prototype

The single C++17 implementation exports
`metadata_residual::Rewrite(std::string_view, bool decode)`. It uses ordered
maps and signed 64-bit integer arithmetic. The CLI and one runnable self-check
are included only when `METADATA_RESIDUAL_MAIN` is defined:

```sh
mkdir -p temp
c++ -std=c++17 -O2 -DMETADATA_RESIDUAL_MAIN research/metadata-residual.cpp -o temp/metadata-residual
temp/metadata-residual --self-test
temp/metadata-residual encode header.bin residual.bin
temp/metadata-residual decode residual.bin restored.bin
```

On the complete 243,425-record header, C++ produces the same **14,060,752
bytes** as the frozen Python candidate and restores the **15,717,602 original
bytes**, both checked by direct byte comparison. The self-check covers both
branches, half-away rounding, nearest-page ties, coordinate translation,
optional/IP lines, canonical dates/numbers and duplicate-key rejection.

One local GCC 16.2.1 CLI run took **0.337 s encode / 0.362 s decode** including
file I/O, with peak RSS **63,668 / 71,600 KiB**. This is a transform-cost sample,
separate from the strong-coder screen. With `-Os -fno-exceptions -fno-rtti
-ffunction-sections -fdata-sections`, the no-CLI object's code/constant sections
total **7,725 B**, or **15,450 B** for two copies before linking. The actual
linked package increment remains a later measurement. The core also builds
with the runner's exception-disabled flags; malformed input reports an error
and exits in that build.

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


## Whole-block dependency-order observation

The preprocessor can collect the complete header before reconstruction, so
residual dependencies can follow an order derived from already-visible keys.
One fixed observation used page-sorted predecessors for old revisions and
revision-sorted endpoint clocks followed by recursive medians for new clocks.
The output kept the same row order, markers and target fields. No permutation
or external table was transmitted; the decoder reconstructed the same
dependency order from page keys, branch markers and verbatim new revision IDs.

The complete header round-tripped exactly. Its zlib-9 size was **3,335,512 B**,
just **6,683 B** below the online recipe's 3,342,195 B. Most of the useful
cross-field information is already exposed by the simpler online predictor.
This whole-block variant stops at the observation stage; the real-coder queue
and C++ implementation retain the original online recipe. The observation used
10.08 CPU seconds and 227.5 MiB peak RSS. [Small result](results/metadata-dependency-order.json).

[Complete-header C++ check and build sizes](results/metadata-cpp.json).
