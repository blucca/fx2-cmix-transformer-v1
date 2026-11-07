# Complete causal16: e2 survives the trajectory and saves real range bytes

The whole-stream causal-prior method has completed **16 epochs**,
**1,267 optimizer steps** and **47,060,912 training targets**. Its frozen
native-loss selection remains **epoch2 / step159**. The selected weight file
saves **697 B** on the complete 8 MiB prefix and **221 B** on the complete
1 MiB q2 sample, both with verified restoration.

The two sample payloads save 918 B. The one shared two-copy weight change is
−92 B, yielding **826 B saved in that sampled component**. Formal full-archive
S0/S1 measurements are a separate stage.

[Selected e2 weights](assets/causal-e02.tfwc3) ·
[Complete trajectory and paired results](results/causal-complete-16-mixed.json) ·
[Standard-library check](check-causal-complete-16.py) ·
[Whole-stream input method](causal-priors.md)

## Complete trajectory and selected identity

The original architecture, signed BF16 scales, causal input, full-piece resets
and fixed integer-coordinate recipe stay unchanged through the trajectory.
Each epoch visits 2,941,307 training targets. The original sampled split has
1,313 train pieces and 473 validation pieces; the latter contain 1,082,505
targets. The independent maximum-length audit has 131,071 targets.

| Native domain, nat/target | Original W0 | Frozen e2 | Final e16 |
|---|---:|---:|---:|
| All sampled validation | 0.966248339 | **0.964965105** | 1.023256920 |
| Three long validation pieces | 1.258614301 | **1.256361755** | 1.327866396 |
| Independent maximum-length audit | 0.038986426 | **0.038884940** | 0.062126634 |

All 17 exports, including e0, are recorded in the complete result. e2 is the
one trained export selected by the original target-weighted held proxy plus
double-weight cost. W0 is 2,904,325 B; e2 is 2,904,371 B; e16 is 2,909,677 B.
The same e2 file drives both genuine codec samples. Original weight identities
and restored-slice identities are retained in the result.

## Genuine samples and scored intervals

| Full encoded sample | W0 payload B | e2 payload B | Saved B |
|---|---:|---:|---:|
| Prefix `[0,8388608)` | 1,072,279 | 1,071,582 | **697** |
| q2 `[293232482,294281058)` | 208,769 | 208,548 | **221** |

The prefix’s original mature score interval is
**`[7344758,8386448)`**, containing 1,041,690 bytes. Its interval saves
**98 B**. Whole-prefix payload and this mature interval are distinct recorded
quantities. The q2 score interval covers its entire 1 MiB parent trace.
Candidate decodes restore 8,388,608 and 1,048,576 bytes respectively. The
baseline decodes reuse the original matching golden records.

## From guarded neural loss to actual range bytes

`G` is the observed neural f16/guarded probability loss. `M` is the pre-SSE
mixer-logit ideal loss. `u16` is the final probability presented to the range
coder. The table uses the same neural-target mask on both weight sides.

| Scored scope | Neural targets | G saved bit | M saved bit | u16 saved bit | Actual interval saved B | κ_u16 |
|---|---:|---:|---:|---:|---:|---:|
| Prefix mature interval | 1,041,397 | 2,107.869 | 875.318 | 790.380 | **98** | 0.374967 |
| Full q2 | 1,048,367 | 2,621.310 | 1,834.163 | 1,769.508 | **221** | 0.675047 |
| q2 excluding direct train input | 1,037,183 | 2,536.603 | 1,765.846 | 1,701.251 | **212** | 0.670681 |
| Old W64 window by seed414 held role | 505,207 | 940.880 | 519.127 | 471.841 | **59** | 0.501489 |

All four rows compare the same W0 and e2 files. The last row reuses the old
W64 native-evaluation window as its input geometry. Each κ belongs to its
stated input, mask and retained history. Native randomized
validation loss is recorded as its own input scope. The original online PPMd,
mixer and range histories remain intact across excluded positions.

## Direct update exposure and separated intervals

The causal sample contains one q2 training piece. Its input is
`[293818027,293829212)`, 11,185 bytes; its next-token targets are
`[293818028,293829212)`, 11,184 positions. The q2 sample contains zero targets
from the realized seed414 sampled validation pieces. Original W0 pretraining
and the complete online codec history remain part of the provenance.

Removing the full direct training input leaves two reporting intervals:

```text
[293232482,293818027) + [293829212,294281058)
1,037,391 total bytes = 1,037,183 neural targets + 208 first-token fallbacks
206,871 → 206,659 actual interval bytes, saving 212 B
```

The old-W64 reporting scope uses the stated seed414 held-role window and
removes the same current causal input. Its two source intervals and original
parent-range endpoints are recorded separately. These are overlapping
reporting scopes; each paired value keeps its own definition.

## Flush and range residual are explicit

The reader replays from parent byte0 and sums
`closed(end)−closed(begin)`, with `closed=emitted+1`. Each full q2 payload has
one equal final Flush byte. For q2, the excluded intervals plus Flush are
1,898 B under W0 and 1,889 B under e2, so

```text
full q2 saving = 212 B outside direct train input + 9 B in its complement = 221 B
```

The non-direct-update neural u16 saving is 1,701.250898 bit. First-token
fallbacks contribute −0.976110 bit; the signed range residual is −0.534349 B.
Their sum gives the measured 212 B exactly. The result supplies the same
accounting for all four scopes, including unchanged override-bit counts.

## Named full-stream decision

Under the original **historical 952,225 B margin scenario**, the mature-rate
projection is

```text
586459020 × 98 / 1041690 − 92 = 55080.828730 B
```

That scenario requires **1,692 B** mature interval savings. The measured 98 B
triggers **stop_new_fullS** for this fixed candidate and scenario. The historical
record, component assumptions and gate identity are retained in the JSON.

Actual S0/S1 package sizes and a new latest official L are unmeasured fields
here. A formal entry uses the actual complete packages and
`S1 ≤ floor(99×latestL/100)`. The published result is a complete trained
trajectory and two genuine sample round trips.

## Check the released result

```sh
python3 research/check-causal-complete-16.py
```

The standard-library check covers all 17 exports, frozen e2 selection, decoded
identities, four bit-to-range layers, direct exposure, Flush, two-copy weight
accounting and the named gate. Eight existing trace reads take 1.921726 CPU s;
this collection adds zero model, training or codec reruns. The original trace
reader is documented in [codec-local probability accounting](codec-aware-byte-surrogate.md).
