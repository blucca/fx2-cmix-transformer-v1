# A fixed nonuniform seven-integer weight alphabet

A seven-symbol quantizer changes both reconstruction error and weight-code
cost. The completed initialization recovers 3.000% of the uniform-seven
primary native quantization loss; this fixed candidate stops at the initial
measurement gate. This experiment keeps the full twelve-block model and the existing
[static15 container](weights-static15.md), and fits a single signed integer
alphabet to the published [weight census](results/weights-census.json).
The three measured coordinates are kept separate: original-q distortion,
native network NLL, and encoded weight bytes.

## 1. Exact histogram fit: original-q coordinates

The census contains **111 INT8 matrices and 5,868,864 values**, with histogram
bins `−7…7`. Each candidate has seven codes `{0, ±a, ±b, ±c}`, where
`1 ≤ a < b < c ≤ 7`. All tensors share the alphabet; each tensor has one fixed
positive scale multiplier `r_t`. The objective is

```
sum_t sum_q histogram_t[q] * (q - r_t * nearest_code(q / r_t))**2
```

There are 35 alphabet states. For each tensor, assignment changes occur at
`r = 2*q/(c_j+c_(j+1))` for positive q and adjacent nonnegative codes.
Between changes, the objective is `A*r**2 - 2*B*r + C`; its optimum is the
interval-clipped `B/A`. Exact rational comparisons select the global minimum.
Nearest ties go toward zero; alphabet ties use lexicographic positive levels,
and scale ties select the smaller positive value.

The selected alphabet is **`{−7,−4,−2,0,2,4,7}`**, with tensor multipliers
**0.814592632–0.872075326**. Every tensor has the same initial integer mapping:
positive q `0→0`, `1,2→2`, `3,4→4`, `5,6,7→7`, with odd symmetry for negative q.
The histogram-domain mapping retains every original nonzero position and sign.

| Fixed representation | Count-weighted q MSE | Tensor-weighted entropy, bit/value |
|---|---:|---:|
| Original 15 values, reference | 0 | 3.780998365 |
| Uniform seven, fixed multiplier 7/3 | 0.414538373 | 2.608748127 |
| Uniform seven, exact tensor multiplier fit | 0.390292858 | 2.618791172 |
| Selected nonuniform seven | **0.357232015** | **2.763767183** |

The selected fit reduces MSE by **13.8241%** against the existing fixed uniform
mapping, and by **8.4708%** against the scale-fitted uniform control in the same
35-state algorithm. It improves 101/111 tensors; ten tensors increase MSE.
The zero mass falls from **1,839,876 / 31.350%** to the original
**631,815 / 10.765%**. Thus 1,208,061 original `q=±1` positions retain nonzero
symbols, while the outer reconstruction magnitudes contract.

The seven aggregate counts, in alphabet order, are
`[681072, 797293, 1145747, 631815, 1141648, 794133, 677156]`.
Global entropy is 2.767493578 bit/value; the table uses the count-weighted
per-tensor entropy actually relevant to static15.

[Full tensor results](results/nonuniform-seven-histogram-fit.json) contain all
111 fitted rational multipliers, mappings, histograms and costs. The
[standard-library fit source](nonuniform-seven-fit.py) takes **11.355775 CPU
seconds** in the recorded run. Existing-census count/cost checks pass. A
minimal exactly representable toy `{0,±2,±4,±6}` with codes `{0,±1,±2,±3}`
reaches the global minimum SSE=0 at multiplier 2.

## 2. C3 expected cost and physical weight bytes

The existing C3 model has 15 positive frequencies totaling 4096: reserve one
per code, distribute the remaining 4081 proportionally, then allocate largest
remainders with the original symbol tie-break. Eight unused codes retain
frequency one. The nominal table cost remains **28 B/tensor, 3,108 B/W**.

| Representation | Expected INT8 data + nominal tables, B | Predicted complete W, B | Encoded complete W, B |
|---|---:|---:|---:|
| Original static15 | 2,776,881.052 | 2,904,325 | 2,904,325 |
| Existing uniform seven | 1,918,976.296 | 2,046,420.244 | 2,044,814 |
| Nonuniform seven | 2,032,699.118 | 2,160,143.066 | **2,159,125** |

Predicted complete size anchors the expected INT8-cost difference to the
existing encoded static15 file. The uniform prediction exceeds its known
encoded size by 1,606.244 B. Metadata adaptation, BF16 byte entropy and range
framing belong to the actual encoded-size coordinate.

On the same expected-cost basis, nonuniform seven adds **113,722.822 B/W**,
or **227,445.644 B for two weight copies**, against uniform seven. Under a
unit-transfer rate model over 586,459,020 targets, paying that difference
requires 0.002150579 additional native nats/target of improvement. The actual
payload mixer, range coder and full package supply their own measured costs.

Physical initialization applies each exact positive multiplier to every
paired signed BF16 row scale, rounds once to nearest-even, then recomputes
the nearest code from the resulting actual scale ratio. Other tensors,
activation scales, names, shapes and model architecture are retained. The
existing C3 replacement recoder and native INT8 kernels handle this alphabet.
The actual rounded ratios are **0.811764706–0.874172185**. Original absolute
BF16 scales span **6.02295991e−15–0.19140625**, and exported scales span
**5.24580379e−15–0.1611328125**. All 25,946 paired row scales preserve their
signs; the post-rounding integer mapping matches the fitted support. The
replacement recoder verifies all **434 reconstructed tensors** against the
**429 real payload tensors** and original derived values. Its C3 archive is
**2,159,125 B: 12 header bytes plus 2,159,113 range-stream bytes**. Compared
with encoded uniform seven, it costs **114,311 B/W, 228,622 B for two copies**;
compared with encoded W0, it saves **1,490,400 B for two copies**. The actual
two-copy difference corresponds to 0.002161702 nats/target under the same
unit-transfer model; the measured payload-code coordinate remains separate.

The MSE above is in **original q-squared units**, before BF16 rounding.
For original row scale `S_tj`, physical row squared error is
`S_tj**2 * SSE_q_tj`. The census aggregates q counts per tensor; global physical
MSE requires rowwise errors and scales. Native inference measures the network
response to the physical exported weights.

## 3. Initial native NLL: paired retained controls

The [completed initial-native result](results/nonuniform-seven-initial-native.json)
uses the full12 native evaluator, 205-token vocabulary, KVI8 kernels,
original priors and explicit piece boundaries. The original W0 NLL is reused.
The uniform-seven e0 comparison comes from the completed
[1,213,576-target control](results/q7-initial-native.json).

| Native scope | Targets | Retained W0 NLL | Existing uniform-seven e0 NLL | New nonuniform e0 NLL |
|---|---:|---:|---:|---:|
| Primary full-held, 473 pieces | 1,082,505 | 0.966248339 | 1.328081687 | **1.317228198** |
| Contained long-three | 107,647 | 1.258614301 | 1.669055486 | **1.666407778** |
| Independent giant | 131,071 | 0.038986426 | 0.286442469 | **0.241693964** |

The long-three targets are included in primary full-held once; the giant is
independent. The exported W SHA256 is
`2d3dceddcedeb62de4b9cba3ae15aa6fba1cb9440105b18c7e602b43af41a7bd`.
The result includes original input/evaluator identities, 473 paired pieces,
rounded row-scale hashes and both complete native outputs.

| Scope | New initial loss above W0, nats/target | NLL gain over uniform e0 | Initial quantization-loss recovery |
|---|---:|---:|---:|
| Primary | +0.350979860 | 0.010853489 | **2.999582%** |
| Contained long-three | +0.407793477 | 0.002647708 | **0.645088%** |
| Independent giant | +0.202707538 | 0.044748506 | **18.083416%** |

The original continuation criterion required approximately **85% primary
initial-loss recovery**. This gate failed. **Stop this nonuniform fixed
initialization branch and retain its completed measurements.** The q-space
MSE improvement, native NLL response and true archive fee have the distinct
values above; the actual mixed payload and closed payload-range coordinates
remain unmeasured in this assay.

### Assay source and resource result

The [physical initializer](nonuniform-seven-initializer.py) and
[actual two-call runner](nonuniform-seven-initial-native.py) preserve the
assayed source shape and its declared paths. The small
[initialization manifest](results/nonuniform-seven-initialization-manifest.json)
records the exact 111 multipliers and source histograms used by that run.

The first initializer reached the 512 MiB envelope while constructing Python
RoPE arrays, using **6.710751 CPU seconds** before any native call. A shape-only
zero-stride view defers the five derived values to the original C++ recoder;
its 434-tensor reload verifies their exact original bytes. The successful
assay uses **117.306406 CPU seconds**, including **111.411468** for the two
native calls. Cumulative CPU including the failed attempt and recorded
setup is **124.451460 seconds**. Successful peaks are **63,784 KiB self** and
**192,976 KiB child**, inside the same 512 MiB envelope. The two native calls
finish with exit zero and cover 1,213,576 independent targets.

## Reproduce the histogram result

From the repository root:

```sh
python3 -B research/nonuniform-seven-fit.py \
  --census research/results/weights-census.json \
  --out /YOUR/workspace/temp/nonuniform-seven
```

The output is `nonuniform-seven-histogram-fit.json`. For physical replay, use
the published initialization manifest and initializer with the declared
original C2/readers, then the existing `weights-recode --replace-payloads`
path. The actual native source declares its original retained cache, probe,
output and one-run resource-accounting paths; an independent replay needs
that input bundle and a fresh output/window. All source licenses and original
reader attribution remain with their existing files.
