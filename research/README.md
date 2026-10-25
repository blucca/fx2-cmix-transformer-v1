# First experiment: does fxcm_v26 complement the 6M transformer?

## Fixed sources

**Latest independent component:** [lossless per-tensor weight coding](weights-static15.md)
saves 26,327 B per weight file and **52,022 B** across the two weight/executable
copies in the measured development package. All 434 tensor payloads are identical.

The separate [seven-value weight screen](weights-quant7.md) saves 885,838 B per
model copy but adds 44,118 B on the verified q2 1 MiB round trip, exceeding its
1,297-byte payload budget. This fixed mapping stops; the original model stays.
Protecting 39 recurrent-gate and input/output matrices also fails the fixed
raw-network gate: regressions are **40.07× and 53.58×** the remaining byte-budget
proxy on the two 1 MiB samples. The seven-value/GPTQ preparation branch stops;
[six measured evaluations](results/weights-quant7-protect39.json) preserve the result.

A separate [bounded neural-cache screen](neural-cache.md) improves held-out
halves by 0.00054394 nats/token at an added 3.054 microseconds/token. This is
4.53% of its 0.012 quality gate, so the fixed cache recipe stops as well.

A [read-only MLP unit census](mlp-unit-budget.md) also closes its fixed pruning
branch. The lowest-energy 3,686 of 9,216 units still carry **9.65% / 11.66%**
of the two held-out samples' local diagonal output energy, above the 0.1%
continuation gate. Only one unit is unobserved in both samples. Original
probabilities and model weights are preserved.

The original QAT model also loses accuracy when its three native attention
windows are [extended directly to 2048](attention-window.md). On the q2 1 MiB
sample, NLL rises by **0.014237 nats/token**, with **20.81%** extra raw-network
CPU time. The default 1024 window stays; this direct extension stops.

- Transformer champion: `astOwOlfo/fx2-cmix-transformer-v1@83f2603f3f7751da5f429fd32669807eb8510494`.
- Statistical expert: `blahem/cmix-lex@370e698f7ea62168cc64326ff97950c3dc212691`.
- Exact enwik9: 1,000,000,000 bytes, MD5 `e206c3450ac99950df65bf70ef61a12d`.
- Upstream int4 weights: `models/6m-q4-fp32.tfwc2`. Preserve the original dictionary, PHDA9/WRT transform, article order, PPMd and outer mixer.

The imported expert’s `Predict()` float array is a neutral compatibility view. The adapter reads `RawPredictions()` and `ActivePredictions()`, converts active logits with `RawPredictionProbability()`, and passes probabilities through the original mixer. Inactive capacity slots remain exactly 0.5. Original author and GPL notices remain in the imported files.

## Reproduce

Use a Linux CPU environment with Clang 17, make, Python 3 and sufficient memory/disk. All generated data belongs in an experiment scratch directory. Preserve `-mrecip=none` and the portable x86-64-v3 instruction target for cross-vendor arithmetic consistency.

1. `python3 research/download.py /YOUR/temp/hutter/data` downloads and checks the official zip. Compare its emitted MD5 to the fixed value above.
2. Keep a pristine checkout of the transformer champion as `upstream` and this experimental source as `candidate`.
3. `python3 research/build.py --source upstream --out /YOUR/temp/hutter/build/baseline --extra upstream/research/probe.cpp` builds from the original makefile’s compiler plan in the scratch directory. Copy the same research helpers to the pristine checkout first. Repeat with candidate and output `build/v26`.
4. Build `prepare.cpp` using the command in its header. Run `prepare ENWIK9 DICTIONARY ARTICLE_ORDER EMPTY_OUTDIR`; the output is the exact full pre-arithmetic-coding `stream.bin`, its 32-byte vocabulary mask, article boundaries, and metadata. After completion, the other generated intermediates can be removed with Python.
5. A probe uses `probe encode|decode INPUT OUTPUT VOCAB DICTIONARY WEIGHTS OFFSET LENGTH`. Both sides use the same exact full-stream vocabulary and dictionary pretraining. Output paths must be fresh.
6. `python3 research/batch.py --root /YOUR/temp/hutter --plan PLAN.json --out /YOUR/temp/hutter/runs/first-ab` processes baseline/candidate serially, verifies every round trip, and checkpoints results. Root contains `upstream`, `prepared`, and `build/{baseline,v26}`. Each plan entry has `name`, `offset`, and `length`. Completed identical trials are reused; interrupted trials are preserved and restarted. A recorded real failure stops the batch for inspection.

For long runs, `--phase encode` checkpoints completed compressed files with
status `encoded` (decode pending). A later identical invocation with
`--phase decode` reuses those files and validates exact restoration; the
source, binary, vocabulary, dictionary and weights identities must match.
`--phase both` remains the default. `active_seconds` sums completed probe
phases; `elapsed_seconds` also includes any wait between execution windows.
Run the small orchestration check with
`python3 research/batch-phase-check.py --work /YOUR/temp/batch-phase-check`.

The first plan compares a 256KiB prefix, a 1MiB prefix, and 1MiB article-aligned spans near each quartile of the complete stream. Prefixes overlap deliberately for scale diagnostics. Non-prefix spans restart statistical models and measure cold-start behavior. Larger continuous-prefix and full-file experiments follow the initial measurements.

## Measurement contract

Report encoded bytes, coding time, peak RSS and actual lossless restoration together. Development probes measure coded bytes of the preprocessed stream; the official Hutter score counts the complete compressor and self-extracting archive, including both copies of weights and all required side information. A formal record needs the complete 1GB round trip and the official single-core/time/memory limits.

## Initial measured results

Raw, verified receipts: [initial-prefixes.json](results/initial-prefixes.json).

| Prefix | Champion v22 | v26 | Saved | Relative saving |
|---|---:|---:|---:|---:|
| 256 KiB | 20,972 B | 20,954 B | 18 B | 0.0858% |
| 1 MiB | 84,988 B | 84,839 B | 149 B | 0.1753% |

The first article-aligned 1 MiB cold-start span (offset 146,621,076) also
round-tripped: 231,075 → 231,016 B, saving 59 B / 0.0255%.
[Full receipt](results/initial-article-q1.json). Its two-way times were
642.79s versus 904.20s.

The middle article-aligned 1 MiB span (offset 293,232,482) also round-tripped:
208,769 → 208,696 B, saving 73 B / 0.0350%.
[Full receipt](results/initial-article-q2.json). Its two-way times were
648.28s versus 930.58s.

The final article-aligned 1 MiB span (offset 439,844,819) round-tripped:
153,766 → 153,544 B, saving 222 B / 0.1444%.
[Full receipt](results/initial-article-q3.json). Its two-way times were
672.61s versus 908.12s. All three disjoint cold-start cases improve, with
substantial variation in the size of the gain.

Both variants restored each prefix exactly. At 256 KiB, coding time was 63.765s
versus 86.689s; peak RSS across encode/decode was about 5.34 versus 7.45 GiB.
The unmodified v26 binary first hit global OOM on an 8 GiB machine with no swap;
4 GiB disk swap let the same experiment finish. The 1 MiB round trips took
571.18s and 861.56s including initialization, dictionary pretraining and I/O.
The two prefixes overlap. Longer continuous-prefix measurements are queued. Full-file S and
formal resource qualification are pending.

### First long prefix: 8 MiB

Both unchanged, untraced binaries completed real lossless round trips on the
same VM. [Full receipts](results/initial-prefix-8m.json):

| Prepared input | Baseline | v26 | Saved | Combined encode/decode wall time |
|---|---:|---:|---:|---:|
| Prefix, 8 MiB | 1,072,279 B | 1,068,581 B | 3,698 B (0.3449%) | 4,850.31 / 6,740.03 s |

Combined coding time was 4,799.20 / 6,634.04 s; peak RSS was 6,018,532 /
7,883,680 KiB. The 8 MiB prefix contains the earlier 1 MiB sample. Cumulative
savings grew from 149 to 3,698 B across those prefix lengths, a further
3,549 B. The payload gain is 0.0035267 bits per prepared input byte. The
complete archive, its runtime, and the 1% prize gate remain unmeasured.
The next scale check retains these binaries and changes only prefix length.

### 32 MiB prefix: both lossless round trips verified

Both unchanged binaries completed real encoding and decoding of the same
33,554,432-byte prefix. Input, vocabulary, dictionary and weight identities
agree, and both restored slices match the original. [Complete receipts](results/prefix32-verified.json).

| Prepared input | Baseline | v26 | Saved |
|---|---:|---:|---:|
| Prefix, 32 MiB | 5,703,804 B | 5,696,316 B | **7,488 B (0.1313%)** |

The payload gain is **0.0017853 bits/input-byte**. These runs used different
machines; their measured phase times are reported separately:

| Variant | Encode wall seconds | Decode wall seconds | Combined coding seconds |
|---|---:|---:|---:|
| Baseline | 8,992.897 | 9,092.429 | 18,024.109 |
| v26 | 15,414.957 | 13,464.821 | 28,727.037 |

The **8→32 MiB increment is 3,790 B across 24 MiB**, or **0.001205
bits/input-byte**. Relative savings fall from 0.3449% at 8 MiB to 0.1313%
at 32 MiB. The earlier in-flight emitted-byte curve at common 8 / 16 / 24 /
28 MiB prefixes saved 3,698 / 4,451 / 5,695 / 6,590 B; its pre-flush
[snapshot remains available](results/prefix32-inflight.json).

**Stop automatic 64 MiB/full-data expansion.** Reconsider after independent
long-span or combined-model gains justify it. Complete archive size and the
full-file prize gate remain separate measurements.

### Executable byte budget

The same non-PGO Clang 17 development builds, packed with UPX 5.2.1 `-9`,
measure 181,644 B for v22 and 202,304 B for v26. The champion's packaging
includes the executable twice, so this comparison adds **41,320 B** to the
v26 byte budget before compressed dictionary/order differences. This is a
development estimate; final PGO/LTO packaging supplies the official sizes.
The cold-start coding gains therefore need a continuous long-stream test
and a complete package comparison. [Size receipt](results/development-binary-sizes.json).

The upstream packaging structure is `S = raw_payload + 2B + 2D + O + 2W + 69`:
packed binary B and compressed dictionary D occur twice, compressed article
order O once, and the 2,930,652 B neural weights W twice. The 69 bytes cover
two 16-byte package headers and the 37-byte stream header. Actual release B,
D and O are measured during final packaging.


## Completed controlled mechanism: the FXCM auxiliary skip

The fixed-capacity last FXCM slot is used twice: as a stage-1 skip input and in
the auxiliary-context selector. The imported v26 implementation emits a dense
active prefix; padding can occupy that last capacity slot.
[cmix-obias `KH_FXCM_FIX`](https://huggingface.co/dfreelan/cmix-obias/blob/51488a0c1228dbeab7c1be837fc90ceaed351728/cmix-obias/src/predictor.cpp)
selects the last *active* output instead. `HUTTER_ACTIVE_AUX` enables this one
intervention here; the full input row, padding, predictor internals and neural
weights stay fixed. The default remains the original port for comparison.

Build with `research/build.py --define HUTTER_ACTIVE_AUX`. The completed test
increased coded sizes by 5 / 14 / 18 B relative to the original v26 on the
256 KiB prefix, 1 MiB prefix and middle article-aligned 1 MiB span. The two
common 1 MiB cases therefore selected the original v26 for the 8 MiB run.
The auxiliary variant is stopped; full receipts accompany the calibration screen below.

## Small causal-calibration traces

Compile with `--define HUTTER_TRACE`. `research/batch.py --variants v26aux
--trace` sets `HUTTER_TRACE_PATH` for the encoder and verifies one complete trace
record per coded bit, along with the normal lossless round trip. Without the
compile definition there is no trace code in the encoder. Source-preserving
trace builds of v22 use the same instrumentation and retain its original
expert and auxiliary wiring.

Each native little-endian x86 record is `<HHfff>` (16 bytes): exact u16 coder
probability, flags (observed bit and byte-mixer vocabulary override), neural bit
probability, current FXCM skip probability, and raw pre-SSE mixer logit. Records
are written before `Perceive`; dictionary pretraining is excluded. A 1 MiB
slice produces a 128 MiB trace. The baseline trace run must reproduce the
already round-trip-verified encoded file byte for byte before evaluation.

`replay.cpp` screens six small online intercept calibrators: global, bit
position, confidence, neural disagreement and pre-SSE disagreement, with fixed
priors. Each prediction is scored before its label updates a bounded Newton
intercept. Vocabulary overrides retain the base probability. An exact range
counter reproduces the encoder's inclusive intervals, renormalization and final
byte; each complete replay must match its source `encoded.bin` size in
`base_coded_bytes`. Candidate `coded_bytes` and signed `saved_bytes` accompany
the log-loss diagnostics. Intermediate reports close a copy of the live range
state and report cumulative prefix sizes.

The base mixers and SSE update from their own cached predictions and the
observed bit. A correction placed after SSE and vocabulary override therefore
leaves the future feature stream unchanged. This placement permits exact
causal codelength screening from frozen traces. The selected correction still
receives a real encoder/decoder experiment.

**Integer boundary:** replay emits `round(65536*q)` frequencies. Install a
correction after the original `Discretize`, shared by encoder and decoder;
feeding `q` through the original float API quantizes it a second time. Keep
replay's strict floating-point semantics in that shared implementation. At
zero correction, all 65,535 legal frequencies must remain unchanged.

```sh
c++ -O3 -std=c++17 research/replay.cpp -o /YOUR/temp/replay
/YOUR/temp/replay --self-test
/YOUR/temp/replay /YOUR/temp/run/trace.bin
```

The source builds and synthetic known-bias/override/range-counter checks pass.

The first auxiliary-wiring runs round-tripped at **20,959 B / 84,853 B** on
the 256 KiB / 1 MiB prefixes: 5 B / 14 B larger than the original v26 port.
Exact replay matched both real encoded sizes. `neural_gap_64` saved 4 B / 34 B;
its cumulative 256 KiB-block savings were 4, 9, 21 and 34 B. These prefixes overlap. The independent middle span also
round-tripped at
**208,714 B**, 18 B larger than v26. The two prespecified 1 MiB cases sum to
293,757 / 293,535 / 293,567 B for the champion / v26 / v26aux: **retain v26**.

All five traces passed exact base-counter checks. On the champion's 1 MiB
prefix / middle span, the six online controls save −1/−1, −2/2, 17/46,
**39/50**, 18/39 and 8/37 B, respectively. On v26aux the best control saves
34/42 B. These small causal gains remain replay-only.
[Full screening receipts](results/calibration-screen.json).

The raw pre-SSE loss was lower than the final loss in both champion cases.
A fixed ablation now reports `pre_sse_coded_bytes` and `pre_sse_saved_bytes`:
override frequencies stay unchanged; other bits use
`clamp(round(65536*sigmoid(clamp(raw_logit, ±log(65535)))), 1, 65535)`.
It saves **40/66 B** on the same two cases. This is an explicit strict-integer
candidate, with no fitted parameters. Its full-stream usefulness depends on
SSE's longer adaptation curve; a real SSE-removal build has not been run.

### One frozen history-conditioned head

Inspired by the historical-error feature families in
[cmix-obias BITLSTM32](https://huggingface.co/dfreelan/cmix-obias/blob/51488a0c1228dbeab7c1be837fc90ceaed351728/cmix-obias/BITLSTM32_SPEC.md),
an additional research-only experiment fits 18 coefficients using the same
trace. Its features combine bit position, current neural/pre-SSE disagreement,
and **past** base residuals, surprise changes and relative expert loss. Each
bit is scored before its label enters the bias-corrected exponential averages.
Overrides retain their frequency and still advance history. The underlying
predictor remains unchanged.

The fixed features are: eight bit-position indicators; normalized base logit
and two expert gaps; each expert gap times its past relative loss; residual
averages at 32/512/8192-bit scales; and base logit/neural gap times the
64-versus-4096-bit surprise difference. Expert-loss averages use scale 512.
`HistoryHead` in `replay.cpp` fixes their normalization and update order.

Feature state is computed in strict double arithmetic, then rounded to 18 f32
values shared by fitting and scoring. `fit-history.py` uses the existing
NumPy/SciPy stack, streaming 64K-bit chunks through one bounded L-BFGS-B fit:
zero initialization, coefficient bounds ±2, NLL + 32‖weights‖², at most 64
iterations. It saves optimizer status and freezes 72 bytes of f32 weights.
Each 1 MiB case reserves input-byte blocks 3 and 7 of eight 128 KiB blocks.
Feature history continues through those blocks; their labels are excluded
from the fitting objective.

```sh
replay --history-features PREFIX_TRACE.bin /YOUR/temp/prefix.features.f32
replay --history-features MIDDLE_TRACE.bin /YOUR/temp/middle.features.f32
python3 research/fit-history.py \
  --case PREFIX_TRACE.bin /YOUR/temp/prefix.features.f32 \
  --case MIDDLE_TRACE.bin /YOUR/temp/middle.features.f32 \
  --out /YOUR/temp/history-fit
replay --history-score PREFIX_TRACE.bin /YOUR/temp/history-fit/weights.f32
replay --history-score MIDDLE_TRACE.bin /YOUR/temp/history-fit/weights.f32
```

The scoring mode reports exact whole-trace bytes and separate held-out block
bytes for the frozen head and all six online controls. Only block counters
reset; model histories keep running. Each whole-trace base count must still
equal the real `encoded.bin` size. The existing C++ self-check covers label
causality, overrides and zero-head frequencies; `fit-history.py --self-test`
checks the regularized objective gradient.

The first fit finished its fixed **64-iteration limit** in 79.53s, rather than
reaching the optimizer's convergence criterion. Its frozen 72-byte head saves
26/95 B on the full prefix / middle traces, which include training blocks.
The held-out 256 KiB from each case save only **4/11 B**, versus **12/12 B**
for `neural_gap_64`. Combined held-out gain is **0.000228882 bits/input-byte**,
below the prespecified 0.003 continuation threshold. A provisional 20 KiB
code reserve plus both copies of the weights also exceeds the extrapolated
held-out payload saving. **Stop this head after the initial screen.**
The main encoder/decoder remain unchanged.
[Fit and exact held-out results](results/history18-initial.json) ·
[Frozen f32 weights](results/history18-weights.f32).

## Output-neutral throughput experiment

Two independent compile definitions retain the original paths by default:

- `HUTTER_FAST_FXCM`: cmix-lex-style raw-to-stretched lookup, initialized through
  the existing `MixerInput::SetInput` clamp and logit table. Raw-value clamping
  and the exact neutral padding value are preserved.
- `HUTTER_CACHE_MIXER`: retain the context selected by `Mix` until `Perceive`.
  Context updates and that mixer's next possible map insertion occur later.
  All weight arithmetic and update ordering remain unchanged.

`fast-path-check.cpp` checks all 4,096 possible FXCM probability values plus
neutral padding. Link it with `src/mixer/{mixer,mixer-input,sigmoid}.cpp` and
`-I src`, once with and once without `-DHUTTER_CACHE_MIXER`; compare the two
stdout streams. The local check produced 50,000 byte-identical float
predictions, covering context-map growth, reuse and the 10,000-context fallback.
This is an equivalence check; end-to-end speed is measured by the probe.

The queued real gate builds both definitions on the selected v26/aux variant,
round-trips the existing 256 KiB prefix, and compares its compressed bytes
against the original verified stream. Adoption for the 8 MiB run requires
identical bytes and at least 3% less combined encode/decode coding time in this
short paired measurement. The larger run records actual wall/CPU/RSS and, when
trace-enabled, the longer causal-calibration curve. Final package size and
formal resource qualification remain full-data measurements.

The first trace-enabled v26-fast 256 KiB run restored the input and reproduced
the original v26 compressed stream byte for byte. Combined coding time was
**179.786s versus 173.026s** for the original untraced probe, about 3.9% longer.
It failed the adoption criterion in this paired measurement, so the completed
8 MiB run used the original v26 binary. [Gate and receipt](results/fast-path-initial.json).

## Completed structural experiment: retain all mx9 contexts

`Mixer::GetContextData` keeps the first 10,000 distinct contexts forever.
Every unseen context after that shares one fallback weight vector. In the
first-layer mx9 mixer, the key is
`(((previous2*16 + previous1) & 255)*256 + byte_prefix)`: at most **65,280**
possible keys. Dictionary pretraining updates byte context but leaves these
outer mixer maps empty.

Reconstructing keys from the actual prefix / disjoint middle traces finds:

| 1 MiB case | Distinct mx9 keys | Fallback bits | Share of final loss on those bits |
|---|---:|---:|---:|
| Prefix | 33,088 | 6.52% | 13.22% |
| Middle | 44,322 | 11.92% | 17.34% |

Pretraining state, MSB-first reconstruction, original input hashes and the
complete key domain were checked. These loss shares locate the bottleneck;
the completed paired run below measures the actual coding effect.
[Observation and validation](results/context-admission-observation.json).
Reproduce key counts with `python3 research/context-cap-observe.py TRACE.bin`.

`HUTTER_MX9_FULL_CONTEXT` raises **only mx9** to a 65,536-context capacity.
Other caps remain 10,000; learning rates, SSE and neural/statistical experts
stay fixed. `context-cap-check.cpp` checks original fallback behavior and
retention of all 16-bit keys. The first coding trial applies this isolated
patch to the **pristine v22 champion**, using its existing 1 MiB prefix/middle
goldens. At 461 input weights, all remaining legal mx9 keys would add about
97.2 MiB of weight storage, plus map/allocator overhead. There are no frozen
weights to append to an archive.

The prespecified short-screen required a combined saving of 0.003
bits/input-byte (about 787 B across both cases), positive results on both,
and acceptable measured memory/runtime costs before an 8 MiB test.
The completed decision is below. Further cap values and learning-rate
variants remain unqueued.

### mx9 capacity screen: completed, stopped

[Full receipts](results/mx9-cap-initial.json). The two candidate round trips
passed, the first 132,950 / 60,301 trace records matched the original baseline,
and exact range replay matched both actual compressed sizes.

| 1 MiB case | Baseline | cap 65,536 | Saved | Peak RSS increase |
|---|---:|---:|---:|---:|
| Prefix | 84,988 B | 84,988 B | 0 B | 44,372 KiB |
| Independent q2 | 208,769 B | 208,766 B | 3 B | 65,416 KiB |

The combined saving was 0.00001144 bits/input-byte versus the prespecified
0.003 gate. Stop this cap increase; keep the default 10,000-context cap. Its
fallback traffic was a real structural observation, while the additional
contexts brought almost no final coding gain in these samples.

## PPMd reliability: observe first, calibrate causally

`HUTTER_PPMD_TRACE`, together with `HUTTER_TRACE`, adds an independent
16-byte observation sidecar. The original trace layout stays unchanged.
The new native little-endian x86 record is `<fffHH>`:

| Field | Available information |
|---|---|
| `ppmd` | Actual raw PPMd P(bit=1), before mixer/coder clamping |
| `entropy` | Current byte prior's vocabulary-masked, normalized entropy in bits |
| `escape` | First prepared PPMd SQ escape frequency/total, before masking |
| `suffix_depth` | Actual suffix-link count from MaxContext to the root |
| `flags` | Bit 0: a real ByteUpdate has prepared this prior |

The three byte-level features are prepared after the preceding byte and remain
fixed for the current eight bits. Both records are written before Perceive.
The initial byte retains ByteModel's uniform 256-symbol prior:
`(.5, 8, -1, 0, 0)`. Dictionary pretraining contributes no records.
The observer reads the existing PPMd result and distribution; it preserves
the model feedback and the original arithmetic coding path.

For a scratch root containing the original prepared data and golden trials:

```sh
# SCRATCH is an absolute caller-owned scratch directory.
python3 research/build.py --source . --out "$SCRATCH/build/v26-ppmd" \
  --extra research/probe.cpp --define HUTTER_TRACE --define HUTTER_PPMD_TRACE
mkdir -p "$SCRATCH/ppmd-trial"
HUTTER_TRACE_PATH="$SCRATCH/ppmd-trial/trace.bin" \
HUTTER_PPMD_TRACE_PATH="$SCRATCH/ppmd-trial/ppmd.bin" \
TMPDIR="$SCRATCH/ppmd-trial" \
  "$SCRATCH/build/v26-ppmd/probe" encode "$SCRATCH/prepared/stream.bin" \
  "$SCRATCH/ppmd-trial/encoded.bin" "$SCRATCH/prepared/vocab.bin" \
  dictionary/english.dic models/6m-q4-fp32.tfwc2 0 262144
cmp "$SCRATCH/ppmd-trial/encoded.bin" \
  "$SCRATCH/runs/first-ab/prefix-256k/v26/encoded.bin"
c++ -O3 -std=c++17 research/ppmd-replay.cpp -o "$SCRATCH/ppmd-replay"
"$SCRATCH/ppmd-replay" --self-test
"$SCRATCH/ppmd-replay" "$SCRATCH/ppmd-trial/trace.bin" \
  "$SCRATCH/ppmd-trial/ppmd.bin"
```

For the unchanged v22 expert, apply [baseline-ppmd.patch](baseline-ppmd.patch)
to a separate checkout of the pinned `83f2603` champion, then copy this
repository's `research/{build.py,probe.cpp,trace.h}` into its research folder.
Build that source with the same two definitions. The patch changes observation
only and preserves the original v22 expert and mixer wiring.

The fixed replay bank contains base, the existing `neural_gap_64` control,
a four-dimensional old-information Newton head, and three nested additions:
PPMd gap; gap times entropy/escape/depth; and gap times its past relative-loss
EMA512. All heads start at zero, use prior64, clip weights to ±16 and the
logit correction to ±1, and update after scoring the current bit. Cold-first-byte
and vocabulary-override frequencies stay exact, with no fitting or EMA update.
The JSONL protocol line records every feature and parameter. Every 128 KiB
reports cumulative and separately closed block range counts; model history
continues across block boundaries.

The observation check compares plain/traced PPMd predictions on 1,024 bytes.
The replay self-test covers the integer boundary, label causality, overrides
and a synthetic case with genuinely new side information. Real evaluations
add two gates: byte-identical golden encoded output and exact base replay size.

### First real observation gate: 256 KiB

Both instrumented encoders reproduced their already losslessly verified
golden streams exactly. Both paired traces have one 16-byte record per real bit.
[Full receipts and prespecified protocol](results/ppmd-prefix-256k.json).

| Expert family | Real/base bytes | Old-info replay bytes | PPMd-gap replay bytes |
|---|---:|---:|---:|
| Champion v22 | 20,972 | 20,960 | 20,951 |
| v26 | 20,954 | 20,943 | 20,927 |

The added PPMd gap saves 9/16 B beyond the old-information control on this
one prefix. Quality and reliability additions give 20,951/20,951 B for v22
and 20,928/20,928 B for v26. These are replay-only results on a small, overlapping
prefix. The fixed next screen is prefix1MiB plus the middle article1MiB; its
choice is frozen before the two independent article spans and continuous8MiB
are scored. Full archive size, runtime qualification and prize eligibility
remain later measurements.


### Frozen PPMd heads: independent 1 MiB cold validation

The fixed prefix1MiB + middle1MiB screen selected `ppmd_quality_64` for
baseline and `ppmd_reliability_64` for v26 **before** either validation span.
All four new observation streams equal their original, losslessly verified
golden files; each base replay size is exact.
[Selections, receipts and 128 KiB curves](results/ppmd-cold-validation.json).

| Family | Selected head: q1 / q3 saved | Combined bits/input-byte | Old-info combined | Neural control combined |
|---|---:|---:|---:|---:|
| Baseline | 92 / 72 B | 0.0006256 | 98 B | 128 B |
| v26 | 98 / 76 B | 0.0006638 | 75 B | 99 B |

Both heads improve both cold spans and beat the old-information control.
The magnitude falls below the prespecified 0.003 bits/input-byte cold gate.
The original continuous 8 MiB experiment used its previously fixed alternative
gate: at least 0.003 bits/input-byte and positive last-4MiB increment. Its
completed decision follows below. The middle screen's gains concentrate near startup and its last
128 KiB regresses slightly, making the long curve especially informative.
These heads remain outside the main coder.


### Frozen PPMd heads: 8 MiB long gate completed

[Long trials, fixed decisions and cumulative curves](results/ppmd-long-validation.json).
Both observer encodes exactly reproduce the already verified 8 MiB golden
streams; both base range replays match their actual compressed sizes.

| Family / frozen head | Replay saving | Bits per prepared byte | Saving in the last 4 MiB |
|---|---:|---:|---:|
| v22 / `ppmd_quality_64` | 435 B | 0.000415 | 288 B |
| v26 / `ppmd_reliability_64` | 345 B | 0.000329 | 209 B |

The prespecified long gate requires 0.003 bits/input-byte (3,146 B here)
and a positive second-half increment. Both fail the magnitude gate, so
**stop these frozen PPMd heads**. Relative to the old-information linear
control, their extra savings are 35 / 53 B. The separate neural-gap control
saves 907 / 720 B in the PPMd replay convention, also below the component gate.

Reusing the same traces for the previously specified pre-SSE bypass gives
428 / 111 B on the long prefix. On v26's disjoint q1/q3 cold spans it loses
3 / 12 B. Keep the original SSE and main coder; its removal is not selected.
The original six-control replay's baseline neural-gap count differs by one
byte because the PPMd experiment additionally bypasses the first byte.

### Byte-level mixture capacity: completed, stopped

A separate diagnostic asks whether the eight bit predictions hide useful
within-byte expert correlation. A fixed prior over base/neural/PPMd byte
likelihoods yields a causal Bayesian expert-posterior update as bits arrive.
`byte-simplex-budget.cpp` fits the convex constant-prior optimum over the
same two screening traces; its self-check covers an interior simplex edge,
a dominated-expert boundary and the Bayesian likelihood identity.

```sh
c++ -O3 -std=c++17 research/byte-simplex-budget.cpp -o "$SCRATCH/byte-budget"
"$SCRATCH/byte-budget" --self-test
"$SCRATCH/byte-budget" "$SCRATCH/runs/ppmd-observe/v26/prefix-1m" \
  "$SCRATCH/runs/ppmd-observe/v26/article-q2-1m"
```

The best baseline/v26 priors assign **0.10359 / 0.08880** to the neural
expert, **zero to PPMd**, and the remainder to base. Best in-screen ideal
savings total only **73.51 / 57.54 B** across 2 MiB, below the existing
PPMd-quality heads' **167 / 181 B** exact savings. Stop the constant-prior
mixture here. [Full capacity results](results/byte-simplex-capacity.json).

### Profile-guided CM3 prefetch: pooled timing gate failed

A 30-second, 2,938-sample user CPU-clock profile of the v26 PPMd observer
attributes 15.15% to `ContextMap3::mix` and 9.33% to `E1<14,128>::get`.
Within the latter, 67.15% lands immediately after the initial bucket load.
This motivates one opt-in `HUTTER_CM3_PREFETCH` experiment: prefetch current
state bytes and imminent hash buckets before the original serial updates.
Prediction arithmetic, update order and capacities stay fixed.

The planned gate is a real 256 KiB golden-stream round trip, then a same-machine
1 MiB encode/decode timing pair. A gain of at least 3% triggers one disjoint
1 MiB confirmation in reversed variant order. Golden bytes must match in all
cases; aggregate coding gain must reach 3% with a positive confirmation.
The flag stays off by default. The 256 KiB round trip and both 1 MiB round
trips have now passed, all byte-identical to the original golden streams.
The fresh same-machine pair measures **718.483 → 690.875 s** of combined
encode/decode coding time, a **3.8425%** improvement. The independent q2
reverse-order confirmation then measured **778.837 → 773.432 s**, a **0.6940%**
improvement, with both 208,696 B streams exactly golden. Together the pairs
measure **1,497.320 → 1,464.308 s**, only **2.2048%**, below the 3% gate.
**Stop this prefetch variant**; its opt-in macro remains off.
[Initial measured receipts](results/cm3-prefetch-initial.json) and
[completed paired experiment](results/cm3-prefetch-screen.json).


### Upstream PGO/LTO flags: a separate arithmetic family

`build.py` now supports the upstream `prof_gen` / `prof_use` compiler plans
without changing the source checkout. The 30 ordinary objects use PGO (and
LTO for profile-use); all nine transformer objects keep their original flags.
A research main stays unprofiled and non-LTO, while its final link and shared
objects retain the release flags. This keeps the probe caller stable and
avoids confusing its `main` with the profiled upstream runner.

```sh
SOURCE=$(pwd)
python3 research/build.py --source "$SOURCE" --out "$SCRATCH/build/profile-gen" \
  --profile-generate "$SCRATCH/pgo"
(cd "$SCRATCH"; ./build/profile-gen/cmix -c "$SOURCE/prof_input/input" profile.cmix)
llvm-profdata-17 merge -output="$SCRATCH/pgo/default.profdata" "$SCRATCH/pgo/"*.profraw
python3 research/build.py --source "$SOURCE" --out "$SCRATCH/build/v26pgo" \
  --profile-use "$SCRATCH/pgo/default.profdata" --extra research/probe.cpp
python3 research/build-profile-check.py --work "$SCRATCH/build-plan-check"
```

The fixed experiment uses the original 50,051-byte profile input and training
command once, with no dictionary or transformer, matching upstream's recipe.
It retains the original v26 algorithm, independently of the prefetch variant.
After the existing timing queue finishes, one CPU runs a 256 KiB golden
round trip, a fresh prefix1MiB A/B, and a reverse-order q2 only if the prefix
gain reaches 3%. Final retention requires all original golden bytes,
aggregate gain of at least 3%, and positive q2 gain. A changed stream ends
this output-neutral comparison and is recorded as a separate compile family.
[Prespecified protocol](results/release-flags-protocol.json).
The actual build and single profile training completed. Its 256 KiB real round
trip passes at **20,954 B**, matching the old size while changing the encoded
bytes. Combined coding is **166.987 s**; fresh same-machine baseline timing
is still needed. The original neutral comparison therefore stops as prescribed.
[Initial measurement](results/release-flags-initial.json).

A [separate fixed-family protocol](results/release-family-protocol.json) reuses
these two built binaries, with zero retraining or flag sweeps. A fresh prefix
pair must gain at least 3% coding speed while adding at most 16 encoded bytes
to trigger reverse-order q2. Retention requires the pooled 3% speed gate,
positive q2 speed gain, each byte delta ≤16 B, and real lossless round trips.
The original variant still reproduces its old golden; the PGO variant stores
its own golden stream and hash. This tests a speed/size tradeoff while keeping
the old traces and arithmetic family separate. Full-data qualification follows
a retained candidate.


## First-layer mixer support: observation-only interface

The single second-layer mixer globally combines 23 first-layer predictions.
The next information experiment asks whether each selected context's actual
update count and shared-fallback status explain residual reliability.
`HUTTER_MIXER_TRACE` (requires `HUTTER_TRACE`) adds a separate
`HUTTER_MIXER_TRACE_PATH` file; the normal build keeps these counters and I/O off.

Each 72-byte record contains 23 IEEE binary16, ties-to-even **clamped logits
actually received by layer two**, followed by 23 support bytes, flags=1 and
two zero bytes. Each support byte uses bits 0–5 for
`floor(log2(1 + saturated_uint32_applied_updates))` (0–32), bit 7 for shared
fallback, and leaves bit 6 zero. The snapshot belongs to the current Mix,
before its label; an early-returned Perceive applies no counted update.
Dictionary pretraining produces no rows. Quantization affects the sidecar only.

`mixer-support-check.cpp`, linked as described in its header, produced identical
49,152-byte prediction streams with and without observation across 4,096
selections and three mixers. It checks context reuse/fallback, pre-label
snapshots, early returns, count saturation, half rounding and layout.
[Local check receipt](results/mixer-support-interface.json).
The first full-model 256 KiB observation has passed: **20,954 B**, exactly
the original golden stream and base replay, with 2,097,152 valid 72-byte rows.
[Real observation receipt](results/mixer-support-prefix-256k.json).
Each following screen repeats the exact base-count and sidecar-layout checks.
Both 1 MiB traces have now passed those checks. Their fixed linear heads
missed the held-out magnitude gate below; the single nonlinear interaction
screen also failed that gate, reusing the same traces.


### Fixed support screen: contiguous tails, then frozen cold validation

The [prespecified protocol](results/mixer-support-protocol.json) compares just
two nested, frozen offset heads. `logit27` adds the 23 first-layer gaps to the
same four old-information features. `support73` adds, per mixer, an independent
context's gap times `2^(-support_bucket/2)` and a separate shared-fallback gap.
This keeps a fallback's aggregate update count separate from local reliability.
All gaps come from the sidecar's **half-rounded** logits; the common feature and
weight boundary is f32, followed by an ordered double-precision dot product.

Each of the two 1 MiB screens uses its first 768 KiB for fitting and its last
256 KiB as a contiguous holdout. Later training features therefore never absorb
updates from these holdout labels. Both fits start at zero and use the same
convex offset-logistic loss, ridge Hessian 64, coefficient box ±16,
64 L-BFGS-B iterations / 96 function evaluations, and no logit-correction clip.
Vocabulary overrides keep their exact original frequency and skip fitting.

```sh
c++ -O3 -ffp-contract=off -std=c++17 research/mixer-support-replay.cpp \
  -o "$SCRATCH/mixer-replay"
"$SCRATCH/mixer-replay" --self-test
# Python fitting uses NumPy and SciPy; BLAS is limited to one thread.
python3 research/fit-mixer-support.py --self-test
for name in prefix-1m article-q2-1m; do
  "$SCRATCH/mixer-replay" --export \
    "$SCRATCH/runs/mixer-support/v26/$name/trace.bin" \
    "$SCRATCH/runs/mixer-support/v26/$name/mixer.bin" \
    "$SCRATCH/$name-features.f32"
done
for model in logit27 support73; do
  python3 research/fit-mixer-support.py \
    --case "$SCRATCH/runs/mixer-support/v26/prefix-1m/trace.bin" "$SCRATCH/prefix-1m-features.f32" \
    --case "$SCRATCH/runs/mixer-support/v26/article-q2-1m/trace.bin" "$SCRATCH/article-q2-1m-features.f32" \
    --model "$model" --out "$SCRATCH/$model-fit"
done
"$SCRATCH/mixer-replay" --score BASE_TRACE.bin MIXER_SIDECAR.bin \
  "$SCRATCH/logit27-fit/weights.f32" "$SCRATCH/support73-fit/weights.f32"
```

The two held-out tails must each save at least zero exact coded bytes and
jointly save at least 0.003 bits/input-byte: **197 B across 524,288 input B**.
Select the support head only if it also saves at least 16 B beyond `logit27`,
a screening allowance for a provisional 16 KiB incremental package cost.
Otherwise select `logit27` if it passes, or stop this fixed experiment. Selection
and both weight files are frozen before collecting q1/q3; these cold spans must
again be individually nonnegative and jointly reach 0.003 bits/input-byte.
A selected support head must keep a positive cold increment over `logit27`.
Actual package cost and continuous-length behavior follow a cold pass.

The feature expansion occupies 4.5625 GiB for both screens, is streamed through
64K-row batches, and is discarded after freezing/scoring; original traces stay.
Fit after the heavy encoder exits on an 8 GiB machine, with a combined 30 CPU
minute fitting cap. Completed fits include iteration, evaluation, convergence
and elapsed-time receipts. This measures a fixed-budget fit rather than a
certified capacity optimum.

[Local check receipt](results/mixer-support-fit-check.json): all 65,535 zero-head
frequencies exactly match base; support/fallback, half decode, overrides and
current-label independence pass. The 73-dimensional analytic gradient matches
finite differences. A 65,536-row paired-file smoke check reproduces the f32
features, exact zero-head range counts and the contiguous tail split.

### Fixed linear screen result: stop both heads

| Frozen head | Full two-screen savings | Prefix tail | q2 tail | Combined tail |
|---|---:|---:|---:|---:|
| `logit27` | 323 B | 16 B | 15 B | 31 B |
| `support73` | 434 B | 23 B | 24 B | 47 B |

Both tails are positive and both heads fall below the prespecified **197 B**
gate. Stop their integration and cold collection. The full-screen totals include
training data; the held-out tails decide the gate. Both fits reached the fixed
64-iteration cap, with 73/75 evaluations and projected gradients
4.14e-7/7.89e-7. These are bounded-fit results.
[Exact scores, fits and process receipts](results/mixer-support-screen.json),
[frozen logit27 weights](results/mixer-support-logit27.f32),
[frozen support73 weights](results/mixer-support-support73.f32).

### One nonlinear interaction screen: MLP16

The same first-layer traces also support a distinct hypothesis: expert reliability
depends jointly on other experts and completed-byte context. The frozen
[MLP16 protocol](results/mixer-mlp-protocol.json) tests **88 → 16 tanh → 1**,
with 1,441 f32 parameters, above the existing frozen `logit27` offset.
Inputs are its 26 nonconstant features, 23 support buckets, 23 fallback flags,
eight bit-position indicators, and eight completed bytes. Current/incomplete
bytes stay out of that history; overrides advance history but retain base coding.

```sh
c++ -O3 -ffp-contract=off -std=c++17 research/mixer-mlp-replay.cpp \
  -o "$SCRATCH/mixer-mlp-replay"
"$SCRATCH/mixer-mlp-replay" --self-test
python3 research/fit-mixer-mlp.py --self-test
for name in prefix-1m article-q2-1m; do
  "$SCRATCH/mixer-mlp-replay" --features \
    "$SCRATCH/runs/mixer-support/v26/$name/trace.bin" \
    "$SCRATCH/runs/mixer-support/v26/$name/mixer.bin" \
    "$SCRATCH/$name-mlp88.f32"
done
python3 research/fit-mixer-mlp.py \
  --case "$SCRATCH/runs/mixer-support/v26/prefix-1m/trace.bin" "$SCRATCH/prefix-1m-mlp88.f32" \
  --case "$SCRATCH/runs/mixer-support/v26/article-q2-1m/trace.bin" "$SCRATCH/article-q2-1m-mlp88.f32" \
  --theta27 research/results/mixer-support-logit27.f32 --out "$SCRATCH/mlp16-fit"
"$SCRATCH/mixer-mlp-replay" --score BASE_TRACE.bin MIXER_SIDECAR.bin \
  research/results/mixer-support-logit27.f32 "$SCRATCH/mlp16-fit/weights.f32"
```

The first 768 KiB per screen supply the training rows; the final 256 KiB remain
held out. A single seed, width and 64-iteration/96-evaluation fit is capped at
30 CPU minutes, within a 45 CPU minute experiment. Its 5.5 GiB expanded features
are temporary and use the existing paired traces. The exact replay must save
at least 197 B across both tails, each nonnegative, and at least 48 B beyond
the best linear head. A pass freezes weights for a separate cold-validation
decision. A failure stops this fixed head. Original heavy-model binaries stay
unchanged. [Small implementation checks](results/mixer-mlp-check.json) cover
the gradient, causal byte history, untouched holdout, and exact zero-head coding.


### MLP16 screen result: stop this fixed head

Exact held-out savings are **15 / 14 B**, totaling **29 B** against the 197 B
floor. This is 2 B worse than `logit27` and 18 B worse than `support73`.
The full screens save 112 / 244 B, including training rows: only 33 B beyond
the linear control. Stop this fixed head's cold collection and integration.

The fit reached its prescribed 64-iteration cap after 70 evaluations,
with projected gradient 3.08e-5; it used 1,355.18 CPU seconds, and the full
experiment used 1,382.05. This records the completed bounded optimization,
with convergence status and training/held-out results kept separate.
The 5.5 GiB feature expansion has been removed; paired traces and the
[frozen 5,764-byte weights](results/mixer-mlp16.f32) remain reusable.
[Exact scores, optimizer and process receipts](results/mixer-mlp-screen.json).


### Bit-level recurrent residual: fixed LSTM32 experiment

The next hypothesis uses information absent from the earlier MLP: the previous
coded bit, a 64-bit recurrent state, and individual experts' recent losses.
The content/history features follow the [cmix-obias BITLSTM32 specification](https://huggingface.co/dfreelan/cmix-obias/blob/51488a0c1228dbeab7c1be837fc90ceaed351728/cmix-obias/BITLSTM32_SPEC.md),
with its constant XML and corpus-specific position columns removed. Our neural
expert is explicitly the half-rounded logit of the raw neural probability;
it differs from the champion's table-stretched layer input.

**78 inputs → 32 SiLU → LSTM32 → 1 residual**, with 11,009 f32 weights and
original base-logit offset. The previous bit and completed-byte history are
strictly causal. Each aligned 64-bit sequence starts with zero recurrent state;
feature windows continue. Vocabulary overrides retain their exact base frequency
while advancing all state. One shared C++ feature implementation produces both
training data and strict-double frozen replay. Its frequency rule preserves
all 65,535 base frequencies for a zero head.

```sh
c++ -O3 -ffp-contract=off -std=c++17 research/bit-recurrent-replay.cpp \
  -o "$SCRATCH/bit-recurrent"
"$SCRATCH/bit-recurrent" --self-test
python3 research/fit-bit-recurrent.py --self-test
"$SCRATCH/bit-recurrent" --export BASE_TRACE.bin MIXER_SIDECAR.bin FEATURES.f32
python3 research/fit-bit-recurrent.py \
  --case PREFIX_BASE.bin PREFIX_FEATURES.f32 \
  --case Q2_BASE.bin Q2_FEATURES.f32 --out "$SCRATCH/recurrent-fit"
"$SCRATCH/bit-recurrent" --score BASE_TRACE.bin MIXER_SIDECAR.bin \
  "$SCRATCH/recurrent-fit/weights.f32"
```

The [fixed protocol](results/bit-recurrent-protocol.json) uses two existing
1 MiB v26 traces, their first 768 KiB for training, and their last 256 KiB
for one final frozen integer evaluation. AdamW runs exactly 10 epochs
(15,360 steps), batch 128 sequences, seed 237, within a 7,200 CPU-second
fitting ceiling. Checkpoints save optimizer, RNG and next sequence position.
A budget-interrupted fit is recorded as incomplete. The initial screen requires
nonnegative savings on both tails and at least 197 B combined. Full compression
and integration require separate mature/independent-data and resource evidence.

[Implementation checks](results/bit-recurrent-check.json) cover causal state,
integer identity, nonzero Torch/C++ inference parity, training-only access, and
exact next-step checkpoint continuation. The 78-float expansion occupies
4.875 GiB and is removed after scoring; original paired traces remain.
Training and inference use CPU only. The completed cold-screen result follows;
full-package results remain pending.


### LSTM32 cold screen: positive signal, size gate failed

The fixed **10 epochs / 15,360 optimizer updates** completed in 418.4 CPU
seconds; the full experiment used 521.2. Frozen exact savings are **52 / 65 B**
on the two held-out tails, totaling **117 B**, below the 197 B screen floor.
The full screens save 408 / 510 B (918 B), including training rows. Both base
range counts reproduce their original 84,839 / 208,696-byte streams exactly.

This improves on the earlier best linear tail total of 47 B, while its larger
weight/runtime cost makes the current cold fit too small for adoption. Stop
this cold fit's integration and parameter sweep. The separately planned mature
prefix8 observation and frozen-weight diagnosis are complete below. The
4.875 GiB feature expansion has been removed; the original traces, optimizer checkpoints and
[frozen 44,036-byte weights](results/bit-recurrent-cold.f32) remain.
[Exact block scores, training metrics and process receipts](results/bit-recurrent-screen.json).


### Frozen LSTM32 mature replay: size gate failed

The original **44,036-byte cold weights** were replayed once through the full
8 MiB prefix, preserving feature history and aligned seq64 resets. Before
scoring, the new decision holdout was fixed to **[7,8) MiB**. The first 7 MiB
and the full-prefix score are diagnostic: they include the original training
interval and its previously observed cold tail.

The new mature holdout saves **154 B**. Combined with the previously observed
q2 cold-tail result of **65 B**, this gives **219 B over 1.25 MiB**, below
the frozen **492 B / 0.003 bits-per-input-byte** floor. Both tails are positive;
the size gate fails. **Stop this frozen cold-weight mature extension.** The
original cold-screen failure stays recorded; further fitting or integration
requires a separate decision.

The diagnostic full-prefix range count is **1,068,581 → 1,067,352 B**
(1,229 B saved). Changes in cumulative savings over successive MiB are
**408, 213, 204, −75, 242, 12, 71, 154 B**; [3,4) MiB regresses. The
observer stream exactly matches the original verified v26 golden stream,
and the replay base count is exact. Candidate figures here are frozen
integer range-replay counts. Production integration is a separate decision.

This one-pass diagnosis used **353.03 CPU seconds / 353.45 wall seconds**
on one CPU, reusing the original strict-double ELF. New training, compilation
and feature expansion were all zero. [Frozen protocol, input identity, exact
scores, MiB trends and process receipts](results/bit-recurrent-mature.json).

```sh
"$SCRATCH/bit-recurrent" --score PREFIX8_BASE.bin PREFIX8_MIXER.bin \
  research/results/bit-recurrent-cold.f32 7340032
```


## Same-weight positive activation quantization

The [fixed u8 activation screen](positive-activation.md) passes neural A:
weighted raw NLL improves by 0.0007830732 nats/token, measured evaluation time
rises 0.75%, and the development double-UPX package adds 24 B.

Real coder B completes all four lossless round trips with exact baseline
goldens. The two 1 MiB slices save **76 / 151 B**, **227 B** together, at
**+2.99%** combined coding time. The fixed **263 B** size gate fails; stop
longer confirmation for this recipe. The switch stays disabled by default.
[Complete B receipts](results/positive-activation-B.json).

The original v26 middle 8 MiB comparison is now [fully verified](results/middle8-verified.json):
1,465,827 → 1,464,162 B, saving 1,665 B / 0.1136%. The independently compiled
[PGO family](results/release-family-verified.json) also completes both paired
1 MiB round trips: total coding time improves 4.50%, with 0 / -1 B size deltas.
It retains its separate stream family and awaits any justified long confirmation.


## Causal online hidden-state readout

The [fixed dynamic readout screen](dynamic-head.md) learns a zero-initialized
205×192 residual from final normalized Transformer states. Both 1 MiB slices,
their four halves and all 64 blocks improve; pooled gain is 0.0015339819
nats/token against the predeclared 0.002 gate. Stop this recipe before paired
timing/coder integration. Original base probability fingerprints and NLL sums
are preserved. [Frozen protocol and measurements](results/dynamic-head-A.json).


## Context-balanced learning age: stopped at the size gate

A [default-disabled schedule experiment](context-age.md) replaces the global
learning age of an independent mixer key with its own exposure count times
the number of admitted keys. Shared fallback keeps its original schedule.
The default golden and all candidate round trips pass. The two 1 MiB slices
save 43 / 89 B, totaling **132 B against the fixed 263 B gate**. This recipe
stops before paired timing, packaging or longer work.
[Measured results](results/context-age.json).


## Metadata: old page order and a new revision clock

A [causal numeric rewrite](metadata-residual.md) separates two observed
revision-ID regimes. On all 243,425 header records, the complete reversible
rewrite saves **823,081 B with zlib-9**, including every residual marker.
This direction measurement precedes the unchanged v26 coder test. The fixed
screen uses two disjoint 16,384-record cases, charges 16 B framing per
candidate, verifies real compression plus inverse restoration, and requires
at least 32,768 B pooled savings before any longer run.


### Metadata screen outcome

The fixed two-regime ASCII rewrite completed all four real coder round trips.
Clock-heavy records saved 4,596 B and page-heavy records grew 1,516 B, after
16 B framing per case. The 3,080 B combined saving missed the 32,768 B gate.
This recipe and its full-tail extension stop. The unchanged-model,
original-header field-cost observation follows the small [result](results/metadata-coder-screen.json)
and [mechanism report](metadata-residual.md).
