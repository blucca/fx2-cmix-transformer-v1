# First experiment: does fxcm_v26 complement the 6M transformer?

## Fixed sources

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

### Executable byte budget

The same non-PGO Clang 17 development builds, packed with UPX 5.2.1 `-9`,
measure 181,644 B for v22 and 202,304 B for v26. The champion's packaging
includes the executable twice, so this comparison adds **41,320 B** to the
v26 byte budget before compressed dictionary/order differences. This is a
development estimate; final PGO/LTO packaging supplies the official sizes.
The cold-start coding gains therefore need a continuous long-stream test
and a complete package comparison. [Size receipt](results/development-binary-sizes.json).

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
