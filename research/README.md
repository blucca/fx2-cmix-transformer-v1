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

Both variants restored each prefix exactly. At 256 KiB, coding time was 63.765s
versus 86.689s; peak RSS across encode/decode was about 5.34 versus 7.45 GiB.
The unmodified v26 binary first hit global OOM on an 8 GiB machine with no swap;
4 GiB disk swap let the same experiment finish. The 1 MiB round trips took
571.18s and 861.56s including initialization, dictionary pretraining and I/O.
The two prefixes overlap. The remaining cold-start article-aligned span and longer continuous-prefix
measurements are running/queued. Full-file S and
formal resource qualification are pending.

### Executable byte budget

The same non-PGO Clang 17 development builds, packed with UPX 5.2.1 `-9`,
measure 181,644 B for v22 and 202,304 B for v26. The champion's packaging
includes the executable twice, so this comparison adds **41,320 B** to the
v26 byte budget before compressed dictionary/order differences. This is a
development estimate; final PGO/LTO packaging supplies the official sizes.
The cold-start coding gains therefore need a continuous long-stream test
and a complete package comparison. [Size receipt](results/development-binary-sizes.json).

## Next controlled mechanism: the FXCM auxiliary skip

The fixed-capacity last FXCM slot is used twice: as a stage-1 skip input and in
the auxiliary-context selector. The imported v26 implementation emits a dense
active prefix; padding can occupy that last capacity slot.
[cmix-obias `KH_FXCM_FIX`](https://huggingface.co/dfreelan/cmix-obias/blob/51488a0c1228dbeab7c1be837fc90ceaed351728/cmix-obias/src/predictor.cpp)
selects the last *active* output instead. `HUTTER_ACTIVE_AUX` enables this one
intervention here; the full input row, padding, predictor internals and neural
weights stay fixed. The default remains the original port for comparison.

Build with `research/build.py --define HUTTER_ACTIVE_AUX`. The queued test uses
the 256 KiB prefix, 1 MiB prefix and middle article-aligned 1 MiB span. Selection
between v26 and v26aux uses summed coded bytes on the two 1 MiB cases, followed
by a scale-up comparison with the champion on a continuous 8 MiB prefix.

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
Real auxiliary round trips, trace equivalence and causal-calibration gains are
queued; their results will be added after execution.

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
