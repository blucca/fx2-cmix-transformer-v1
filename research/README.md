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

Both variants restored each prefix exactly. At 256 KiB, coding time was 63.765s
versus 86.689s; peak RSS across encode/decode was about 5.34 versus 7.45 GiB.
The unmodified v26 binary first hit global OOM on an 8 GiB machine with no swap;
4 GiB disk swap let the same experiment finish. The 1 MiB round trips took
571.18s and 861.56s including initialization, dictionary pretraining and I/O.
The two prefixes overlap. Three independent cold-start article-aligned spans
and longer continuous-prefix measurements are running/queued. Full-file S and
formal resource qualification are pending.

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
intercept. Vocabulary overrides retain the base probability. The screen
changes neither the base model nor its future feature stream; promising
corrections still receive a real encoder/decoder experiment.

```sh
c++ -O3 -std=c++17 research/replay.cpp -o /YOUR/temp/replay
/YOUR/temp/replay --self-test
/YOUR/temp/replay /YOUR/temp/run/trace.bin
```

The source builds and synthetic known-bias/override check pass. Real auxiliary
round trips, trace equivalence and causal-calibration gains are queued; their
results will be added after execution.
