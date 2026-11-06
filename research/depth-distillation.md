# Eight-block causal distillation

The first fixed eight-block native family compiled with the retained production
flag plan: 20 commands, 19.289341 CPU seconds, single CPU, 1 GiB build AS limit,
and four native executables. The [build receipt](results/depth-native-build.json)
records its actual sources and terminal result. Its full probe and runner contain
the current raw-v26 FXCM. These binaries provide the NN/RAW shape acceptance;
the mixed deployment below retains the original v22 codec objects. Student,
gradient and codec calls were **0** at that first build publication.

## One student, one complete method

Keep original blocks **0,1,2,3,8,9,10,11** and renumber continuously. This gives
six KDA blocks, two attention blocks, 192 hidden dimensions, 768 MLP dimensions,
three 64-wide heads and the original 1024 attention window. Four retained skip
pairs use the original destination weights at indices **2,3,4,5**.

The genuine student schema is 289 state tensors: 75 INT4 matrices, 75 weight
scales, 80 activation scales and 59 raw F32 tensors. Its complete weight file
adds the three original RoPE tensors and two eight-block configuration tensors.
[CPU construction/export](depth_student.py) strictly maps original keys/shapes,
keeps the three original RoPE arrays, writes a genuine eight-block v2 template,
and passes it through the existing lossless static15 recoder. The original
12-block template/replacement path stays fixed.

The [original histogram estimate](depth-distillation-contract.md) suggests about
1.816 MB reduction in two weight copies. Training supplies the actual weight
file and prediction-loss tradeoff. Final score uses each side's executable and
weight size: `S=P+2B+2D+O+2W+69`.

## Frozen learning recipe

Reuse the complete-piece, article-cluster-separated causal inputs from the
[whole-stream prior method](causal-priors.md). Teacher inference uses the
original twelve-block checkpoint with identical priors/history and per-piece
state resets. Cache all 205 logits and the final 192-dimensional RMS feature.
`compute_last_activation` and `last_activation_to_logits` provide both values
through the original forward path.

The [trainer](depth-distillation.py) learns 75 bounded integer-coordinate
matrices and the 59 retained raw F32 tensors. Every signed BF16 row scale and
activation scale stays fixed. Epochs 1–8 use continuous `z*scale`; epochs 9–32
use the original straight-through rounded/clamped INT4 coordinates. Coordinate
learning rates are 0.03 then 0.01; raw F32 rates are 0.0003 then 0.0001.
AdamW uses beta=(0.9,0.99), epsilon=1e-8 and weight decay=0.

For a target-weighted complete batch, the objective is

```
CE + KL(teacher || student) + 0.1 * mean_d((h_student-h_teacher)^2)
   + 2 * soft_H(z) / 586459020
```

Whole-piece sums are divided by the batch target count; the short final batch
merges into its predecessor. Gradient graphs are freed piece by piece. Hard
static15 exports and full held/long/native-length scores are produced every
four epochs. One single trajectory uses at most 48 CPU hours, including its
teacher cache, exports, native scoring and completed preparation costs. Each
process has one CPU, numerical-library threads=1 and a 7 GiB training AS limit.
Checkpoints retain coordinates, raw tensors, optimizer, RNG and epoch cursor.

## Reused causal inputs across workers

The [input transfer worker](depth-input-transfer.py) consumes a short private
descriptor after the source worker's original `baseline.json` becomes `passed`.
It directly copies existing causal tokens, raw 205-column f16 priors, piece
bounds, baseline results and the original seed414 plan between cloud workers.
Each raw GET checks `Content-Length` against the received file. Original piece
order, per-piece NN resets, continuous PPMd history and the separate
131071-target article-cluster audit remain intact. Capture and PPMd resampling
are **0**. Actual transfer CPU enters the preparation ledger; `ready.json` is
written last, and terminal handling removes the short-lived descriptor.

The existing single 30-minute watcher reads source-worker baseline readiness.
The destination's input-wait phase performs local file checks and can
idle-suspend. It resumes for actual ready inputs or an already active job.
The destination's single pipeline is admitted and the existing watcher follows
the readiness transition. The original full causal capture and native baseline
have passed. Its first transfer completed 1,210,499,798 bytes of inputs, then
received 377,389,056 of 444,020,980 bytes for the validation prior before the
response ended. The inspected `--resume-failed` path keeps the completed inputs
and partial prefix, reuses the same waiting pipeline, and charges the failed
9.406861163 CPU seconds once in the original preparation allowance. The file
API returned HTTP 200 for a one-byte Range request, so recovery advances through
the saved prefix in the whole response. New capture and resampling remain zero.
The original error log is preserved. The recovery has now passed: all eight
inputs total 1,708,393,274 bytes. The same pipeline started its sole admitted
trainer; its first full-causal snapshot has 390,559 targets and all 134 gradient
tensors finite and nonzero. Original preparation plus the single transfer item
is 121.190978347 CPU seconds; transfer terminal CPU includes the old failure.
Full native evaluation and mixed coding follow the original trajectory. Run the
model-free regression from the workspace root with
`python3 -B research/check-depth-input-transfer.py`.

## Native identity and actual measurement

`FX2_MODEL_LAYERS` accepts the two fixed families, 8 and 12; its default is 12.
Both native implementations share the new shape contract. Existing matrix,
KDA, PPMd and range kernels are reused; the selected eight-block attention path
is specified below. Build every new eight-block object in its own directory:

```sh
python3 research/depth-native-build.py --out "$SCRATCH" --depth 8 \
  --flags-from "$ORIGINAL_PRODUCTION_BUILD_JSON"
```

The existing RAW `test_xcheck` compares reference and optimized inference. An
actual 1346-target complete uniform-prior piece has passed student construction,
294-tensor export, one optimizer update, fixed scales, initialization restoration
and native reference/optimized comparison. Its initialized static15 file is
1955187 bytes. This input serves the shape/export/gradient/native contract;
causal held and maximum-length inputs supply quality evidence.

The eight-block production path retains canonical reference-attention
arithmetic, preserving INT4 rounding at half-integer boundaries. Its actual
native link includes `tf_kernels_reference.o`; the twelve-block family retains
its original default. The fixed-scale CPU teacher and student opt into integer-unit FP32 GEMM followed
by one BF16 scale fold, matching native matrix arithmetic. Quantized activation
and weight divisions recover the INT8/INT4 values; the maximum dot bound is
`768*128*7 < 2**24`. Continuous coordinates retain the same real-valued
function and straight-through gradients. The default twelve-block CPU route
keeps its original arithmetic. A same-piece resource comparison reduced the
CPU/native NLL distance from 0.00455607 to 0.00226304 nats/target; all 74 integer
linear recoveries and 134 parameter gradients passed. The four-file production
source update and one actual opt-in adjoint check have now passed with the same
NLL, 74 selected linear modules and 134 finite coordinate/raw gradients. The
[complete numerical and delivery receipt](results/depth-native-scaled-gemm.json)
includes the first check failure, its completed adjoint, source acceptance and
95.809881314 seconds of measured preparation CPU. Complete causal native
scores and real codec bytes retain their selection role.

## One actual mixed deployment

The [single mixed entry](depth-mixed-audit.py) adapts the original
[causal mixed audit](causal-mixed-audit.py), reusing its target-weighted native
reader, frozen epoch ranking, separator/first-token mask and strict paired
G→pre-SSE-logit→u16→actual-byte accounting. Training targets **8 continuous +
24 hard epochs**, bounded by the single 48-CPU-hour ledger. A terminal
`complete` or `cpu_budget` trajectory supplies only its completed every-fourth-
epoch hard/static15 exports and full native evaluations. The receipt records
actual continuous/hard epochs, partial cursor, target coverage and CPU.
Failed/stopped training and zero completed trained exports stop entry.

The twelve-block reference is an explicit `baseline_weight`: original W0,
2904325 bytes and SHA256
`659c0767bce4146d2c457b9aaa13b3e707561ed5d37173a2a6ea1e5965e86b18`.
The student's epoch0 file has the genuine eight-block shape and a different
byte identity; it is initialization metadata outside selection. Choose exactly
one positive-epoch trained hard export, using the original **1082505 held
targets** and the earliest-epoch tie rule:

```
(held_NLL0-held_NLL1)*586215331/(8*ln2)
  + 2*(W0-W1) + 2*(B0-B1)
```

Every candidate's full held and separate 131071-target native audit must match
its actual logs. Held-long3 and the maximum audit retain their report-only
role. The unchanged code delta is constant across these epochs. A negative
trained proxy still receives the method's one mixed measurement. All exports
retain the 294-tensor schema and fixed signed BF16 scales.

### Original v22 objects, real deployed code

Use the original `build/baseline-trace` non-NN objects and its
`probe_research.o`; replace only `tf_*` objects with the accepted native8
objects, including the canonical-attention `tf_kernels_reference.o` from the
actual eight-block production link. The retained Predictor/Mixer/FXCM/PPMd/range paths keep the original
430-output v22 expert, 23+1 mixer, seed923, update limit3000, PPMd order25 and
14000-MiB mapping. `TransformerOpt` keeps its public pointer-based ABI. Its
eight-block implementation carries the new config and four-skip state.

For deployment B1, use the original **untraced** `build/baseline` non-NN
objects and `runner.o`, replacing the same `tf_*` set. Retain the original
production compile flags and `-m64 -Wl,--gc-sections -std=c++17` link with
`-s`; apply **UPX 5.2.1 -9** to that actual runner. The adapter verifies
original source/object hashes, both recorded links, output identities and the
UPX receipt. The first 419 probe/runner sizes remain diagnostic. B0 is the
passed original deployed UPX file, **181956 bytes**. D=100098 bytes and
O=201111 bytes are the passed original shared assets. B1 is read from its
deployed file, with its actual SHA256 and size in the mixed receipt.

The [actual retained-v22 depth8 build](results/depth-v22-mixed-build.json)
has passed its two links, UPX and build contract: **B1=177444 bytes**, giving
**9024 bytes** of double-code savings. The original NN acceptance binaries and
the admitted training pipeline remain intact. Its deployment descriptor waits
for the existing reference sample receipts; the one trained mixed run follows
their readiness and the actual training terminal.

The [actual single mixed process](results/depth-mixed-queue.json) is now queued
with `--wait-for-ready`. It checks local `pipeline.exit` and the deployment
descriptor every 30 seconds, shares the original seven-day mixed wall bound,
and performs **0 network calls** while waiting. The trainer's `wait4` finishes
before its exit marker; failed/stopped training ends the queued measurement.
The existing watcher follows the student through the mixed terminal, keeping
the original single trajectory and its native pause/resume behavior. The
original immediate command remains available.

The alternate original reference worker has its v22 build, probe and strict
reducer. Its four production W0 sample results/encodes wait for their already
admitted mixed phase. Old golden encodes remain present; their original traces
have been released. No new W0 encodes, goldens or trace capture are added to
this method. Existing reference results are copied once after their actual
production readiness and checked by the same deployment contract. Causal
training targets and selected-candidate codec measurements are **0** at queue
publication.

```sh
python3 -B -u "$J/source/research/depth-mixed-audit.py" --job "$J" \
  --deployment "$J/native/v22-depth8-mixed/deployment.json" \
  --range-plan "$J/incoming/source-plan.json" --wait-for-ready
```

### Same scopes, one shared component fee

Reuse existing passed W0 encodes, strict interval reductions and golden
decodes after checking actual encoded bytes, input/vocabulary/dictionary/
weight identities, coordinates and restoration digests. New W0 model runs,
new goldens and prior capture are **0**. Run the selected eight-block W1 once
per side on the two original samples and verify all restored prepared bytes:

- continuous prefix8: encode prepared `[0,8388608)`; mature score
  `[7344758,8386448)`, 1041690 bytes, 1041397 NN targets and 293 fallbacks;
- fresh q2: prepared `[293232482,294281058)`, 1048576 bytes, 1048367 NN
  targets and 209 fallbacks.

The method uses at most **8 CPU hours**, including its new link/UPX preparation
and failed commands, single CPU and 24-GiB AS. Original measurements retain
their prior CPU receipts. Only the two candidate traces are new, 1207959552
bytes. Mature G/M/u16 and range residual stay aligned on the original mask;
the whole-prefix payload remains a separate scope. The actual sample total is

```
prefix8_payload_saved + q2_payload_saved
  + 2*(W0-W1) + 2*(B0-B1)
```

The mature interval lies inside prefix8 and is reported separately. Each named
mature or prefix fullstream-rate scenario adds the shared W/B deltas once.
The complete archive path runs two whole native `cmix -e` programs and two
formal archive decodes, reusing passed identical W0 phases in the same event
ledger. Its final result uses actual `S=P+2B+2D+O+2W+69`, both deployed code
families, restored official 1 GB and the latest record. The depth-specific
fullstream entry is formed from the actual mixed receipt. The
[shared full-stream runner](causal-fullstream.md) now reads that receipt,
uses each side's frozen probe/B and charges its actual package components.
Its `original_input` entry selects the canonical 1 GB source and includes
all compressor preprocessing in the same single NN/range pass. The native
program assembles its own archive; measurement reads that authentic output.
Its [model-free acceptance](results/depth-fullstream-orchestration.json)
covers the depth schema, signed payload changes, double-W/dual-B fees and
the named integer record margin; the original causal contract stays fixed.

Depth admission uses the named **net-S** scenario and latest record margin.
Actual double-weight and double-code savings can pay a measured payload
increase. P, mature G/M/u16 and their changes retain separate reports; the
near-equal-capacity causal method's positive-payload gate stays in its original
family. The single Hutter-event ledger, actual archive sizes and complete
qualification determine the depth method's next two encodes and two archives.


The original causal method's [first complete native epoch](causal-first-epoch.md)
shows a held delta of -0.001188047662 nats/target across 473 source articles.
Distillation's own complete held curve follows its every-four-epoch export;
its actual first-causal training snapshot and preserved-input terminal are in
[the input recovery result](results/depth-input-recovery.json).


## First native hard export: continuous epoch four

The [observed e4 result](results/depth-first-native-e4.json) covers 1,082,505
held targets: native NLL 1.1871585191, +0.2209101803 against the original
twelve-block model. Both short and long held scopes have higher loss. The
eight-block export saves 1,901,728 bytes across the two weights and two binaries.

This is continuous epoch 4/8; hard-QAT coverage is 0/24. The admitted 32-epoch
trajectory and its single mixed-code measurement continue. The next existing
stages measure how much fractional-coordinate fitting survives hard rounding.
The [stdlib replay](check-depth-first-native-e4.py) reconstructs the target
weighting, actual package space and explicitly named linear size scenarios.
Full-stream admission follows real mixed bytes, the final package and latest
record margin.
