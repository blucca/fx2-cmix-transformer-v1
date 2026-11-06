# Whole-stream causal priors, complete-piece learning

The [same-text control](integer-coordinate-attribution.md) measured an
integer-coordinate epoch-4 regression of **+0.003996 nats/target** with mature
PPMd priors and **−0.003014** with fresh priors. The **+0.007010** interaction
locates a concrete distribution problem: the original cold training inputs
rewarded a different use of the statistical prior.

This method replaces those inputs with one causal PPMd traversal of the exact
prepared stream. The original network, signed BF16 scales, integer STE,
CE + teacher KL + full-stream soft symbol entropy, and static15 format stay
fixed. The new trajectory has **16 complete epochs / 48 CPU hours**.

## Exact input contract

- Prepared stream: 586,459,020 bytes, 205-byte vocabulary, 243,427 natural
  articles, 243,689 native pieces, 586,215,331 scored next-token targets.
- Native article separator **ends** and the original `2^17` piece cap determine
  every reset. Each retained piece contains all its targets.
- Training admits complete pieces up to 32,768 bytes. This covers
  **529,030,842 / 586,215,331 targets (90.24%)** of the stream. The boundary
  comes from the measured 35,162-target full-backpropagation memory cost.
- Seed 414 assigns natural articles to a 25% held cluster, followed by
  independent equal-probability piece sampling at `p=0.0075`. Target-weighted
  means preserve the byte objective across the highly uneven article lengths.
- The realized random sample has **1,313 training pieces / 2,941,307 targets**
  and **473 held pieces / 1,082,505 targets**. Its held lengths include
  32,859, 36,254 and 38,537 bytes. Its maximum native piece is 38,537 bytes.
- Explicit little-endian int32 piece boundaries prevent concatenated,
  nonadjacent pieces from sharing neural state.
- PPMd observes every earlier stream byte, including earlier held text, exactly
  as it does while encoding. New neural gradients use the sampled training
  targets; the original network supplies teacher logits on these new inputs.

The largest article spans `[554867326, 586453679)`, with 31,586,353 bytes.
A separate 5,341-byte final span follows it. The realized
random sample selected zero training or held pieces from that article. A
**separate max-native-length audit** reserves its middle complete `2^17` piece
at `[570595966, 570727038)`, with 131,071 targets. PPMd learns from stream zero;
its entire natural article has zero sampled training pieces. This fixed audit
and the randomized held estimator are reported separately each epoch.

The realized held sample has a different length composition from the full stream:
16,384–32,768-byte pieces contribute 7.795% of held targets versus 13.279% of
full targets; 32,769–65,536-byte pieces contribute 9.944% versus 3.134%; pieces
above 65,536 bytes contribute 0% versus 6.621%. The randomized mean reports
this realized sample. Selection uses that mean; full-stream code length comes
from actual paired encoding. The separate maximum-length audit stays distinct.

## Sparse causal capture

`causal-sample-plan.py` writes the original source intervals and local reset
boundaries. `causal-ppmd.cpp` advances the original 25-order, 14,000 MiB PPMd
model over every real byte and writes only retained rows. Its row exit reuses
the original `HalfFileWriter`, vocabulary ordering and float16 conversion.
The production predictor's member layout is preserved.

```sh
python3 research/causal-sample-plan.py PREPARED/article_boundaries.i32 WORK/data
# Link the research main with the original model objects and updated predictor.
WORK/bin/causal-ppmd PREPARED/stream.bin PREPARED/vocab.bin WORK/data/plan.tsv WORK/capture
WORK/bin/activation-probe eval-pieces WORK/capture/validation.u8 \
  WORK/capture/validation.f16 ORIGINAL_STATIC15 WORK/data/validation.i32
```

The optional `audit` role writes the independent max-length piece during that
same pass. It uses the same post-byte `WritePpmdRow` as the existing warm-start-0
capture, with its own token/prior files. The randomized plan and boundaries
stay fixed; only a sorted, disjoint audit interval is added to the capture plan:

```sh
python3 - WORK/data/plan.tsv WORK/data/joint-plan.tsv <<'PYTHON'
from pathlib import Path
import sys
rows = [(int(a), int(b), role) for a, b, role in
        (line.split() for line in Path(sys.argv[1]).read_text().splitlines())]
rows = sorted([*rows, (570595966, 570727038, 'audit')])
assert all(a[1] <= b[0] for a, b in zip(rows, rows[1:]))
Path(sys.argv[2]).write_text(''.join(f'{a} {b} {r}\n' for a, b, r in rows))
PYTHON
WORK/bin/causal-ppmd PREPARED/stream.bin PREPARED/vocab.bin \
  WORK/data/joint-plan.tsv WORK/capture
# AUDIT/bounds.i32 contains [0, 131072]. Keep these rows out of training.
WORK/bin/activation-probe eval-pieces WORK/capture/audit.u8 \
  WORK/capture/audit.f16 ORIGINAL_STATIC15 AUDIT/bounds.i32
```

The fixed maximum article has zero selected randomized rows, so the added
interval is disjoint from all 1,786 original intervals. The new plan has 1,787
intervals. Audit row `k` comes after source byte `570595966+k` is completed;
its 205 f16 values predict the next byte. All 131,072 rows are saved and native
scoring uses 131,071 next-token targets.

Use dedicated scratch storage with room for the 14,000 MiB disk map, the
1,654,520,778-byte randomized token/prior files and 53,870,592 additional audit
bytes. Capture uses one CPU and an 18 GiB virtual address-space cap. The
recovery's cumulative preparation ceiling is **9.5 CPU hours**: the initial
4-hour failed pass, one replacement scan capped at 5 hours, and 0.5 hours
for source preparation and both native baselines. Shared audit rows are charged
once with the full scan. Neural training retains one CPU, its 7 GiB virtual
address-space cap and the original 48 CPU-hour allowance. CPU/RSS and virtual
mapping are tracked separately.

### Actual capture recovery

The initial pass ended with SIGKILL (`-9`) at its configured 14,400-second
hard CPU ceiling. The last progress row records **577,765,376 source bytes /
14,374.780 CPU seconds**, with peak RSS **4,670,472 KiB**. Kernel OOM matches
were zero. All randomized intervals ended by source byte 554,866,347;
logical sample counters were complete, while the half writer and stdio buffers
left the physical output prefixes short. The replacement reconstructs the
original state from byte zero and writes fresh train/held/audit outputs.

The original mean throughput projects a complete replacement at about
**14,591 CPU seconds**. Joining the independent audit avoids the former second
warm-start-0 traversal to byte 570,727,038. Same seed, PPMd update order,
vocabulary and original half conversion preserve the exact row contract.
[Recovery receipt](results/causal-capture-recovery.json) reports the added-role
check, complete input counts, prior failure charge and actual current stage.
The teacher, 16 epochs, independent native scores and sole mixed candidate
retain their original recipe.

The added role check passes across train **3,840**, held **32,768** and audit
**8,192** bytes, including all 205 f16 columns and token bytes. The two ready
original binaries were the transferred experiment assets. Recovery rebuilt the
original adapted predictor/mixer objects and added caller; its missing short
reference was generated once with the unchanged original capture binary. The
completed three-role fixture was reused. All preparation attempts cost
**4.114 CPU seconds**, including their dependency failures. The recovered full
scan is running; new teacher/optimizer and real-payload saving are still zero.

The actual [capture job](causal-capture-job.py),
[training job](causal-training-job.py) and
[recovery builder](causal-capture-recovery-build.py) preserve this measured
original experiment layout. Copy them to WORK as `capture.py`,
`training-job.py` and `recovery-build.py`, respectively. They use the original
build manifest, adapted source headers, ready original evaluator, loader object,
full prepared input and fixed teacher/export assets. Their explicit limits and
prior-failure charge describe this recovery batch.

## Training and measurements

Use the original [CPU full-piece loader and adjoints](cpu-backward.md), kernel,
checkpoint, exporter and recoder. In `integer-coordinate-qat.py`, select
`--recipe causal-v1`. Training pieces point to the sampled training file;
score inputs point to the independent held file and `boundaries`. A second
sample has empty `train_pieces` and scores only the max-length audit.

For `causal-v1`, the short final optimizer batch merges into its predecessor.
Each complete piece still performs and frees its own backward graph. In the
fixed 16-epoch ordering this changes 1,283 updates to 1,267, while preserving
2,941,307 targets per epoch. The epoch-11 tail had 242 targets; its merged
batch has 37,015. Maximum merged batch size is 87,824 targets and maximum
individual backward piece remains 29,586 targets. The original fixed-scale
trajectory keeps its original batching. [Replay receipt](results/causal-prior-batches.json).

The first small complete-piece update checks original teacher/student logits,
frozen scales, integer bounds and zero-step static15 bytes. The same saved
trajectory then runs to its epoch/CPU boundary. Every epoch stores the
randomized held mean, its three longer-piece mean, the independent `2^17`
audit, and actual double-weight bytes. The reserved audit stays outside the
randomized population estimate.

After the trajectory exits, rank completed **byte-different trained exports**
by `(baseline held NLL - candidate held NLL) * 586215331 / (8 ln 2)` plus the
actual double-weight byte saving, with earliest epoch winning ties. Measure
one such export even when this proxy is negative. The original zero export
remains the keep/delivery baseline; reuse its passed round trips when every
trained export is byte-identical. The independent max-length audit and three
longer held pieces remain report-only.

The original v22 codec routes NN bit probabilities into both outer mixing and
FXCM's internal inputs and argmax context. Weight changes therefore alter a
coupled online trajectory. This is why the single trained candidate gets a
real coded measurement. [Measurement runner](causal-mixed-audit.py) and
[original-object builder](causal-mixed-build.py) implement the bounded stage.
Copy them to WORK as `mixed-audit.py` and `build-mixed.py`, with the original
static15 loader object and unchanged `mixed-interval.cpp` under WORK/assets.
Use the original 8 MiB continuous prefix and 1 MiB cold middle sample, matched
neural/fallback masks, actual encode/decode and one shared double-weight delta.
The runner reports guarded-neural → mixer → u16 → code-length credit and
κ_G for the same score interval; raw NLL stays paired with its own native inputs.

The existing same-codec mixed round-trip protocol measures encoded bytes,
lossless restoration and complete package cost before a full-stream record
attempt. The [full-stream stage](causal-fullstream.md) runs the actual
`cmix -e` on the canonical 1 GB original with the frozen untraced codec.
[archive-package.py](archive-package.py) first builds its compressor from
the original B/D/O/W and then checks the native program's authentic archive,
37-byte stream header and two 16-byte self-extraction headers. It measures
`S = P + 2B + 2D + O + 2W + 69` from the two resulting files.

Run each actual `./archive9` once for its full 1,000,000,000-byte restoration;
the original enwik9 MD5 is `e206c3450ac99950df65bf70ef61a12d`. This is the
full-stream candidate's decode stage, avoiding an additional prepared-stream
probe decode. Record whole compressor and 1GB archive CPU/RSS/wall by
their actual ranges, including D/O subprocess CPU. Formal active time and
the machine's score determine qualification. The package fixture checks
the native section layout and complete two-copy charge:

```sh
python3 research/archive-package-check.py --work /YOUR/temp/archive-check
```

Formal archive savings and award receipts are **0 at this stage**.

## Prepared-to-original provenance

The shared `prepare.cpp` adapter also accepts the complete inverse:

```sh
WORK/bin/prepare --restore PREPARED/stream.bin DICTIONARY/english.dic FRESH_OUT
# Verify FRESH_OUT/enwik9_uncompressed: 1,000,000,000 bytes,
# MD5 e206c3450ac99950df65bf70ef61a12d.
```

It starts directly at the original WRT/block decoder, then restores PHDA9,
article order and intro/main/coda. Neural, PPMd and range calls are zero.
The original forward adapter and transform helpers stay intact.
[Compiled-input checks](results/prepared-inverse-build.json) cover the shared
adapter and both complete-input guards. The full original restoration is
queued after the causal pipeline and its single mixed measurement finish.

[The serial inverse job](causal-prepared-inverse.py), copied to
`ROOT/round415/prepared-inverse-job.py` beside the deployed experiment, shares
0.5 CPU hours between a source build and the complete inverse. It uses one
CPU, a 4 GiB inverse address-space ceiling and an 8 GiB scratch reservation.
It records the actual original MD5, CPU/RSS and generated-file disk peak,
then retains the original for final compressor timing and removes duplicate
intermediates. This is the complete prepared input's provenance check; the
formal archive stage measures the actual coded W0/W1 restoration.

## Measured implementation checks

[Preparation receipt](results/causal-prior-prepare.json):

- 64 KiB original full capture reused; every retained row in all **205 f16
  columns** matches byte-for-byte.
- Default native resets and explicit resets produce identical probabilities
  across **65,439 targets / 97 pieces**.
- Build, fixtures and full-stream plan took **14.349 CPU seconds**.
- One synthetic 8-target check verifies target-weighted long-piece means.

Original fixed-input coordinates and its terminal native/mixed queue remain
separate. This method changes the causal training distribution and keeps its
own weights, teacher cache, held measurements and trajectory.


### Live capacity adjustment, same causal pass

A single near-cap observation found 514850816 / 586459020 bytes processed
(87.7897%) and 12920.58 CPU seconds used. Recent 15–4 MiB rates projected
15031.59–15477.56 total CPU seconds; the observed final 1 MiB slow-tail
condition projected 16762.42, crossing the original 16200-second cap.
The same capture process received an applied 18000-second limit, adding
1800 seconds to the ceiling. The complete preparation ceiling is 34200
seconds, including the preserved original 14400-second failed pass.

A [single finite handoff](results/capture-cap-handoff-round423.py) pauses the
original waiting pipeline group while capture continues. The original capture
wrapper holds its launch budget in memory; at its existing terminal, the
handoff preserves that snapshot, records the applied limit and one overhead
cost item, then continues the same pipeline group. The original data, sampling,
18-GiB capture AS, teacher/16-epoch recipe, 48-CPU-hour training and one mixed
measurement stay fixed. [Observation and applied-limit receipt](results/causal-capture-capacity.json)
records the launch-era waiting terminal handoff. These rates cover the causal capture
program; final compressor and archive time are measured on their own programs.

### Complete causal pass and first real native baseline

[The terminal receipt](results/causal-capture-terminal.json), observed on
2026-11-06 at 08:00:58 UTC, records the completed **586,459,020-byte** pass.
Train/held/audit have 2,942,620 / 1,082,978 / 131,072 rows, all **205 f16
columns**, and 2,941,307 / 1,082,505 / 131,071 NN targets. The fixed article
clusters, complete-piece boundaries and independent audit coordinates stay
unchanged. Capture reports **14,713.965606 CPU seconds** and 4,687,380 KiB
peak RSS. Its full child rusage is 14,715.443169 seconds and wrapper CPU
27.890923642 seconds; these are distinct resource scopes.

The finite handoff completed, preserved the launch 16200/32400 budget
snapshot and resumed the original pipeline group. Preparation used
**29,267.181989056004 / 34,200 CPU seconds**, including the conservative
14,400-second previous failed pass and one 0.038544376-second capacity
handoff cost. The original 16-epoch/48-CPU-hour trajectory is running.

Original W0 is evaluated on the true causal rows with native piece resets:

| Scope | Targets | Model logit NLL | Model probability NLL |
|---|---:|---:|---:|
| Randomized complete-piece held | 1,082,505 | 0.966248338794 | 0.966248314271 |
| Three >32-KiB held pieces, within that total | 107,647 | 1.258614301310 | 1.258614281198 |
| Independent max-article middle audit | 131,071 | 0.038986426132 | 0.038986374735 |

`logits_mean` uses double logsumexp of the model's logits; `probs_mean`
uses its fp32 output probability. Both columns score the NN model. The
three long held pieces are included once in the full held mean; their
standalone view and the independent audit retain their report roles.

[Baseline analysis](results/causal-baseline-analysis.json) derives the
remaining 974,858-target short mean, **0.933964335620**, and the long
target share, **9.94425%**. Using the historical 952,225-byte record gap
as a scenario and the actual initialized depth W/B saving of 1,907,300
bytes, the smaller model can pay a 955,075-byte payload increase. At the
selection proxy's kappa=1, that is about +0.009034 nats/native-target.
For example, a +0.1 long-piece delta then requires about -0.001010 in
the short portion. Actual mixed code, trained W1 bytes and the record at
full-stream admission supply the final comparison.

The observed training snapshot is epoch0/step19, with 692,078 targets seen
and zero completed native evaluations. Its reported 1,294.790013969 CPU
seconds includes the 104.944287-second baseline carry already paid in
preparation. The unique event training cost is 1,189.845726969 seconds;
event accounting deducts that carry once while retaining the original
48-hour runtime cap. This early scope includes startup, preflight and
lazy teacher-cache work. The same early average projects 22.47 hours
for 16 epochs, with remaining native/export costs reported separately.

The completed source inputs are ready for the existing student transfer.
Student input copying, its 32-epoch trajectory, the sole mixed measurements
and both actual full-stream package sizes follow their existing queues.
