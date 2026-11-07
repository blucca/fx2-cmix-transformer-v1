# Complete depth32: structural repair, residual loss and same-weight coded failure

The fixed eight-block method completed **8 continuous + 24 hard epochs**,
**2,524 optimizer steps** and **94,121,824 training targets**. Its selected
native export is **e12 / step949**. That same weight file grows the complete
8 MiB prefix payload by **163,082 B** and the fresh 1 MiB q2 payload by
**36,218 B**, with both candidate decodes restoring the original input.

The fixed selected-e12 full-stream expansion **stops**. The complete trajectory,
zero-step control, four retained-history codec scopes and physical byte charge
are preserved together.

[Selected e12 weights](assets/depth-e12.tfwc3) ·
[Full trajectory and 473-piece matrix](results/depth-complete-32-trajectory.json) ·
[Terminal identities, four codec scopes and resource receipts](results/depth-complete-32-mixed.json) ·
[Original method/source](depth-distillation.md) ·
[Teacher fidelity and output objective](teacher-targets-and-label-oracle.md)

## Complete native curve

Each epoch revisits the original **1,313 complete training pieces /
2,941,307 targets** from the seed414 whole-stream split. The sampled held
estimator contains 473 pieces / 1,082,505 targets. Its three long pieces
contain 107,647 targets; the independent giant article has 131,071 targets.
The long subset stays inside full held; the giant audit is report-only.

| Export | Step | Weight B | Full held NLL | Held-long NLL | Giant NLL |
|---|---:|---:|---:|---:|---:|
| W0 | — | 2,904,325 | .966248339 | 1.258614301 | .038986426 |
| e0 | 0 | 1,955,187 | 1.663139060 | 2.052032208 | .222665870 |
| e4 | 315 | 1,957,973 | 1.187158519 | 1.518069217 | .186305475 |
| e8 | 632 | 1,960,160 | 1.168803883 | 1.498125758 | .224976626 |
| **selected e12** | **949** | **1,960,982** | **1.145141393** | **1.466907443** | **.268978606** |
| e16 | 1,261 | 1,961,715 | 1.149255266 | 1.472786730 | .275808525 |
| e20 | 1,578 | 1,962,438 | 1.158834673 | 1.485189066 | .282757025 |
| e24 | 1,894 | 1,963,088 | 1.160351253 | 1.489268056 | .321076007 |
| e28 | 2,207 | 1,963,682 | 1.163296659 | 1.492149831 | .332882576 |
| **final e32** | **2,524** | **1,964,319** | **1.167243559** | **1.492512627** | **.375119321** |

All NLL values are nats per target. The original
[zero8 control](depth-zero-native.md) supplies e0's complete same-piece
native measurement. Its separate resource receipt is retained. Scheduled
training-native evaluations occur every four epochs.

e12 repairs **74.3298%** of the initial full-held structural gap, leaving
**+.178893054 nat/target** against W0. e32 then adds **.022102166 nat/target**
against e12; 383/473 pieces and **95.4260%** of held targets get worse.
The giant improves at e4 relative to e0 and later rises to .375119321 at e32.

Online pre-update CE falls in every epoch, from 1.263541627 at e1 to
.948673847 at e32. Teacher KL reaches its minimum at e10 (.141197699),
then ends at .148438320. Final-hidden MSE rises from .166851478 at e12 to
.184845994 at e32. All 32 epoch rows, target-weighted rate/objective terms,
2,566 metric-record counts and 134 finite/nonzero gradient tensors per step
are retained in the full trajectory. Every static15 hard export preserves
the decoded tensors and frozen signed BF16 scales.

## One selected file, one physical charge

The original selection maximizes

```text
(held_NLL0-held_NLL1)*586215331/(8 ln2) + 2*(W0-W1) + 2*(B0-B1)
```

over the eight positive-epoch byte-different exports, with earliest-epoch
tie breaking. Held-long and giant keep their report-only roles. The selected
e12 projected κ1 net saving is **−17,016,191.703 B**.

The physical W0/W1 sizes are **2,904,325 / 1,960,982 B**; packed B0/B1 are
**181,956 / 177,444 B**. Two weight copies save **1,886,686 B**, and two
packed executable copies save **9,024 B**: **1,895,710 B** in total, applied
once across the two complete sample payloads. Research probes have their own
artifact identities.

Selected e12 SHA256:
`8dd9113b676830598d2dad13999f0a90ce738a84fe5ed482fa749aed9db7ce9e`.

| Actual encoded scope | W0 B | e12 B | Saved B |
|---|---:|---:|---:|
| Complete prefix, 8,388,608 input B | 1,072,279 | 1,235,361 | **−163,082** |
| Prefix mature interval, 1,041,690 input B | 177,427 | 205,760 | **−28,333** |
| Complete fresh q2, 1,048,576 input B | 208,769 | 244,987 | **−36,218** |
| q2 outside direct train input, 1,037,391 input B | 206,871 | 242,892 | **−36,021** |
| Old W64 window by seed414 held role, 505,309 input B | 84,844 | 98,545 | **−13,701** |

Both complete candidate decodes restore the original lengths and match the
recorded golden slice SHA256 values. W0 reuses its matching original encode
and golden decode receipts. All production samples use e12; e32 is the final
training-native export.

## Four same-mask transmission scopes

`G` is observed neural f16/guarded probability loss, `M` is pre-SSE mixer-logit
ideal loss, and `u16` is the ideal loss at the actual range-coder probability.
Each row compares the same neural-target mask and selected weight pair.

| Retained-history scope | NN targets | G saved bits | M saved bits | u16 saved bits | Actual saved B | κ_u16 |
|---|---:|---:|---:|---:|---:|---:|
| Prefix mature | 1,041,397 | −308,232.784 | −227,510.657 | −226,659.438 | **−28,333** | .735351 |
| Full q2 | 1,048,367 | −356,204.154 | −290,586.348 | −289,754.465 | **−36,218** | .813451 |
| q2 outside direct train input | 1,037,183 | −354,242.127 | −288,996.762 | −288,175.362 | **−36,021** | .813498 |
| Old W64 window by seed414 held role | 505,207 | −149,790.130 | −110,014.090 | −109,622.448 | **−13,701** | .731840 |

The q2 direct training input is article76410,
`[293818027,293829212)`, containing 11,184 next-token targets. q2 contains
zero realized sampled-validation targets. The excluded-input row retains
parent trace intervals `[0,585545)` and `[596730,1048576)`; the old-window row
uses `[531039,585545)` and `[596730,1047533)`. These reporting scopes overlap.
Original PPMd, mixer and range histories continue across excluded positions.

The reader sums `closed(end)-closed(begin)`, with `closed=emitted+1`, from
the original parent byte0 history. Both complete q2 files have one final
Flush byte. The excluded-input complement plus Flush is **1,898 B / 2,095 B**,
giving `−36,021 + (1,898−2,095) = −36,218 B`. Fallback masks, override bits,
range endpoints and residuals accompany all four rows in the result.

## Resource and full-package scope

The original trainer ledger is **91,367.257759 CPU s**, including a
121.716697 s preparation carry; peak RSS is **4,903,756 KiB**. Its complete
32-epoch trajectory uses **25.379794 CPU h** of the original 48 h cap.
The selected-e12 mixed queue uses **5,182.042207 CPU s**; four later read-only
q2 intervals add **.512891 CPU s**. Original source2 baseline and current
source3 candidate resources remain separate recorded buckets.

Candidate prefix encode/decode program CPU is **2,225.899930 / 2,283.678043 s**,
with RSS **5,981,120 / 6,017,124 KiB**. q2 uses **320.859226 / 342.398373 s**,
with RSS **5,828,860 / 5,828,608 KiB**. The separate zero8 control uses
96.722153 CPU s. Program, wait4-wrapper and included-native ledgers retain
their individual scopes.

The two complete payloads save **−199,300 B**. Adding the one physical asset
delta yields **1,696,410 B saved in that sampled component**. The original
mature-rate scenario projects about **15.951 MB extra full-stream payload**
against **1.896 MB fixed-asset space**, with net saving **−14,055,430.372 B**.
The full-prefix rate scenario gives **−9,505,574.921 B** net saving.
Formal S0/S1 and latest-record L are null in this result. Complete package
accounting stays `P+2B+2D+O+2W+69`.

## Reuse and next complete control

The [original trainer](depth-distillation.py),
[pipeline](depth-training-job.py), [sole mixed runner](depth-mixed-audit.py)
and [interval reader](mixed-interval.cpp) are already published. The two
JSON files include all 32 epoch rows, all nine exports and the complete
473-piece native loss matrix. `ORIGINAL/` labels identify retained source
artifacts; scientific values are embedded in the published results.

Receipt-only standard-library readout, from the repository root:

```sh
python3 - <<'PY'
import json
from pathlib import Path
r = Path('research/results')
t = json.loads((r/'depth-complete-32-trajectory.json').read_text())
s = json.loads((r/'depth-complete-32-mixed.json').read_text())
assert t['coverage']['seen_targets'] == 94121824
assert len(t['online_epochs']) == 32 and len(t['exports']) == 9
assert sum(p[0] for p in t['paired_pieces']) == 1082505
assert s['sample_charge']['one_time_double_W_plus_double_B_saved_bytes'] == 1895710
print({k: v['paired']['interval_coded_saved_bytes']
       for k, v in s['four_scope_attribution']['scopes'].items()})
PY
```

A complete teacher-only eight-block control is next: retain the original
architecture/cache, start from zero8 with a fresh optimizer, keep the
32-epoch hard-export rhythm and measure its own selected-W codec result.
The [curvature-matched 2KL comparison](teacher-objective-audit.md) retains
the original hidden/rate terms while replacing observed-label CE + KL.
The [teacher-target study](teacher-targets-and-label-oracle.md) records the
target-fidelity and label-objective measurements for that comparison.
Execution of this next control is pending.
