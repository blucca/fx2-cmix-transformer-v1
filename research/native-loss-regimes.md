# Where native gains and deployment losses land

The causal-prior and eight-block methods share an exact held sample. This
comparison asks which parts carry the early causal gain, its later regression,
and the smaller model's compensation. It reuses the completed native outputs.

**Main result:** the high-surprisal domain carries most absolute loss, while
the low-surprisal domain has the largest relative distortion. The causal
e2-over-e1 improvement is a tradeoff: its high-domain gain pays for regressions
in the other two domains. The original complete trajectories and single mixed
measurement per method retain their selection role.

## One frozen partition, the same native targets

Partition the **473 complete held pieces / 1,082,505 targets**, one piece per
article, by original W0 per-piece native logit NLL: **[0,.25), [.25,1), [1,∞)**
nats/target. The thresholds were chosen before reading trained contrasts.
The grouping variable is baseline-model native surprisal; PPMd confidence
and encoder weights have their own measurement definitions.

The causal comparison uses W0, e1, e2 and e6; the depth comparison uses
hard-rounded exports after continuous epochs e4 and e8. Each row retains the
same input order, piece boundaries and target counts. The independent
131,071-target maximum-article audit has a separate report-only scope.

[Per-piece native loss accumulators](results/native-loss-regimes-pieces.json)
and [compact result](results/native-loss-regimes.json) reproduce the partition
and comparisons with the Python standard library:

    python3 -B research/check-native-loss-regimes.py

## Absolute loss and relative distortion

Delta is trained minus reference NLL within each domain, in nats/target.

| W0 native NLL | Pieces / targets | W0 surprisal share | Causal e2−W0 | Causal e6−e2 | Depth e8−e4 | Depth e8−W0 |
|---|---:|---:|---:|---:|---:|---:|
| <.25 | 140 / 115,152 | 1.61% | −.00001107 | +.00941005 | −.00971301 | +.06394449 |
| .25–1 | 203 / 296,310 | 19.31% | −.00090938 | +.01714928 | −.01635395 | +.17431453 |
| ≥1 | 130 / 671,043 | 79.08% | −.00166662 | +.02417128 | −.02072099 | +.23881169 |

The ≥1 domain contains **61.99% of targets** and **79.08% of baseline native
surprisal**. It contributes **80.51% of causal e2's gain**, 72.46% of its later
e2→e6 regression, 69.98% of depth e4→e8 recovery and 73.09% of depth e8's
remaining loss. Its gain share closely follows its baseline information load.

Depth e8 raises native loss relative to W0 by **43.70% / 25.57% / 19.38%**
in the low / middle / high domains. Continuous e4→e8 fitting recovers
**13.19% / 8.58% / 7.98%** of the respective e4 gaps. Low-domain recovery is proportionally larger and leaves higher proportional
distortion; most absolute residual remains in the high domain.

**356 pieces / 1,012,417 targets (93.53%)** show both causal e2→e6 regression
and depth e4→e8 recovery. The methods move in opposite directions on largely
the same held targets. Their frozen stages and parameter changes differ;
tensor-level attribution retains the scope of the original diagnostics.

## The small e2-over-e1 advantage is domain-sensitive

The full held sample improves by **103.039901 nats**, equivalent to 18.581894
raw native bytes. Its three contributions are:

| Domain | e2−e1 summed native loss, nats |
|---|---:|
| <.25 | +36.483468 |
| .25–1 | +24.656922 |
| ≥1 | −164.180291 |

The real weight files are 2,904,329 B at e1 and 2,904,371 B at e2, adding
**84 B across two copies**. Under the named scenario of the same held-domain
composition covering 586,215,331 full NN targets and constant native-to-code
transmission factors inside each domain, the net e2-over-e1 advantage is:

    97.65889660 × (164.180291 κ_high − 36.483468 κ_low − 24.656922 κ_mid) − 84 B

All three factors at 1 give **+9,978.76 B**. With κ_low=κ_mid=1, the high
factor reaches break-even at **0.37763684**. This is a sensitivity of the
native-score ranking. The original selector still chooses one completed
trained export; the actual paired codec measures that same chosen weight.

## What the coded experiment decides

The deployed v22 codec feeds guarded/f16 neural probabilities into the outer
23+1 mixers and FXCM's internal inputs and contexts, then applies SSE and
u16/range coding. The coupled online trajectory determines transmission.
The original measurement reports matched **G→M→u16→actual bytes** on the
mature prefix interval and fresh q2; whole-prefix bytes remain a separate
scope. The complete archive charges the shared two-copy W/B cost once.

The native domains locate the empirical tradeoff. The original hard-stage
depth checkpoints and terminal causal curve complete the learning evidence;
same-weight coded results and the full-S/resource gate decide deployment.
Current coded gain and prize results remain recorded in their original
measurement receipts.

## Observed-article precision of the paired native ratios

The same 473 observed articles support an influence-function standard error
for each target-weighted paired ratio. With article target count `n_i`,
summed loss difference `d_i`, `N=sum(n_i)` and `m=473`:

    theta = sum(d_i) / N
    se = sqrt(m/(m-1) * sum((d_i - theta*n_i)^2)) / N

| Paired contrast | Native delta, nats/target | Observed-article standard error | Normal interval, theta ± 1.96 se |
|---|---:|---:|---:|
| Causal e2−W0 | −.001283234 | .000126080 | [−.001530351, −.001036117] |
| Causal e2−e1 | −.000095187 | .000128656 | [−.000347353, +.000156979] |
| Causal e6−e2 | +.020678944 | .000550419 | [+.019600124, +.021757765] |
| Depth e8−e4 | −.018354636 | .000869630 | [−.020059111, −.016650160] |
| Depth e8−W0 | +.202555545 | .004559358 | [+.193619203, +.211491886] |

The e2-over-e1 margin is **0.740 standard errors**, alongside **10.178**
for e2-over-W0 and **37.569** for e6's late regression. Removing one observed
article at a time keeps the e2-over-e1 delta between **−.000123696 and
−.000018671 nats/target**. Sampling precision and the leverage of one article
answer distinct questions.

These intervals describe resampling of the **observed article support**.
The original design assigns an article-held role with probability .25, then
independently samples native pieces with probability .0075. All 473 observed
articles contribute one piece. Native pieces above 64 KiB carry **6.621%**
of full-stream targets and **zero** observed held targets; the separate
maximum-article audit retains its report-only scope. e2 is selected from the
completed trajectory, so independent deployment confirmation comes from the
original same-W coded/full-archive measurement.

[The compact precision result](results/native-sampling-precision.json) and
the existing [stdlib replay](check-native-loss-regimes.py) retain all five
contrasts, intervals and leave-one-article ranges. The original fixed
selector, trajectories and sole mixed runs continue unchanged; remaining
uncertainty is concentrated on real native-to-codec transmission.

## Complete hard stage: the early low-domain recovery ends

The original **32-epoch CE+KL** trajectory now supplies all nine fixed native
exports to the same predeclared W0 partition. From e12 to e32, native NLL
increases in all three domains:

| W0 native NLL | Depth e12 | Depth e16 | Depth e32 | e32−e12 | Share of summed late loss |
|---|---:|---:|---:|---:|---:|
| <.25 | .202819 | .201901 | .207156 | +.004337 | 2.09% |
| .25–1 | .837478 | .840622 | .858205 | +.020728 | 25.67% |
| ≥1 | 1.442699 | 1.448105 | 1.468457 | +.025758 | 72.24% |

The low domain reaches its best positive-epoch native score at **e16**; the
middle and high domains reach theirs at **e12**. By e32 the full held sample
adds **23,925.705667 nats / +.022102166426 nats per target** versus e12.
**383/473 pieces / 1,032,991 targets (95.43%)** worsen; the high domain has
127 worsening pieces and three improving pieces. The original e12 selection
and sole same-weight coded measurement retain their frozen roles.

The independent **131,071-target maximum-article audit** moves from .268979
at e12 to .375119 at e32, a **+.106140715121** increase. It remains a separate
long-history stress case alongside the complete held partition.

[Complete native-regime result](results/depth-complete-native-regimes.json)
retains all nine scores, observed target counts, per-domain loss sums and
the separate audit. The existing standard-library replay above recomputes
them from the published complete trajectory. These are the frozen CE+KL
controls for the independent [teacher-only full-trajectory experiment](teacher-only-distillation.md).
