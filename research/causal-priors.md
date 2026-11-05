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

The largest natural article occupies the final 31,586,353 bytes. The realized
random sample selected zero training or held pieces from that article. A
**separate max-native-length audit** reserves its middle complete `2^17` piece
at `[570601307, 570732379)`, with 131,071 targets. PPMd learns from stream zero;
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

The max-length audit reuses the existing warm capture, serially after the sparse
pass so only one PPMd disk map is live:

```sh
WORK/bin/activation-probe capture PREPARED/stream.bin AUDIT/tokens.u8 \
  AUDIT/priors.f16 PREPARED/vocab.bin 570601307 131072 0
# AUDIT/bounds.i32 contains [0, 131072].
WORK/bin/activation-probe eval-pieces AUDIT/tokens.u8 AUDIT/priors.f16 \
  ORIGINAL_STATIC15 AUDIT/bounds.i32
```

Use dedicated scratch storage with room for the 14,000 MiB disk map and the
1,654,520,778-byte randomized token/prior files. Capture uses one CPU and an
18 GiB virtual address-space cap; full capture, the reserved audit and original
native baselines share an 8 CPU-hour preparation ceiling. Neural training uses
one CPU and a 7 GiB virtual address-space cap. CPU memory and virtual mapping
are tracked separately.

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
attempt. Formal archive savings and award receipts are **0 at this stage**.

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
