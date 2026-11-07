# A physical half-buffer path oracle

## One bounded mechanism

Use the same **12 saved real-logit rows / 12 actual targets / 96 path bits** as
`byte-path-half-guard-control.json`: four saved rows from each of its three held
pieces, with the initial mapped eight-layer student and zero updates. Rebuild
the control's FP64 softmax → FP32 → RNE half words and source decode/guard.
The original teacher lookup indices and KL values replay the prior control.

For the actual target, partition the 205 vocabulary entries into its leaf and
up to eight sibling subtrees. Split each group into **F16C indices 0..199** and
**scalar-tail indices 200..204**. The scalar decoder halves subnormal values;
all word permutations stay within their original decoder domain. The target
leaf remains fixed.

Within each nonempty group × domain cell, sort the teacher's effective
decoded-and-guarded values and place their **original uint16 half words** at
destinations sorted by the student's effective values. Ties use source and
destination indices. The fixed multiset holds `Σ R log R` constant, while the
rearrangement inequality maximizes `Σ R log S`. This yields the global
`KL(R||S)` minimum within this domain-preserving permutation family.

The certificate's feasible family consists of those fixed half-word cell
multisets. Broader half-valued group-mass families retain their own optimization
problem. Decoder-domain preservation gives exact effective group mass equality;
integer checks use units of `2^-43`, including the source FP32 floor. Replaying
the permuted pre-half teacher FP32 values also reproduces every output half word.

## Observed result

KL is in nats, using normalized effective rows. Means weight each of the 12
actual saved targets once.

| Quantity | 12-row mean |
| --- | ---: |
| Original `KL(T||S)` | 1.339565416706 |
| Permuted-target `KL(R||S)` | 1.184486979731 |
| Teacher→oracle full `KL(T||R)` | 0.181126596690 |
| Common group/path KL | 1.066339893701 |

With the student held fixed, the alternative target's `KL(R||S)` distance is
**0.155078436976 / 11.5768%** smaller. This relaxes **56.7584%** of these saved
rows' within-sibling target distance. The common group KL stays fixed;
teacher→oracle KL describes the changed output target.
The oracle changes **1,536 / 2,460 actual half words** and **968 effective values**.

Exact group mass error is **0**, as is group × decoder-domain mass error.
Source-order serial FP32 observations follow the real raw-byte subtree order:
sum right, seed the left sum with right, divide, inspect normal/0/1 override,
clamp, and multiply/truncate for the 100,001-entry lookup. Results:

- Teacher→oracle lookup changes: **0/96**; override changes: **0/96**.
- The prior guarded probability-space projection followed by fresh half
  quantization changed **31/96** lookup indices.
- Maximum serial-FP32 conditional change: **2.384185791e-7**; table indices stay
  equal. Teacher/oracle each have 93 normal and three zero-override bits.

## Real native consumer: one failed lookup pair

The [compiled source-linked observer](native-byte-path/findings.md) checks these
same buffers under Clang17's actual fast/O3 flags. It records **12/96 changed q
bit patterns, 1/96 changed lookup/Stretch outputs and 0/96 changed overrides**.
At hard/piece14/row0, target byte220, first-bit q changes from
`.5272147059440613` to `.5272147655487060`, moving the real lookup from 52721 to
52722. All 4,920 decoded/guarded FP32 values match the saved-row contract.
The exact mass and half-representation certificates hold; the compiled
fast/O3 same-observation acceptance **fails** at this pair.

## Artifact and continuation scope

[`teacher.f16`](results/byte-path-permutation-teacher.f16) and
[`oracle.f16`](results/byte-path-permutation-oracle.f16) contain 12 consecutive 205-word little-endian
rows, **4,920 bytes each**, in [the saved result](results/byte-path-half-permutation.json) case order. The JSON records source paths, buffer sizes, every lookup/override,
and the exact source-index permutation
as 205 u8 entries per row. Only these saved rows enter this experiment.

This is an **output-buffer oracle** selected by the actual target during offline
analysis. Learned architectural realizability and complete native fast/O3 and
payload evaluation remain attached to the decision after the original complete
2KL trajectory. Model forwards, optimizer updates, remote calls, native/codec
runs and new learning engineering are all **0**.

Replay from the repository root (standard library only):

```sh
PYTHONDONTWRITEBYTECODE=1 python3 research/byte-path-half-permutation.py \
  research/results/byte-path-initial8-geometry.json \
  research/results/byte-path-half-guard-control.json \
  --output-dir temp/half-path-oracle
```
