# Eight-block distillation: a fixed structural candidate

This records the original static contract and byte-budget study. The
[complete eight-block distillation method](depth-distillation.md) is now being
implemented; its fixed native family has compiled successfully. Actual student
construction, training, native encoding and weight-file measurements have zero
calls at that publication. The original [causal trajectory](causal-priors.md)
stays fixed.

## One coherent initialization

The original model has twelve 192-wide blocks, MLP width 768 and three
64-wide heads, with `[KDA, KDA, KDA, attention]` repeated three times.
Keep original blocks **0, 1, 2, 3, 8, 9, 10, 11**, renumbered continuously.
This retains two complete groups, six KDA and two attention blocks, and
reduces the INT4 matrix count from 111 to 75.

Original skip sources 0..5 connect in reverse to destinations 6..11.
The retained outer pairs are 0→11, 1→10, 2→9 and 3→8. The eight-block
student initializes its four destination weights from original indices
**2, 3, 4, 5**. Distillation must absorb the removed four-block transform.

## Existing-histogram byte estimate

Sum the original [tensor census](results/weights-census.json)'s per-tensor
static4096 coding cost, including its 28-byte frequency description.
Names, BF16/F32 payloads, global range rounding, newly trained histograms and
new executable size enter the later actual file measurement.

| Removed original blocks | One-copy INT4 saving estimate | Two-copy estimate | Historical payload allowance |
| --- | ---: | ---: | ---: |
| 4, 5 | 465,342 B | 930,685 B | −21,540 B |
| 4, 5, 6 | 697,802 B | 1,395,605 B | 443,380 B |
| **4, 5, 6, 7** | **907,849 B** | **1,815,698 B** | **863,473 B** |

The allowance subtracts the historical 952,225-byte prize-gap scenario.
For the fixed 586,459,020-byte prepared stream, the eight-block allowance is
**0.0117788 bits per prepared byte**, or **0.00816444 nats**. The native mixed
codec supplies the actual relationship to raw-network loss.

The [real original native auxiliary package](original-release-native-aux.md)
fixes the current B/D/O/W envelope at 6,573,938 bytes before payload P.
Actual score uses `S=P+2B+2D+O+2W+69`; current full payload and latest official
record replace the historical scenario at full-stream entry.
[Detailed estimate](results/depth-distillation-static-budget.json).

## Required implementation contract

- `cpu_model.py` currently checks twelve layers/nine KDA blocks. The CPU
  exporter expects 429 payloads and 111 quantized matrices.
- `weights-recode.cpp` accepts a v2 template and its replacement mode retains
  all template names/shapes plus five derived payloads. An eight-block model
  needs a genuine eight-block v2 template. Reuse the existing v2 writer and
  static15 coding; set `config.ints`, `config.kimi` and the four-element skip
  tensor to the student shape. The three RoPE tables keep their original
  head width and sequence-length identities.
- `model.cpp`, `opt/arena_build.h`, `opt/arena_build.cpp` and
  `opt/model_opt.cpp` contain the original layer, 9/3 attention, six-skip and
  `11-l` contracts. A fixed eight-block native family updates all of them.
  The 192/768 matrix kernels, KDA, attention, PPMd and range interface stay.

## Complete candidate method

Reuse the whole-stream causal priors and article-cluster split, the existing
full teacher-logit cache and complete pieces. Add only final RMS hidden-state
teacher targets (about 2.10 GiB f32 on the realized training set); match final
features without imposing conflicting middle-layer/skip alignments.

Train the single student with CE, full 205-way teacher KL and final-feature
MSE. A continuous-weight distillation stage precedes hard QAT; retained
weights and residual/skip parameters learn the missing transform. Exported
weights and mature whole-piece native scores select one checkpoint, followed
by one actual mixed measurement and paired full-stream archive restoration.

The static plan retains the original complete-piece learning range; the
current causal curve and mixed result determine its next research entry.
The completed [fixed near-zero mask](mlp-unit-budget.md) and
[seven-value map](weights-quant7.md) remain separate measured recipes.

```sh
python3 research/depth-distillation-budget.py
```

The standard-library script reads the existing census and checks total
INT4 counts and historical budget signs. It constructs zero models.
