# Exact window tiles, lower training activation storage

The compact CPU backend already recomputes complete transformer blocks and
carries KDA state adjoints across its 128-token checkpoints. A complete
35,162-target article used 5.277 GiB RSS. Reaching the native `2^17` piece
length requires reducing the remaining backward activation peak.

This opt-in change checkpoints **each existing 1,024-query math-SDPA tile**.
Its query/key layout, 3-D causal/document/window mask, FP32 scaling order and
full-piece loss stay fixed. Each backward tile rebuilds its mask and attention
intermediates, then returns the same query/key/value adjoints. Gradients pass
through the complete article, with the original neural reset only at its
boundary. Teacher inference uses the ordinary no-grad route.

The default `cpu_ops.window_attention(args)` stays on the original path.
`window_attention(args, checkpoint_tiles=True)` enables the memory tradeoff.
The production coder, weight format and inference code stay fixed.

## Measured attention result

One original-size CPU head layout, `T=2049, H=3, K=64`, includes a document
boundary at token 37, the 1,024-token window crossing and the short final tile.
Debian Torch 2.6.0 runs all three modes with one thread.

| Mode | Saved unique storage | Forward + backward CPU |
| --- | ---: | ---: |
| Original tiles | 64,326,924 B | 0.188800 s |
| Inner tile checkpoints | 4,720,896 B | 0.565863 s |
| Whole-block + inner tile checkpoints | 4,720,896 B | 0.299353 s |

**Every output and all three input gradients are bit-exact** in this fixture.
Saved unique storage falls **92.66%**. The complete check, including Torch
import, used 2.642018 CPU seconds and 490,360 KiB peak RSS. The table measures
one attention call; full-model memory and compute use their own measurements.

Storage accounting deduplicates tensors by their underlying storage. Tile
slices share the original query/key/value arrays. The measurement records
storage held by the forward graph; temporary recomputation memory enters the
process's actual peak RSS. The timing is a single short observation.

## Full-model resource check

The whole-piece probe uses the original 6M checkpoint and fixed signed BF16
scales. It compares the original no-grad teacher with a student carrying all
111 integer-coordinate tensors, then backpropagates complete CE + teacher KL.
It records loss, exact logits, frozen original parameters, finite gradients,
CPU and peak RSS. Optimizer updates are zero.

The native maximum shape is 131,071 next-token targets, within the training
backend's original 7 GiB virtual address-space ceiling. Genuine prepared
bytes exercise the original vocabulary mapping. A uniform f16 prior, where
used, defines a resource/numerical fixture; causal PPMd native and coded
measurements use their original independent experiment.

### Terminal full-model measurements

| Shape / backend | Teacher CPU | Tile student forward CPU | Complete backward |
| --- | ---: | ---: | --- |
| 1,346 targets / reference BLAS | 4.002737 s | 4.043078 s | Passed; all 111 CE/KL adjoints bit-exact against original route |
| 131,071 targets / OpenBLAS 0.3.29 | 66.312530 s | 67.173544 s | Failed at the 7 GiB virtual-memory ceiling |

Both pairs have **bit-exact teacher/student logits and zero teacher KL**,
using each run's own backend. The short pair preserves all 429 frozen
original tensors and 230 BF16 scale tensors; every integer-coordinate
gradient is finite and nonzero. Its process used 35.108133 CPU seconds,
738,572 KiB peak RSS and 1,197,520 KiB peak virtual address space.

The complete native-cap student's backward fails during whole-block
recomputation at `MLP.relu_squared`, while requesting **402,650,112 bytes**
(`131071 x 768 x 4`). Peak RSS was **6,443,480 KiB** and peak virtual address
space **7,250,308 KiB** before the rejected allocation. The child exited 1
after 143.719536 CPU seconds, including process startup. Full-length gradient
completion is a failed gate for this exact implementation at 7 GiB.

The first full-length attempt used reference BLAS: teacher 462.895825 CPU
seconds and partial child 848.635834 CPU seconds, followed by an orderly stop
and replacement with OpenBLAS. Preparation, that attempt, installation,
short pair and recovered full probe together consumed **1,050.152785 CPU
seconds** from the shared 1,800-second budget. The resource experiment is
closed. [Complete terminal receipt](results/tile-checkpoint-full-model.json).

The currently sampled causal training pieces remain within the original
complete-backward range. This opt-in operator stays available; the running
16-epoch causal data/teacher/trajectory and its original length limit stay fixed.

## Complete learning scope

Full-length learning is a separate data/trajectory decision. It entails the
full causal prior traversal, article-cluster split, original teacher cache,
complete-piece updates, native holdout scores, one actual mixed-codec
candidate and the paired full-stream archives. This activation change leaves
the currently running causal trajectory and its selection rule intact.


## Reproduce the attention check

```sh
python3 -B research/tile-checkpoint-attention-check.py \
  --output /YOUR/temp/tile-attention.json
```

Use the installed CPU Torch runtime; no model checkpoint or training data is
required for this operator fixture. [Receipt](results/tile-checkpoint-attention.json).

## Reproduce the whole-piece probe

The published [input builder](tile-checkpoint-inputs.py) extracts the same two
native pieces from the exact prepared stream and maps ascending vocabulary
members to token IDs. It writes uniform f16 prior rows and a fresh manifest;
source offsets, membership mask, native article boundaries and file digests
are checked. Use the original CPU model runtime and compiled KDA kernel from
[CPU backpropagation](cpu-backward.md).

```sh
python3 -B research/tile-checkpoint-inputs.py \
  --prepared /YOUR/temp/prepared --out /YOUR/temp/tile-inputs
python3 -B research/tile-checkpoint-probe.py \
  --checkpoint models/6m-q4-fp32.tch --kernel /YOUR/temp/cpu-kda.so \
  --manifest /YOUR/temp/tile-inputs/inputs.json --case short \
  --compare-original-gradients --cpu-seconds 180 \
  --output /YOUR/temp/tile-short.json
python3 -B research/tile-checkpoint-probe.py \
  --checkpoint models/6m-q4-fp32.tch --kernel /YOUR/temp/cpu-kda.so \
  --manifest /YOUR/temp/tile-inputs/inputs.json --case full \
  --cpu-seconds 893 --output /YOUR/temp/tile-full.json
```

Each probe sets one CPU and a 7 GiB address-space cap. The full command
reproduces the measured memory gate, subject to the runtime's allocator and
library footprint. Bit identity compares the two routes within one backend.
