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

The real 1,346-target complete-piece check also passed: original and tile
routes have bit-exact logits and all 111 integer-coordinate CE/KL adjoints,
with 429 frozen original tensors and zero coordinate updates. This observation
uses its own CPU runtime; the maximum-length resource check remains in progress.

The first maximum-shape runtime used Debian reference BLAS, taking 462.896
CPU seconds for the no-grad teacher. Its generic BLAS dependency is being
replaced with the existing single-thread OpenBLAS backend, within the same
experiment's total CPU ceiling. Full-model memory and completion use that
recovered run's own receipt.

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
