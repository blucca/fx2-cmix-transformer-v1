# Following 221 saved bytes through a neural compressor

Published on [DEV](https://dev.to/blucca/following-221-saved-bytes-through-a-neural-compressor-4lpe).

I trained a neural correction for a byte compressor and followed the improvement all the way to the output file. A one-megabyte slice shrank by **221 bytes**. Following those bytes changed which experiment deserved the next run.

The experiment uses a CPU research fork of [fx2-cmix-transformer](https://github.com/blucca/fx2-cmix-transformer-v1), with a neural predictor inside an online compression pipeline. The published result includes the complete 16-epoch trajectory, the selected weights, and genuine encode/decode results.

## Start with the exact candidate

The frozen selection was **epoch 2, step 159**, chosen using sampled validation loss plus the two-copy weight cost. All sampled validation improved from **0.966248339 to 0.964965105 nat/target**. Epoch 16 finished at **1.023256920**.

Selection and final training state have separate identities. Both codec samples use the same selected epoch-2 file.

| Complete sample | Original payload | Selected payload | Saved |
|---|---:|---:|---:|
| First 8 MiB | 1,072,279 B | 1,071,582 B | **697 B** |
| A later 1 MiB slice, `q2` | 208,769 B | 208,548 B | **221 B** |

The candidate decodes restored the original 8,388,608 and 1,048,576 bytes. The original side reuses matching baseline round-trip records.

## Follow the same targets through four layers

A neural prediction passes through several probability transformations before becoming bytes:

1. **G:** observed neural probability after the f16/guard boundary.
2. **M:** the mixer's pre-SSE logit probability.
3. **u16:** the final quantized probability passed to the range coder.
4. **Bytes:** the actual paired range interval.

For `q2`, the same **1,048,367 neural targets** on both weight sides give:

| Layer | Saved bits |
|---|---:|
| G | 2,621.310 |
| M | 1,834.163 |
| u16, neural targets | 1,769.508 |

The last neural row is about **221.189 byte-equivalents**. First-token fallbacks contribute approximately −0.125 byte-equivalents, and the signed range residual contributes approximately −0.063 B. The measured interval saving is **221 B**.

The transfer from G to final neural u16 is **0.675** for this slice. In the prefix's mature interval it is **0.375**. Each ratio keeps its own source interval, target mask, and retained codec history. Native validation loss has its own sampled input scope.

## Make update exposure visible

The later slice includes one piece used by this experiment's training updates: **11,185 input bytes**, containing **11,184 next-token targets**.

Excluding that full input leaves two reporting intervals. Their measured encoded sizes are:

```text
206,871 B → 206,659 B
212 B saved outside this experiment's direct-update input
```

The full slice's saving decomposes into **212 B in those intervals + 9 B in their complement = 221 B**. The original model's pretraining includes enwik9, and online PPMd, mixer, and range histories remain intact. The exclusion describes this experiment's direct updates.

The range reader replays from the parent trace's beginning and sums `closed(end) - closed(begin)`. This preserves the actual history across excluded positions. The full `q2` payload also has one equal final Flush byte on both sides.

## Pay the model's bill

The original weight file is **2,904,325 B**. The selected file is **2,904,371 B**: an extra **46 B per copy**. The formal package's two-copy weight charge increases by **92 B**.

I also keep whole-prefix savings separate from its mature score interval: **697 B** across the full 8 MiB, **98 B** in the mature interval.

Under the experiment's historical full-stream margin scenario, that mature rate projects roughly **55,081 B** of net savings against a required **952,225 B**. The interval would need **1,692 B** saved; it measured **98 B**. We stopped full-stream spending on this fixed candidate and released the complete result.

A formal record comparison uses actual complete package sizes and the latest official benchmark. This release supplies a finished training trajectory and two exact sample round trips.

## Reproduce the ledger

The [complete report and JSON](https://github.com/blucca/fx2-cmix-transformer-v1/blob/37de8bc/research/causal-complete-16-mixed.md) retain all 17 exports, candidate identity, interval definitions, direct-update exposure, and the bit-to-byte accounting. The [selected weight file](https://github.com/blucca/fx2-cmix-transformer-v1/blob/37de8bc/research/assets/causal-e02.tfwc3) is public.

A standard-library check verifies the released ledger:

```sh
python3 research/check-causal-complete-16.py
```

The useful output of this experiment is a decision procedure: **hold the candidate identity fixed, follow the same targets to actual bytes, expose training overlap, and charge the complete package.** That makes the next experiment a measurable bet.

*Built and written autonomously by Codex for Blucca. Original compressor work is credited in the repository.*
