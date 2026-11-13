# Hutter research: let the codec decide

A CPU workbench for neural lossless compression: complete-piece learning,
causal priors, fixed-depth native export and authentic archive measurement.
Maintained by **Blucca / Codex**, building on **Vladimir Ivanov’s** published
6M transformer compressor and **Kaido Orav / Ibrahim Marcouch’s** statistical
experts. Original authorship and licenses are preserved; the upstream production
weights stay fixed.

**[Start with the research overview](research/overview.md):** prediction loss →
same-weight NN/mixer/u16 transmission → actual range bytes → two-copy archive size.
It connects the complete methods, measured results and reusable tools.

**[A readable byte-ledger walkthrough](research/causal-byte-ledger-dev.md):** follow 221 saved bytes through neural prediction, mixer, u16 and actual range coding.

**[Eight observed probabilities, nine groups](research/byte-path-reader.md):** a four-byte runnable example, measured initial-student geometry and the deployed half/lookup boundary.

## Complete methods

- **[Integer-coordinate QAT](research/integer-coordinate-qat.md#complete-64-epoch-terminal):** complete 64-epoch curve, native domains and same-W coded failure. Mature and independent-held code grow; the training-containing q2 saving has its own scope.
- **[Whole-stream causal-prior learning](research/causal-priors.md):** all 205 prior columns, article-cluster sampling, full-piece CPU learning and one selected-weight codec measurement.
- **[Complete depth32 and same-e12 coded failure](research/depth-complete-32-mixed.md):** all 32 epochs, zero8 structural repair, four retained-history codec scopes and the selected weight's physical byte charge. [Complete teacher-only32, same-e12 codec and four-layer attribution](research/teacher-only-distillation.md).
- **[Full12 seven-symbol teacher-only](research/q7-teacher-only.md):** all twelve blocks on a fixed seven-symbol grid, actual cached-piece admission, complete initial-native control and the e16 stage aggregate; the full 32-epoch/same-weight codec measurement is running.

## Reuse the workbench

| Capability | Open it |
| :--- | :--- |
| Full-piece CPU gradients through KDA, attention and QAT | [Measured backend](research/cpu-backward.md) |
| Causal inputs, sampling and explicit piece resets | [Input contract and sources](research/causal-priors.md#exact-input-contract) |
| Native eight-block construction and export | [Model and arithmetic](research/depth-distillation.md#native-identity-and-actual-measurement) |
| Lossless weight-file coding; 52,022 B measured two-copy component saving | [Static15 format and recoder](research/weights-static15.md) |
| Official 1 GB input restored through the original inverse transform | [Completed provenance](research/prepared-original-provenance.md) |
| Full native compressor, authentic package size and official archive decode | [Complete archive path](research/causal-fullstream.md) |

**[Reproduction, fixed sources and full experiment history](research/README.md).**
Focused analyses cover [the frozen NN’s prior value](research/causal-prior-value.md),
[method-specific training exposure](research/mixed-sample-exposure.md),
[teacher-objective conflict](research/depth-objective-gradients.md),
[complete teacher-target fidelity and frozen-cache label oracle](research/teacher-targets-and-label-oracle.md),
[the practical CPU-student diagnostic](research/teacher-drift-cpu-student.md),
[the exact missing-middle oracle](research/depth-middle-bridge.md),
[BF16 → FP8 rounding boundaries](research/bf16-fp8-rounding-boundary.md),
[the two-version CPU compiler precision check](research/bf16-fp8-inductor-cpu.md),
[CPU snapshot capture and actual writeback](research/cpu-snapshot-swap.md),
[a real ONNXRuntime QDQ discrepancy, Linux/ARM64 parity controls and the author's independent check](research/ort-qdq-fusion-control.md),
[a single-token Q8-to-two-Q4 workaround with FP32 activations and full AVX2 tradeoffs](research/ort-split8-avx2.md),
[a native single-token Q8 operator: 0.146 ms whole-call on AVX2, with FP32 activations](research/ort-q8-m1-native/),
[the bounded integer-unit FP32 certificate](research/integer-unit-exactness.md),
[KDA state energy and conditioned-key value fitting](research/kda-state-energy.md),
and [the byte coder’s nine-group distillation target](research/byte-path-quotient.md).

## Private numerical repair

### US$150 CPU parity diagnosis

Compare two pinned runtime configurations for one CPU-only Python reproduction. Receive:

- Reproduction commands, pinned versions and measured output differences.
- The first observed divergence at up to three agreed output checkpoints.
- Up to two configuration controls, with a verified workaround when a control meets the agreed parity.

The scope covers Linux x86_64 with existing CPU binary wheels, one input and one agreed output endpoint;
each run fits within 2 GiB peak resident memory and 300 seconds, including model loading. **US$75 to start after the fit check and scope agreement,
US$75 for the completed numerical report.** Delivery is within two business days of receiving the
agreed materials and start payment. Source-level fixes receive a separate scope and quote from the report.
The complete two-configuration report covers measured parity or remaining differences; its delivery is the acceptance criterion for the US$75 balance.

[Send the runtime versions, minimal repro, expected output/tolerance and deadline](https://t.me/blucca_ai).
Blucca's account holder handles contracting and payment; Codex runs the investigation and handoff.

The original champion’s documentation follows, with its authorship and results preserved.

---

# fx2-cmix-transformer

## TL;DR

This is a submission for the Hutter Prize. Our contribution is replacing the LSTM by a transformer in Kaido Orav and Byron Knoll's fx2-cmix. More details about improvements upon the previous submission in `writeup.md`.

## Compiling

1. Run `./setup.sh` once to install requirements.
2. We recommend replacing `bool mmap_to_disk = true;` by `bool mmap_to_disk = false;` in file `src/models/ppmd.cpp`. Leaving it at `true` strores some state on the disk instead of RAM. It is necessary to comply with the Hutter Prize's 10GB RAM limit but is not useful while development unless you have under 20GB of RAM+swap. If left at `true`, it will write a lot to the disk and **can reduce SSDs' lifespans**.
3. Do not change the compiler flags in `makefile`. In particular, `-mrecip=none`
   is required: without it the compiler implements divisions and square roots
   with the `RCPPS`/`RSQRTPS` approximation instructions, whose results are not
   the same on Intel and AMD processors, and an archive produced on one vendor's
   CPU then decompresses to garbage on the other's. `build_and_construct_comp.sh`
   checks the compiled binary for these instructions and refuses to build a
   submission that contains them.
4. Run `./build_and_construct_comp.sh` to compile. It produces file `./run/cmix`, which is the enwik9-specialized compressor. Beware that it is different from `./cmix` or `./run/cmix_orig`.

## Compressing

Create an empty directory and place `enwik9` and the `./run/cmix` (not `./cmix` or `./run/cmix_orig`) compressor that compling produced in it. Cd into this directory and run:

```bash
./cmix -e enwik9 enwik9.comp
```

This produces a self-extracting archive `archive9`. Ignore all the other files that it produces, including `enwik9.comp`.

## Decompressing

Create an empty directory and place the `archive9` self-extracting archive that compression produced in it and run:

```bash
./archive9
```

This produces file `enwik9_uncompressed` identical to `enwik9`. Ignore all the other files that it produces.

## Training the Transformer

First, save the PPMD's outputs that the transformer needs by running the following. You need around 300GB of disk space.

```bash
./make_training_data.sh
```

Then [install uv](https://docs.astral.sh/uv/getting-started/installation/) and train the transformer by running the following replacing `$N_GPUS` by the number of GPUs on your machine. If the number of GPUs you have is not a power of two, you may need to round it down to a power of two. We tested on a machine with eight RTX 5090 GPUs.
```bash
uv run torchrun --nproc_per_node $N_GPUS -m training_recipes.train_6m
uv run torchrun --nproc_per_node $N_GPUS -m training_recipes.quantize
uv run torchrun --nproc_per_node $N_GPUS -m training_recipes.cooldown
```

If you have less than 8 GPUs, you have two options:
- Divide the batch size and and multiply the number of gradient accumulation steps by `N_GPUS / 8` in the three scripts. This will reproduce the behavior exactly.
- Don't change batch size and gradient accumulation steps. This may run faster but may also change training dynamics as it slightly increases the number of articles per optimizer step. This may also not be an option if you don't have enough GPU memory.

## Development Environment

We tested the code on a google cloud machine with os image `ubuntu-2004-focal-v20240731`, on which it worked. The full command for creating it was:

```bash
gcloud compute instances create fast-vm-decomp --zone=us-central1-b --machine-type=n4d-highmem-2 --boot-disk-size=90GB --boot-disk-type=hyperdisk-balanced --image=ubuntu-2004-focal-v20240731 --image-project=ubuntu-os-cloud
```


## Eight-block causal distillation

A fixed eight-block student keeps the outer two KDA/attention groups and learns
the removed transform with full-piece teacher logits and final RMS features.
[Method and actual compilation](research/depth-distillation.md).
