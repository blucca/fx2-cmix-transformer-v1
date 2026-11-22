# Shared Q8 vocabulary: complete fixed-model research

Research checkpoint. The complete numerical and six-worker timing study on the AVX-512 host has finished; the fixed-text language gate and target AVX2/noVNNI gate are in progress.

This extends [the complete 24-Gemm study](../ort-q8-whole-decode/) on the same pinned DistilGPT2. The existing native kernel/API31 stays unchanged. Pad the vocabulary from 50,257 to 50,272, share one block32 Q8 code/scale pair between standard ONNX row Gather/dequantization and the native output head, then Slice back to 50,257 logits.

`tied_converter.py` reads the previous study's complete artifacts. `tied_runner.py` reuses its full-output oracle in fresh one-session workers. `language_probe.py` measures two 1,024-target teacher-forced blocks and four 64-token greedy continuations. `assemble_target.py` reconstructs the dense vocabulary from the small canonical Q8 transfer. All unchanged external tensor files are ordinary copies shared by the three new graphs; ONNX 1.23 rejects multiple hardlinks.

Source model and companion tokenizer: Xenova/distilgpt2@a41c10485c18a64b6606729b6a082330cbd8f49e, derived from Apache-2.0 DistilGPT2. Native operator/SDK licensing remains with the earlier study. Full acceptance results and reproducible invocation will accompany the finished study.
