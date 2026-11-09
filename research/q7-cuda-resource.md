# CUDA resource gate for the retained q7 arithmetic

This is a resource measurement for the existing full12 q7 research, using generated hard operands. The scientific scope is the seven matrix geometries, integer-unit FP32 dot products and hard-STE VJP. Each geometry represents its existing matrix census; the experiment uses one generated matrix per geometry.

## Actual run

Python 3.13.16, Torch 2.11.0+cu130, CUDA 13.0, driver 580.178.04; two Tesla T4 devices were available, cuda:0 was used. Compute capability is 7.5. FP32 is the compute dtype; TF32 is disabled. T4 hardware BF16 support is false; Torch's default BF16 query includes emulation and returned true.

All seven shapes passed: eight integer rows per shape compared exactly to integer products, and every complete-batch hard-STE VJP had relative L2 error 0.0 against its CPU FP32 expression. The census is 111 matrices / 5,868,864 coordinates. Total script wall time was 7.546727s including setup; peak CUDA allocation was 34,126,848 bytes. The original model, optimizer trajectory and native decoder remain their existing experiments.

| Output × input | Existing matrices | One GEMM, ms |
| --- | ---: | ---: |
| 64 × 192 | 18 | 0.0445 |
| 192 × 64 | 18 | 0.0380 |
| 192 × 192 | 48 | 0.0596 |
| 192 × 205 | 1 | 0.0617 |
| 192 × 768 | 12 | 0.1795 |
| 205 × 192 | 2 | 0.0619 |
| 768 × 192 | 12 | 0.1601 |

Local GEMM timings use eight batch-1199 repetitions. Full recurrent forward/backward is a separate original-article resource gate; it uses the original pretrained checkpoint and cached teacher, the fixed q7 scale mapping, pure-PyTorch KDA/conv/norm and math-SDPA. Its complete result will determine whether that reference backend is useful for the next complete learning method.

## Reproduce the measured arithmetic

With Torch 2.11 and a CUDA GPU:

```sh
RESULT_PATH=temp/q7-cuda-linear-resource.json python research/q7-cuda-linear-resource.py
```

The script has a 120-second wall limit and performs zero optimizer updates, model downloads or networking. Signed/tiny BF16 scales are folded in FP32 after integer-unit GEMM. The bound `768 * 128 * 3 < 2^24` keeps all hard partial sums exactly representable. [Raw result](results/q7-cuda-linear-resource.json), [script](q7-cuda-linear-resource.py), [retained q7 method](q7-teacher-only.md).
