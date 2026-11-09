# CUDA resource gate for the retained q7 arithmetic

The completed resource experiment has two scopes: generated hard operands for the seven existing matrix geometries, and one original full12 q7 causal article with 1,199 targets. Both passed their recorded gates. The CUDA experiments performed **zero optimizer updates and zero native evaluations**. The original CPU learning recipe remains its own experiment.

## Full original article: completed

Python 3.13.16, Torch 2.11.0+cu130, CUDA 13.0, driver 580.178.04; two Tesla T4 devices were available and cuda:0 was used. Each device exposed 15,636,037,632 bytes, compute capability 7.5. Computation was FP32 with TF32 disabled. T4 hardware BF16 support is false; the default Torch query includes emulation.

The original piece `[0,5321,6521]` supplied 1,200 u8 tokens, raw f16 priors and the original cached f32 teacher logits, all 205 columns. The original checkpoint, twelve layers / width 192 / nine KDA + three attention layers, fixed signed-BF16 q7 scale map, 111 external coordinate tensors / 5,868,864 coordinates and 429 frozen tensors were retained. Data VJP was `2KL(teacher||student)`, divided by 1,199 after backward; the rate VJP was `2H/586459020`, with seven centers and tau 0.5. CE was detached reporting.

| Recorded CUDA phase | Seconds |
| --- | ---: |
| Model forward | 5.175270039 |
| Data backward, including checkpoint recomputation | 11.299987716 |
| Rate entropy forward/backward | 0.158445493 |
| Sum of the three compute phases | **16.633703248** |
| Whole script, including loading and checks | **39.366905290** |

The three compute phases use CUDA-synchronized monotonic timers. Detached CE reporting, differentiable KL construction, loading and inter-phase checks occupy the other script wall time. Recorded compute throughput was 72.0826 targets/s. All 111 coordinate gradients were finite and nonzero; every coordinate and frozen parameter retained its initialization. Peak allocated/reserved GPU memory was **281,286,144 / 337,641,472 bytes**. The script had an 840-second wall limit and the original platform task ended COMPLETE.

[Full raw result](results/q7-cuda-full1199-resource.json) · [Full entry point](q7-cuda-full1199-resource.py) · [CPU reference and source hashes](results/q7-cuda-full1199-cpu-reference.json).

## Original CPU reference and actual arithmetic scope

The original CPU preflight used the same article, checkpoint, q7 map, fixed scales, cached teacher and objective. Its source at `52c60a526874f84f7aeddf4d722c297351a949f6` remains byte-identical to the underlying resource source. The temporary CUDA source copy changes one `CastedLinear` device assertion from CPU to CUDA; the original CPU source and ABI remain intact.

| Recorded initial quantity | Original CPU | CUDA reference | CUDA − CPU |
| --- | ---: | ---: | ---: |
| CE / target, report only | 1.036659272538 | 1.037438018010 | +0.000778745471 |
| Teacher KL / target | 0.365677859805 | 0.367251640364 | +0.001573780559 |
| `2KL + 2H/N` | 0.768070476218 | 0.771218037736 | +0.003147561518 |
| Complete gradient L2 norm | 0.392868611880 | 0.395312642047 | +0.002444030167 |
| Finite / nonzero coordinate tensors | 111 / 111 | 111 / 111 | 0 / 0 |

These are complete-pipeline scalar summaries; the norm difference is +0.622099% relative to CPU. The stored full-model results contain two gradient norms. Paired linear-VJP comparisons belong to the separate generated-operand experiment below. Initial soft entropy was 10,765,850.08984375 nats on CPU and 10,765,850.20703125 nats on CUDA, a +0.1171875-nat device-reduction difference.

The CPU receipt records **5.915475907 single-core CPU-seconds** for the broader preflight check. Its timer sums parent and child CPU time and covers article/cache reads, forward, both backward terms, gradient checks, an AdamW update/projection, frozen-state validation, geometry and payload/static15 export with a byte check. The diagnostic step left hard symbols and W bytes unchanged; coordinates, optimizer and RNG were restored after that timer ended. The stored CPU timing is this one aggregate; the comparison JSON marks its separate phase and CPU-wall slots null. The ratio of the recorded CUDA compute total to the CPU check total is **2.811896**, with those timer scopes preserved.

The backend comparison follows the actual source:

- **Quantized linear:** both use FP32 integer-unit GEMM and one post-GEMM fold of signed BF16 activation/weight scales. Hard INT8 × q7 dots satisfy `768 * 128 * 3 < 2^24` under the retained integer-recovery contract.
- **KDA:** CPU uses its compiled C ABI recurrence and adjoint, retaining a state every 128 tokens. CUDA uses the pure-torch per-token FP32 recurrent reference and autograd.
- **Convolution and gated RMS norm:** both already use the same pure-torch reference functions, on their respective devices.
- **Attention:** CPU uses its original math-SDPA window tiles and short-tail query overlap; CUDA uses one dense math-SDPA layout. Both preserve the causal, 1,024-window and article masks.
- **Activation quantizer:** CUDA uses the original untiled adjoint. CPU enables its tiled adjoint above `2^20` scaled elements; this article's largest `1199 * 768` operand stays below that threshold.

KDA reductions, SDPA kernels/layouts, nonlinear functions and noninteger VJP reductions follow their FP32 device execution paths. The table records the resulting full-model differences. Local hard-integer exactness and full-model floating summaries have these respective scopes.

### Resource decision and visible cost drivers

**Keep the free CUDA resource available; stop training expansion of this pure-torch reference backend. New training, extra benchmarks, FLA/compile migration and native evaluations are allocated zero.** The original CPU 32-epoch / 48-CPUh method and its sole-W 8-CPUh codec budget stay unchanged.

Data backward accounts for **67.9343%** of the recorded compute total; the rate entropy phase accounts for **0.9526%**. The source executes a narrow KDA state recurrence through Python once per token, with many small tensor operations and its autograd graph. Layer checkpointing replays that path during backward: the recorded adapter counts are 18 KDA, 54 convolution and 18 gated-norm calls, matching the 9 / 27 / 9 forward calls plus one replay. The original CPU KDA runs its recurrence/adjoint as compiled loops. These are source-identified cost drivers; the timing evidence remains the three measured phases above.

## Reproduce the full resource probe

Use Linux, Torch 2.11 with CUDA, NumPy and jaxtyping. The input directory contains the retained `sample.npz` and `input-readback.json`; the latter is the `data_readback` object in the raw result. Required arrays are `tokens: u8[1200]`, `priors: f16[1,1199,205]`, and `teacher_logits: f32[1,1199,205]`. Their payload hashes and the original sample/checkpoint hashes are in the CPU-reference JSON. The original capture/cache assets and platform dataset references remain private.

From this public repository, prepare the same temporary source copy without changing the production source:

```sh
python -B - <<'PY'
import ast, subprocess
from pathlib import Path
ref = '513a89c1d339848717684fa5993dea482a85208c'
out = Path('temp/q7-cuda-source')
names = ['pysrc/model.py', 'pysrc/quantization.py',
         'pysrc/attention_function.py', 'pysrc/kda_reference.py',
         'research/cpu_model.py', 'research/q7_teacher_only.py',
         'research/integer-coordinate-qat.py', 'research/entropy-qat.py',
         'training_recipes/cooldown.py', 'models/6m-q4-fp32.tch']
for name in names:
    data = subprocess.check_output(['git', 'show', f'{ref}:{name}'])
    if name == 'pysrc/model.py':
        old = b'assert x.device.type == "cpu" and x.dtype == torch.float32'
        assert data.count(old) == 1
        data = data.replace(old, old.replace(b'"cpu"', b'"cuda"'), 1)
    if name.endswith('.py'):
        ast.parse(data)
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
PY

Q7_CUDA_INPUT_DIR=/path/to/retained-article \
Q7_CUDA_SOURCE_DIR=temp/q7-cuda-source \
RESULT_PATH=temp/q7-cuda-full1199-resource.json \
python -B research/q7-cuda-full1199-resource.py
```

The public entry point retains the measured v3 compute body and adds configurable output/input/source paths plus a corrected operator-scope docstring. On Kaggle, its original mounted bundle layout is also accepted. This probe performs zero optimizer updates, downloads and network requests.

## Generated seven-geometry arithmetic gate

One generated hard matrix per geometry represented the original 111-matrix census. All seven shapes passed: eight integer rows per shape matched integer products exactly, and every complete-batch hard-STE VJP had relative L2 error 0.0 against its CPU FP32 expression. Whole-script wall time was 7.546727s including setup; peak CUDA allocation was 34,126,848 bytes.

| Output × input | Existing matrices | One GEMM, ms |
| --- | ---: | ---: |
| 64 × 192 | 18 | 0.0445 |
| 192 × 64 | 18 | 0.0380 |
| 192 × 192 | 48 | 0.0596 |
| 192 × 205 | 1 | 0.0617 |
| 192 × 768 | 12 | 0.1795 |
| 205 × 192 | 2 | 0.0619 |
| 768 × 192 | 12 | 0.1601 |

The GEMM timings use eight batch-1199 repetitions. Signed/tiny BF16 scales are folded in FP32 after integer-unit GEMM. With Torch 2.11 and a CUDA GPU:

```sh
RESULT_PATH=temp/q7-cuda-linear-resource.json python research/q7-cuda-linear-resource.py
```

The linear script has a 120-second wall limit and performs zero optimizer updates, model downloads and networking. [Linear raw result](results/q7-cuda-linear-resource.json) · [Linear script](q7-cuda-linear-resource.py) · [Retained q7 method](q7-teacher-only.md).
