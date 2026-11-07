# Fixed activation scales after eight-block distillation

A bounded, read-only experiment on one original hard-coordinate training snapshot. **Removing activation quantization raises both label loss and teacher KL on this input.** The next output-objective experiment keeps the original scales.

## Paired experiment

The snapshot has completed 23 full epochs and is partway through epoch 24: step **1,832**, 68,310,679 seen targets. The input is the same complete **1,199-target** training piece used in the [parameter-gradient measurement](depth-objective-gradients.md): manifest ordinal 8, capture range `[5321,6521)`, explicit reset `[0,1199]`, and all 205 causal PPMd columns. Original twelve-block teacher logits and final hidden activations come from the existing cache.

- **A — deployed activation quantization:** all 80 original batched INT8 quantizers remain active.
- **B — floating-activation counterfactual:** those same 80 quantizers return their input. All 75 hard INT4 coordinate tensors, 59 raw tensors, signed BF16 scales, layer mapping, KDA/attention path and scale-folded FP32 GEMM remain fixed.

Both passes run under `torch.no_grad()` with zero dropout. The retained GEMM expression is `linear(x / sa, weight / sw) * (sa * sw)`, which accepts floating inputs. B evaluates this fixed model with floating activations; native integer inference and new trained scales belong to separate experiments.

| Same input / snapshot | A: original INT8 | B: floating | B − A |
|---|---:|---:|---:|
| Label CE, nats/target | 0.6584602594 | 0.6934974790 | **+0.0350372195** |
| KL(original teacher ∥ student) | 0.1375257969 | 0.1617886275 | **+0.0242628306** |
| 0.1 × final-hidden MSE | 0.0245388336 | 0.0251851827 | +0.0006463490 |
| 2KL + 0.1hidden | 0.2995904274 | 0.3487624377 | +0.0491720103 |

The scope is one observed training piece and one mid-trajectory hard snapshot. The full held set, the selected epoch-12 model and actual mixed-codec bytes retain their own measurement scopes.

## Activation census

The A pass visits all **80 quantizers once**, observing **22,115,555 activation elements** across the network. Their pre-round `input / signed-BF16-scale` values exceed the quantizer range at **5,838 elements (0.0263977%)**. Nonzero inputs round to zero at **1,145,989 elements (5.18182%)**. These denominators count quantizer elements across layers.

Summed activation-error squared divided by summed input squared is **0.000181218**. Range-exceeding elements carry 57.7439% of the local activation-error squared sum. The complete result contains both passes' per-module counts, signed scale ranges, layer-source mapping and error sums. B retains shadow old-scale range counts while applying identity output.

The paired output difference has RMS **0.534970 logits** and **0.122288 final-hidden units**. CE, KL and hidden loss are independently reconstructed from the supplied scientific arrays.

## Resource and invariance contract

One original worker executes the diagnostic on CPU 1 / one BLAS thread. `wait4` accounts **5.614710 child CPU seconds + 0.054286839 supervisor seconds = 5.668996839 CPU seconds**; wall time is **9.904192 seconds**, maximum single-process RSS **528,496 KiB**. The bounds are 180 CPU seconds, 240 wall seconds and 3 GiB address space.

Byte comparisons preserve all **134 z/raw tensors**, all **289 model-state tensors**, and all **155 scale tensors**. Two model forwards complete; optimizer updates, gradient calls, teacher generation, native evaluator/codec execution, model export and scale fitting each remain zero. The original training and sole selected-weight mixed runs continue.

## Reproduce and decision

- [Complete scientific result](results/depth-activation-counterfactual.json)
- [Original inputs, cached teacher and A/B logits/hidden](results/depth-activation-counterfactual.npz)
- [Read-only measurement source](depth-activation-counterfactual.py)
- [Minimal result replay](check-depth-activation-counterfactual.py)

The measurement CLI accepts the original job/source, CPU kernel and a fresh output directory; `--help` documents the bounded execution. The JSON replay uses the standard library. The NPZ arrays can be inspected with NumPy.

**Decision: stop new activation-calibration engineering.** This immediate paired intervention worsens all three measured losses. The proposed curvature-matched [2KL output objective](teacher-objective-audit.md) retains the original fixed activation scales; complete new learning remains conditional on the original same-weight mixed result and complete remaining delivery.
