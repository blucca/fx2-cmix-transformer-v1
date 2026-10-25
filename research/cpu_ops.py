"""Bounded CPU training operators; production inference and quantizers stay intact."""
import ctypes
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from pysrc import kda_reference as ref


_lib = None
_original_quantize_backward = None


def quantize_backward(ctx, grad_out):
    """Original QAT adjoint, tiled over examples to bound temporary tensors."""
    (scaled,) = ctx.saved_tensors
    if scaled.numel() <= 1 << 20:
        cached = SimpleNamespace(saved_tensors=(scaled,), quant=ctx.quant)
        return _original_quantize_backward(cached, grad_out)
    batch, blocks, width = scaled.shape
    tile = max(1, (1 << 20) // (blocks * width))
    grad_input = torch.empty_like(grad_out)
    grad_scale = torch.zeros(blocks, dtype=grad_out.dtype)
    for lo in range(0, batch, tile):
        hi = min(batch, lo + tile)
        part = SimpleNamespace(saved_tensors=(scaled[lo:hi],), quant=ctx.quant)
        gi, gs, _, _ = _original_quantize_backward(part, grad_out[lo:hi])
        grad_input[lo:hi] = gi
        # Each original call normalized by sqrt(part.numel * qmax).
        grad_scale.add_(gs, alpha=math.sqrt((hi - lo) / batch))
    return grad_input, grad_scale, None, None


def install_quantizer():
    global _original_quantize_backward
    from pysrc.quantization import QuantizeFunction
    if _original_quantize_backward is None:
        _original_quantize_backward = QuantizeFunction.backward
        QuantizeFunction.backward = staticmethod(quantize_backward)


def load_kernel(path):
    global _lib
    _lib = ctypes.CDLL(str(Path(path).resolve()))
    # Dimensions, boundary pointer, then contiguous float32 array pointers.
    for name, arrays in (("kda_forward", 7), ("kda_backward", 12)):
        fn = getattr(_lib, name)
        fn.argtypes = [ctypes.c_int] * 5 + [ctypes.c_void_p] * (1 + arrays)
        fn.restype = None


class _KDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, decay, beta, bounds):
        assert _lib is not None, "call load_kernel with the compiled shared library"
        T, H, K = q.shape
        V = v.shape[-1]
        arrays = tuple(x.contiguous() for x in (q, k, v, decay, beta))
        assert all(x.device.type == "cpu" and x.dtype == torch.float32 for x in arrays)
        assert k.shape == decay.shape == q.shape and v.shape[:2] == (T, H)
        assert beta.shape == (T, H)
        bounds = bounds.to(dtype=torch.int32, device="cpu").contiguous()
        assert bounds.ndim == 1 and bounds[0] == 0 and bounds[-1] == T
        assert torch.all(bounds[1:] > bounds[:-1])
        states = torch.empty(((T + 127) // 128, H, K, V), dtype=q.dtype)
        out = torch.empty_like(arrays[2])
        _lib.kda_forward(T, H, K, V, bounds.numel(), bounds.data_ptr(),
                         *(x.data_ptr() for x in (*arrays, states, out)))
        # Save one state per 128 tokens; native backward recomputes each block.
        ctx.save_for_backward(*arrays, bounds, states)
        return out

    @staticmethod
    def backward(ctx, go):
        q, k, v, decay, beta, bounds, states = ctx.saved_tensors
        T, H, K = q.shape
        grads = tuple(torch.empty_like(x) for x in (q, k, v, decay, beta))
        _lib.kda_backward(T, H, K, v.shape[-1], bounds.numel(), bounds.data_ptr(),
                          *(x.data_ptr() for x in
                            (q, k, v, decay, beta, states, go.contiguous(), *grads)))
        return *grads, None


def kda(q, k, v, g_raw, beta, A_log, dt_bias, cu_seqlens):
    assert q.shape[0] == 1
    T, H, K = q.shape[1:]
    bounds = cu_seqlens if cu_seqlens is not None else torch.tensor([0, T])
    decay = ref.kda_gate_reference(g_raw[0], A_log, dt_bias).exp()
    qn = ref.l2norm_reference(q[0]) * K ** -0.5
    kn = ref.l2norm_reference(k[0])
    return _KDA.apply(qn, kn, v[0], decay, beta[0], bounds).unsqueeze(0)


def window_attention(args):
    """Exact causal/window/document mask, bounded query tiles and full gradients."""
    assert args.queries.shape[0] == 1 and args.dropout == 0.0
    T = args.queries.shape[1]
    bounds = (args.cumulative_sequence_lengths[0].tolist()
              if args.cumulative_sequence_lengths is not None else [0, T])
    assert bounds[0] == 0 and bounds[-1] == T
    window = int(args.window_size) if args.window_size is not None else T
    assert window > 0
    q, k, v = (x.transpose(1, 2) for x in (args.queries, args.keys, args.values))
    documents = torch.empty(T, dtype=torch.int32)
    for i, (start, end) in enumerate(zip(bounds, bounds[1:])):
        documents[start:end] = i
    pieces = []
    for lo in range(0, T, 1024):
        hi = min(T, lo + 1024)
        # Keep aligned key tiles and overlap the short final query tile.
        # This layout preserves the paired full-model FP32/QAT result.
        qlo = lo if hi - lo >= 768 or T < 768 else max(0, (hi - 768) // 256 * 256)
        left = max(0, (qlo - window + 1) // 512 * 512)
        right = min(T, (hi + 511) // 512 * 512)
        qi = torch.arange(qlo, hi)[:, None]
        ki = torch.arange(left, right)[None, :]
        mask = ((ki <= qi) & (qi - ki < window)
                & (documents[qlo:hi, None] == documents[None, left:right]))
        out = torch.nn.functional.scaled_dot_product_attention(
            q[:, :, qlo:hi], k[:, :, left:right], v[:, :, left:right],
            # The original document mask is 3-D and selects CPU math SDPA.
            # Retain its pre-GEMM scaling, instead of CPU flash's post-GEMM scale.
            attn_mask=mask.unsqueeze(0), scale=args.scale,
            enable_gqa=q.shape[1] != k.shape[1])
        pieces.append(out[:, :, lo - qlo:])
    return torch.cat(pieces, dim=2).transpose(1, 2)


def check():
    """One check for the real head shape, resets, forward and all input gradients."""
    import json
    from pysrc.attention_function import AttentionFunctionArguments, sdpa_attention_function

    torch.manual_seed(245)
    torch.set_num_threads(1)
    T, H, K = 273, 3, 64
    base = [torch.randn(1, T, H, K) for _ in range(4)]
    base += [torch.rand(1, T, H), torch.randn(H), torch.randn(H * K) - 3]
    cu = torch.tensor([0, 1, 18, 149, T], dtype=torch.int32)
    weight = torch.randn_like(base[0])
    vals, grads = [], []
    for fn in (lambda *a: ref.kda_recurrent_reference(*a, dtype=torch.float32), kda):
        inp = [x.detach().clone().requires_grad_() for x in base]
        out = fn(*inp, cu)
        vals.append(out.detach())
        grads.append(torch.autograd.grad((out * weight).sum(), inp))
    torch.testing.assert_close(vals[1], vals[0], atol=3e-7, rtol=3e-5)
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=5e-4)
    report = {"kda_output_max_abs": (vals[1] - vals[0]).abs().max().item(),
              "kda_gradient_max_abs": [(a - b).abs().max().item() for a, b in zip(*grads)]}

    # Tile/window crossings and article reset; original 3-head/64-channel layout.
    T = 1377
    base = [torch.randn(1, T, 3, 64) * 0.5 for _ in range(3)]
    weight = torch.randn_like(base[0])
    vals, grads = [], []
    for fn in (sdpa_attention_function, window_attention):
        inp = [x.detach().clone().requires_grad_() for x in base]
        args = AttentionFunctionArguments("sdpa", *inp,
            torch.tensor([[0, 37, T]], dtype=torch.int32), None, 1024, None, 0.0, None)
        out = fn(args)
        vals.append(out.detach())
        grads.append(torch.autograd.grad((out * weight).sum(), inp))
    torch.testing.assert_close(vals[1], vals[0], atol=2e-6, rtol=3e-4)
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=8e-4)
    report.update(attention_output_max_abs=(vals[1] - vals[0]).abs().max().item(),
                  attention_gradient_max_abs=[(a - b).abs().max().item() for a, b in zip(*grads)])
    from pysrc.quantization import Quantization
    install_quantizer()
    ctx = SimpleNamespace(saved_tensors=(torch.randn(2051, 1, 768) * 90,),
        quant=Quantization(256, None, torch.bfloat16, torch.float32, "scaled_mean", None))
    go = torch.randn_like(ctx.saved_tensors[0])
    expected = _original_quantize_backward(ctx, go)
    actual = quantize_backward(ctx, go)
    torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=3e-6, rtol=3e-4)
    report["quantize_scale_gradient_max_abs"] = (actual[1] - expected[1]).abs().max().item()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    import sys
    load_kernel(sys.argv[1])
    check()
