#!/usr/bin/env python3
"""One short CPU comparison of the existing Hutter SDPA tile schedule.

Standalone: no repository imports, checkpoint, training data, or parameter update.
The 3-D boolean mask and qlo/left/right arithmetic match research/cpu_ops.py.
Run with the already installed Debian CPU Torch environment; no installation.
"""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--targets", type=int, default=2049)
    ap.add_argument("--cpu-seconds", type=int, default=20)
    args = ap.parse_args()
    assert 1377 <= args.targets <= 4097
    assert 10 <= args.cpu_seconds <= 120
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (3 << 30, 3 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_seconds - 1, args.cpu_seconds))
    wall, cpu = time.monotonic(), time.process_time()
    result = {"status": "starting", "targets": args.targets, "heads": 3,
              "head_channels": 64, "window": 1024, "threads": 1,
              "optimizer_updates": 0, "cpu_limit_seconds": args.cpu_seconds,
              "address_space_limit_gib": 3, "target_peak_rss_mib": 600,
              "python": sys.version, "cases": []}

    def save(**updates):
        result.update(updates, cpu_seconds=time.process_time() - cpu,
                      wall_seconds=time.monotonic() - wall,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(args.output)

    def cpu_limit(signum, frame):
        save(status="cpu_budget", stage="cpu_limit")
        raise SystemExit(124)

    signal.signal(signal.SIGXCPU, cpu_limit)
    if importlib.util.find_spec("torch") is None:
        save(status="environment_missing_torch", stage="dependency_lookup")
        print(json.dumps(result), flush=True)
        return
    try:
        import torch
        from torch.utils.checkpoint import checkpoint
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(416)
        T, H, K, W = args.targets, 3, 64, 1024
        base = [torch.randn(1, T, H, K, dtype=torch.float32) * .5 for _ in range(3)]
        probe = torch.randn_like(base[0])
        # One short document followed by a complete long document; exact resets.
        bounds = [0, 37, T]
        documents = torch.empty(T, dtype=torch.int32)
        for i, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
            documents[lo:hi] = i
        schedule = []
        for lo in range(0, T, 1024):
            hi = min(T, lo + 1024)
            qlo = lo if hi - lo >= 768 or T < 768 else max(0, (hi - 768) // 256 * 256)
            left = max(0, (qlo - W + 1) // 512 * 512)
            right = min(T, (hi + 511) // 512 * 512)
            schedule.append(dict(lo=lo, hi=hi, qlo=qlo, left=left, right=right))
        result.update(torch_version=torch.__version__, schedule=schedule,
                      bounds=bounds, tensor_dtype="float32", device="cpu",
                      checkpoint_api="use_reentrant=False",
                      attention_api="scaled_dot_product_attention, 3-D bool mask, scale=None, dropout=0")

        def tiled(q, k, v, use_tiles):
            q, k, v = (x.transpose(1, 2) for x in (q, k, v))
            pieces = []
            for row in schedule:
                lo, hi, qlo, left, right = (row[n] for n in ("lo", "hi", "qlo", "left", "right"))

                # Bind tile coordinates at construction; rebuild the exact mask
                # during recomputation instead of keeping all masks as inputs.
                def tile(qi, ki, vi, qlo=qlo, hi=hi, left=left, right=right):
                    qp = torch.arange(qlo, hi)[:, None]
                    kp = torch.arange(left, right)[None, :]
                    mask = ((kp <= qp) & (qp - kp < W)
                            & (documents[qlo:hi, None] == documents[None, left:right]))
                    return torch.nn.functional.scaled_dot_product_attention(
                        qi, ki, vi, attn_mask=mask.unsqueeze(0), scale=None,
                        enable_gqa=qi.shape[1] != ki.shape[1])

                inputs = (q[:, :, qlo:hi], k[:, :, left:right], v[:, :, left:right])
                out = (checkpoint(tile, *inputs, use_reentrant=False)
                       if use_tiles else tile(*inputs))
                pieces.append(out[:, :, lo - qlo:])
            return torch.cat(pieces, dim=2).transpose(1, 2)

        reference_out, reference_grad = None, None
        for mode in ("original_tiles", "checkpoint_tiles", "outer_block_plus_checkpoint_tiles"):
            inputs = [x.detach().clone().requires_grad_() for x in base]
            storages, pack_calls, logical_bytes = {}, [0], [0]

            def pack(x):
                storage = x.untyped_storage()
                key = (storage.data_ptr(), storage.nbytes())
                storages[key] = storage.nbytes()
                pack_calls[0] += 1
                logical_bytes[0] += x.numel() * x.element_size()
                return x

            c, w = time.process_time(), time.monotonic()
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
                if mode == "outer_block_plus_checkpoint_tiles":
                    out = checkpoint(lambda *xs: tiled(*xs, use_tiles=True),
                                     *inputs, use_reentrant=False)
                else:
                    out = tiled(*inputs, use_tiles=mode == "checkpoint_tiles")
                saved = dict(saved_unique_storage_bytes=sum(storages.values()),
                             saved_logical_tensor_bytes=logical_bytes[0],
                             saved_storage_count=len(storages), saved_pack_calls=pack_calls[0])
            # The probe is deliberately outside hooks, so reported saved bytes
            # describe the attention function, with the same q/k/v in each case.
            grads = torch.autograd.grad((out * probe).sum(), inputs)
            row = dict(mode=mode, forward_backward_cpu_seconds=time.process_time() - c,
                       forward_backward_wall_seconds=time.monotonic() - w, **saved)
            if reference_out is None:
                reference_out = out.detach().clone()
                reference_grad = tuple(x.detach().clone() for x in grads)
                row.update(output_exact=True, gradient_exact=[True] * 3,
                           output_max_abs=0., gradient_max_abs=[0.] * 3,
                           gradient_relative_l2=[0.] * 3)
            else:
                row.update(output_exact=torch.equal(out.detach(), reference_out),
                           output_max_abs=float((out.detach() - reference_out).abs().max()),
                           gradient_exact=[torch.equal(a, b) for a, b in zip(grads, reference_grad)],
                           gradient_max_abs=[float((a - b).abs().max()) for a, b in zip(grads, reference_grad)],
                           gradient_relative_l2=[float((a.double() - b.double()).norm()
                                                       / b.double().norm().clamp_min(1e-30))
                                                 for a, b in zip(grads, reference_grad)])
                torch.testing.assert_close(out.detach(), reference_out, atol=2e-6, rtol=3e-4)
                for a, b in zip(grads, reference_grad):
                    torch.testing.assert_close(a, b, atol=3e-6, rtol=8e-4)
            result["cases"].append(row)
            save(status="running", stage=mode)
            del inputs, out, grads
            gc.collect()
        original, tiled_row = result["cases"][:2]
        result["saved_storage_ratio"] = (tiled_row["saved_unique_storage_bytes"]
                                         / original["saved_unique_storage_bytes"])
        result["all_output_and_gradients_bit_exact"] = all(
            r["output_exact"] and all(r["gradient_exact"]) for r in result["cases"])
        save(status="passed", stage="complete",
             peak_rss_within_target=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 600 * 1024)
    except Exception as exc:
        save(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
