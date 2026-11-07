#!/usr/bin/env python3
"""One uniform CPU cast control, eager and two existing Inductor policies.

Synthetic inputs. The control covers finite positive E4M3FN in [1,2).
TORCHINDUCTOR_CACHE_DIR should point to a disposable experiment directory.
"""
import json
import os
from pathlib import Path
import resource
import time

START = time.monotonic()
OUT = Path(os.environ.get("CAST_OUTPUT", "."))
OUT.mkdir(parents=True, exist_ok=True)
result = {"status": "started", "scope": "CPU; FP32 exact uniform [1,2); explicit BF16 and E4M3FN casts"}


def save():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    result["resources"] = {"cpu_seconds": time.process_time() + r.ru_utime + r.ru_stime,
                           "wall_seconds": time.monotonic() - START,
                           "python_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           "child_peak_rss_kib": r.ru_maxrss}
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def staged(x):
    return x.to(torch.bfloat16).to(torch.float8_e4m3fn).to(torch.float32)


try:
    import torch
    from torch._inductor import config
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config.compile_threads = 1
    config.cpp.threads = 1
    n = 1 << 16
    x = 1 + torch.arange(n, dtype=torch.float32) / n
    eager = staged(x)
    direct = x.to(torch.float8_e4m3fn).to(torch.float32)
    changed = torch.nonzero(eager != direct).flatten()
    assert changed.numel() == 2048
    scalar = torch.tensor([1.0626], dtype=torch.float32)
    assert (scalar.to(torch.bfloat16).item(), staged(scalar).item(),
            scalar.to(torch.float8_e4m3fn).float().item()) == (1.0625, 1.0, 1.125)
    result.update(torch_version=torch.__version__, python_version=os.sys.version,
                  torch_git_version=torch.version.git_version,
                  emulate_precision_casts_default=config.emulate_precision_casts,
                  uniform={"samples": n, "eager_staged_vs_direct_changed": changed.numel(),
                           "first_changed_indices": changed[:10].tolist()},
                  scalar={"fp32": scalar.item(), "bf16": scalar.to(torch.bfloat16).item(),
                          "staged_fp8": staged(scalar).item(), "direct_fp8": scalar.to(torch.float8_e4m3fn).float().item()},
                  compiled=[])
    save()
    cache = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    for emulate in (False, True):
        torch._dynamo.reset()
        prior = set(cache.rglob("*.cpp")) if cache.exists() else set()
        row = {"emulate_precision_casts": emulate}
        try:
            with config.patch(emulate_precision_casts=emulate):
                compiled = torch.compile(staged, backend="inductor", fullgraph=True)
                y = compiled(x)
            assert y.shape == eager.shape and y.dtype == torch.float32 and torch.isfinite(y).all()
            mismatch = torch.nonzero(y != eager).flatten()
            row.update(status="measured", mismatches_vs_eager=mismatch.numel(),
                       mismatches_vs_direct=int(torch.count_nonzero(y != direct)),
                       max_abs_vs_eager=(y - eager).abs().max().item(),
                       first_mismatch_indices=mismatch[:10].tolist())
        except Exception as e:
            row.update(status="failed", error_type=type(e).__name__, error=str(e)[:5000])
        generated = sorted(set(cache.rglob("*.cpp")) - prior)
        row["generated_cpp"] = []
        for i, p in enumerate(generated):
            name = f"emulate-{str(emulate).lower()}-{i}.cpp"
            (OUT / name).write_bytes(p.read_bytes())
            row["generated_cpp"].append(name)
        result["compiled"].append(row)
        save()
    result["status"] = "measured"
except Exception as e:
    result.update(status="failed", error_type=type(e).__name__, error=str(e)[:5000])
finally:
    save()
    print(json.dumps(result), flush=True)
