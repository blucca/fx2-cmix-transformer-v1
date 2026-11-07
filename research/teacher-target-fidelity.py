#!/usr/bin/env python3
"""Read-only W0 CPU teacher NLL on original complete causal held/audit pieces."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "kernel", "data", "baseline", "source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--cpu", type=int)
    p.add_argument("--cpu-seconds", type=int, default=1200)
    p.add_argument("--wall-seconds", type=int, default=1800)
    a = p.parse_args()
    assert 10 <= a.cpu_seconds <= 1200 and 10 <= a.wall_seconds <= 1800
    sys.dont_write_bytecode = True
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    core = min(os.sched_getaffinity(0)) if a.cpu is None else a.cpu
    os.sched_setaffinity(0, {core})
    resource.setrlimit(resource.RLIMIT_AS, (7 << 30, 7 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (a.cpu_seconds - 2, a.cpu_seconds))
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for key in ("TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
        path = out / key.lower()
        path.mkdir()
        os.environ[key] = str(path)
    sys.path.insert(0, str(a.source.resolve()))
    started = time.monotonic()
    result = dict(schema="hutter.w0-cpu-teacher-fidelity.v1", status="running",
                  native_scaled_gemm=True, cpu_affinity=[core], threads_per_library=1,
                  cpu_limit_seconds=a.cpu_seconds, wall_limit_seconds=a.wall_seconds,
                  address_space_limit_bytes=7 << 30, teacher_forwards=0,
                  optimizer_updates=0, native_calls=0, exports=0, teacher_cache_files=0,
                  nll_arithmetic="FP32 model logits; FP64 logsumexp and target sum",
                  paths={k: str(getattr(a, k).resolve()) for k in
                         ("checkpoint", "kernel", "data", "baseline", "source")}, scopes={})

    def save():
        result.update(cpu_seconds=time.process_time(), wall_seconds=time.monotonic()-started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp = out / "result.tmp"
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        tmp.replace(out / "result.json")

    def limit(signum, frame):
        raise TimeoutError("cpu_limit" if signum == signal.SIGXCPU else "wall_limit")

    signal.signal(signal.SIGXCPU, limit)
    signal.signal(signal.SIGALRM, limit)
    signal.alarm(a.wall_seconds)
    save()
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from cpu_model import load_model
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        manifest = json.loads(a.data.read_text())
        baseline = json.loads(a.baseline.read_text())
        assert manifest["method"] == "causal-whole-piece-v1"
        assert manifest["source_bytes"] == 586459020 and baseline["status"] == "passed"
        samples = manifest["samples"]
        assert [s["name"] for s in samples] == ["causal", "long-audit"]
        audit = manifest["independent_long_audit"]
        assert audit["training_article_overlap"] == 0 and samples[1]["train_pieces"] == []
        assert audit["article"] == 243425
        assert [audit["source_begin"], audit["source_end"]] == [570595966, 570727038]
        teacher, counts = load_model(a.checkpoint, a.kernel, native_scaled_gemm=True)
        teacher.requires_grad_(False).eval()
        assert teacher.cfg.n_layers == 12 and sum(teacher.cfg.kimi_linear) == 9
        assert teacher.cfg.dropout == 0 and teacher.cfg.window_size == 1024
        result["torch_version"] = torch.__version__
        for sample, scope, expected_pieces, expected_targets in zip(
                samples, ("causal_held", "independent_giant"), (473, 1), (1082505, 131071)):
            score = sample["score"]
            tokens_path, priors_path = Path(score["tokens"]), Path(score["priors"])
            bounds = np.fromfile(score["boundaries"], dtype="<i4").tolist()
            native = baseline["steps"][sample["name"]]["result"]
            assert len(bounds) == expected_pieces + 1 and bounds[0] == 0
            assert all(1 < hi-lo <= 131072 for lo, hi in zip(bounds, bounds[1:]))
            assert bounds[-1] == tokens_path.stat().st_size == native["input_bytes"]
            assert priors_path.stat().st_size == bounds[-1]*410 == native["prior_bytes"]
            assert native["piece_bounds"] == bounds and native["explicit_piece_bounds"] is True
            assert native["vocab"] == 205 and native["rope_offset"] == 0
            assert len(native["piece_losses"]) == native["pieces"] == expected_pieces
            assert sum(hi-lo-1 for lo, hi in zip(bounds, bounds[1:])) == score["targets"]
            assert score["targets"] == native["total"]["count"] == expected_targets
            rows = []
            total = dict(expected_pieces=expected_pieces, expected_targets=expected_targets,
                         scored_targets=0, complete=False, per_piece=rows,
                         score=score, native_baseline_total=native["total"])
            result["scopes"][scope] = total
            for i, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
                n = hi-lo-1
                reference = native["piece_losses"][i]["loss"]
                assert reference["count"] == n
                tok = np.fromfile(tokens_path, dtype=np.uint8, count=n+1, offset=lo)
                prior = np.fromfile(priors_path, dtype="<f2", count=n*205,
                                    offset=lo*410).reshape(n, 205)
                assert tok.shape == (n+1,) and int(tok.max()) < 205
                x = torch.from_numpy(tok[:-1].astype(np.int64))[None]
                y = torch.from_numpy(tok[1:].astype(np.int64))
                pp = torch.from_numpy(prior)[None]
                neural_bounds = torch.tensor([[0, n]], dtype=torch.int32)
                assert bool(torch.isfinite(pp).all())
                with torch.no_grad():
                    hidden = teacher.compute_last_activation(input_tokens=x, priors=pp,
                                      cumulative_sequence_lengths=neural_bounds)
                    logits = teacher.last_activation_to_logits(hidden, priors=pp)
                    assert logits.dtype == torch.float32 and logits.shape == (1, n, 205)
                    loss_sum = float(F.cross_entropy(logits.reshape(-1, 205).double(),
                                                     y, reduction="sum"))
                assert math.isfinite(loss_sum)
                result["teacher_forwards"] += 1
                rows.append(dict(piece=i, local_token_range=[lo, hi], targets=n,
                                 teacher_nll_sum=loss_sum, teacher_raw_nll=loss_sum/n,
                                 native_baseline_raw_nll=reference["logits_mean"],
                                 delta_raw_nll=loss_sum/n-reference["logits_mean"]))
                total["scored_targets"] += n
                del tok, prior, x, y, pp, neural_bounds, hidden, logits
                save()
            assert total["scored_targets"] == expected_targets
            total.update(complete=True, teacher_nll_sum=math.fsum(r["teacher_nll_sum"] for r in rows))
            total["teacher_raw_nll"] = total["teacher_nll_sum"]/expected_targets
            total["delta_raw_nll"] = total["teacher_raw_nll"]-native["total"]["logits_mean"]
            save()
        result.update(status="passed", adapter_counts=counts)
        save()
        print(json.dumps({k: {f: s[f] for f in
                              ("scored_targets", "teacher_raw_nll", "delta_raw_nll")}
                          for k, s in result["scopes"].items()}), flush=True)
    except Exception as exc:
        result.update(status="cpu_budget" if isinstance(exc, TimeoutError) and str(exc)=="cpu_limit"
                      else "wall_budget" if isinstance(exc, TimeoutError) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
