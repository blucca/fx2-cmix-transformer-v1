#!/usr/bin/env python3
"""Read-only CE/KL/H attribution at original q and one immutable z snapshot.

Uses at most four complete 1025-2048-target pieces and the frozen CPU method.
These local gradients diagnose the trajectory; native/full mixed scores remain
the compression decision. Gradient-only: optimizer updates and exports = 0;
the training state and immutable inputs stay fixed.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "kernel", "coordinates", "cases", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--cpu-seconds", type=int, default=300)
    args = ap.parse_args()
    assert 30 <= args.cpu_seconds <= 900
    cases = json.loads(args.cases.read_text())
    assert 1 <= len(cases) <= 4
    for c in cases:
        assert 1025 <= c["end"] - c["start"] - 1 <= 2048
        assert c["role"] in ("training", "held_out", "independent_mature")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (3 << 30, 3 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (args.cpu_seconds - 2, args.cpu_seconds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.process_time()
    result = dict(status="starting", optimizer_updates=0, coordinates=str(args.coordinates),
                  checkpoint=str(args.checkpoint), cases=cases, points=[],
                  cpu_limit_seconds=args.cpu_seconds, address_space_limit_gib=3, threads=1)

    def save(stage, **values):
        result.update(stage=stage, cpu_seconds=time.process_time() - started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **values)
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(args.output)

    def cpu_limit(signum, frame):
        save("cpu_limit", status="stopped")
        raise SystemExit(124)

    signal.signal(signal.SIGXCPU, cpu_limit)
    save("importing")
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        sys.path.insert(0, str(args.source_dir))
        spec = importlib.util.spec_from_file_location(
            "frozen_integer_qat", args.source_dir / "integer-coordinate-qat.py")
        qat = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qat)
        from cpu_model import load_model
        model, _ = load_model(args.checkpoint, args.kernel)
        z, scales, q0, original = qat.install_coordinates(model)
        snapshot = torch.load(args.coordinates, map_location="cpu", weights_only=True)
        assert snapshot.keys() == z.keys()
        for key, value in snapshot.items():
            assert value.shape == z[key].shape and bool(torch.isfinite(value).all())
            assert bool((value >= -7).all() & (value <= 7).all())
        names, parameters = list(z), tuple(z.values())
        states = {"original": q0, "snapshot": snapshot}

        def copy_point(point):
            with torch.no_grad():
                for key, value in z.items():
                    value.copy_(states[point][key])

        # Per-matrix processing limits the soft-assignment allocation.
        hg, hn = {}, {}
        for point in states:
            copy_point(point)
            grads, total = [], 0.
            for value in z.values():
                h = qat.entropy(value)
                total += float(h.detach())
                grads.append(torch.autograd.grad(2 * h / qat.FULL_TARGETS, value)[0])
            hg[point], hn[point] = grads, total
        train_sum, train_n = {}, {}

        def norm(values):
            return math.sqrt(sum(float(v.double().square().sum()) for v in values))

        def cosine(a, b):
            na, nb = norm(a), norm(b)
            return sum(float((x.double() * y.double()).sum()) for x, y in zip(a, b)) / (na * nb) if na * nb else None

        for case in cases:
            lo, hi = case["start"], case["end"]
            n = hi - lo - 1
            tok = np.fromfile(case["tokens"], dtype=np.uint8)
            assert hi <= len(tok) and Path(case["priors"]).stat().st_size == len(tok) * 410
            x = torch.from_numpy(tok[lo:hi - 1].astype(np.int64)).unsqueeze(0)
            y = torch.from_numpy(tok[lo + 1:hi].astype(np.int64)).unsqueeze(0)
            p = torch.from_numpy(np.fromfile(case["priors"], dtype="<f2", count=n * 205,
                                            offset=lo * 410).reshape(1, n, 205))
            bounds = torch.tensor([[0, n]], dtype=torch.int32)
            assert bool(torch.isfinite(p).all()) and int(x.max()) < 205 and int(y.max()) < 205
            copy_point("original")
            with torch.no_grad():
                teacher = model.compute_logits(input_tokens=x, priors=p,
                                               cumulative_sequence_lengths=bounds)
            if case.get("teacher_cache"):
                cached = torch.from_numpy(np.load(case["teacher_cache"], allow_pickle=False))
                assert torch.equal(teacher, cached), "original teacher differs from frozen training cache"
                del cached
            teacher_ce = float(F.cross_entropy(teacher.view(-1, 205), y.flatten()))
            for point in states:
                copy_point(point)
                save("differentiating", current_case=case["name"], current_point=point)
                logits = model.compute_logits(input_tokens=x, priors=p,
                                              cumulative_sequence_lengths=bounds)
                assert bool(torch.isfinite(logits).all())
                ce = F.cross_entropy(logits.view(-1, 205), y.flatten(), reduction="sum") / n
                kl = F.kl_div(logits.log_softmax(-1), teacher.log_softmax(-1),
                              reduction="sum", log_target=True) / n
                if point == "original":
                    assert torch.equal(logits.detach(), teacher) and float(kl.detach()) == 0
                gc = torch.autograd.grad(ce, parameters, retain_graph=True)
                gk = torch.autograd.grad(kl, parameters)
                hgrads = hg[point]
                tensor_rows, tiny_rows = [], []
                changed_stats = dict(count=0, ce_sq=0., kl_sq=0., h_sq=0.)
                task_small = 0
                for key, a, b, h in zip(names, gc, gk, hgrads):
                    assert bool(torch.isfinite(a).all() & torch.isfinite(b).all() & torch.isfinite(h).all())
                    dq = snapshot[key].round().clamp(-7, 7) - q0[key].float()
                    dz = snapshot[key] - q0[key].float()
                    mask = dq != 0
                    task_small += int(((a + b).abs() < 1e-8).sum())
                    changed_stats["count"] += int(mask.sum())
                    for label, grad in (("ce", a), ("kl", b), ("h", h)):
                        changed_stats[label + "_sq"] += float(grad[mask].double().square().sum())
                    row = dict(weight=key, changed_symbols=int(mask.sum()),
                               ce_norm=norm([a]), kl_norm=norm([b]), h_norm=norm([h]),
                               ce_kl_cosine=cosine([a], [b]))
                    for label, grad in (("ce", a), ("kl", b), ("h", h)):
                        row[label + "_dot_delta_z"] = float((grad.double() * dz.double()).sum())
                        row[label + "_dot_delta_q"] = float((grad.double() * dq.double()).sum())
                    tensor_rows.append(row)
                    for r in torch.where(scales[key].abs() < 1e-10)[0].tolist():
                        tiny_rows.append(dict(weight=key, row=r, signed_bf16_scale=float(scales[key][r]),
                                              changed_symbols=int(mask[r].sum()), ce_norm=norm([a[r]]),
                                              kl_norm=norm([b[r]]), h_norm=norm([h[r]])))
                for label in ("ce", "kl", "h"):
                    changed_stats[label + "_norm"] = math.sqrt(changed_stats.pop(label + "_sq"))
                row = dict(case=case["name"], role=case["role"], point=point, targets=n,
                           ce_mean=float(ce.detach()), delta_ce=float(ce.detach()) - teacher_ce,
                           teacher_kl_mean=float(kl.detach()), soft_entropy_nats=hn[point],
                           ce_norm=norm(gc), kl_norm=norm(gk), h_norm=norm(hgrads),
                           ce_kl_cosine=cosine(gc, gk), ce_h_cosine=cosine(gc, hgrads),
                           kl_h_cosine=cosine(gk, hgrads), combined_norm=norm([a + b + h for a, b, h in zip(gc, gk, hgrads)]),
                           task_abs_below_adam_epsilon_fraction=task_small / sum(v.numel() for v in parameters),
                           changed_symbol_subset=changed_stats, tiny_rows=tiny_rows,
                           highest_ce_tensors=sorted(tensor_rows, key=lambda r: -r["ce_norm"])[:8],
                           highest_changed_tensors=sorted(tensor_rows, key=lambda r: -r["changed_symbols"])[:8])
                for label in ("ce", "kl", "h"):
                    for delta in ("z", "q"):
                        field = label + "_dot_delta_" + delta
                        row[field] = sum(t[field] for t in tensor_rows)
                if case["role"] == "training":
                    if point not in train_sum:
                        train_sum[point] = [g.detach().clone() * n for g in gc]
                        train_n[point] = n
                    else:
                        for total, grad in zip(train_sum[point], gc):
                            total.add_(grad, alpha=n)
                        train_n[point] += n
                elif point in train_sum:
                    row["ce_training_piece_cosine"] = cosine(gc, train_sum[point])
                    row["training_diagnostic_targets"] = train_n[point]
                result["points"].append(row)
                save("point_complete")
                print(json.dumps({k: row[k] for k in ("case", "point", "ce_mean", "teacher_kl_mean", "ce_norm", "kl_norm", "h_norm", "ce_kl_cosine")}), flush=True)
                del logits, ce, kl, gc, gk
            del tok, x, y, p, bounds, teacher
        for key, value in model.named_parameters():
            assert value.grad is None and torch.equal(value, original[key]), key
        for key, value in z.items():
            assert torch.equal(value.detach(), snapshot[key]), key
        save("complete", status="passed", frozen_original_tensors=len(original),
             unchanged_snapshot_coordinates=True, gradient_tensors=len(parameters))
    except Exception as e:
        save("failed", status="failed", error=f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
