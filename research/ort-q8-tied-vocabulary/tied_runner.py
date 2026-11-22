#!/usr/bin/env python3
"""Fixed shared-vocabulary study; reuse the earlier complete decode oracle.

One fresh session per worker. Previous FP32-vocabulary graphs are performance
controls, not same-weight numerical references for the new vocabulary.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"


def setup(args):
    sys.path[:0] = [str(args.base_root), str(args.base_root.parent)]
    common = importlib.import_module("native_runner")
    original_session = common.session

    def session(root, mode):
        if mode == "old_native":
            return original_session(args.base_root, "native")
        return original_session(root, mode)

    common.session = session
    return common


def compare(common, root, label):
    names = ("old-before", "dense-before", "native", "q8_acc0", "dense-after", "old-after")
    reports = {n: json.loads((root / "runs" / label / n / "result.json").read_text()) for n in names}
    assert all(r["status"] == "completed" and len(r["rows"]) == 3 for r in reports.values())
    numerical_names = ("dense-before", "native", "q8_acc0", "dense-after")
    precision = {n: reports[n]["all_outputs_passed"] for n in numerical_names}
    rows = []
    for i, past in enumerate(common.PAST):
        old = min(reports[n]["rows"][i]["p50_ms"] for n in ("old-before", "old-after"))
        dense = min(reports[n]["rows"][i]["p50_ms"] for n in ("dense-before", "dense-after"))
        new = reports["native"]["rows"][i]["p50_ms"]
        rows.append({"past": past, "previous_native_conservative_p50_ms": old,
                     "new_native_p50_ms": new, "time_reduction_vs_previous_native": 1-new/old,
                     "same_weight_dense_conservative_p50_ms": dense,
                     "time_reduction_vs_same_weight_dense": 1-new/dense,
                     "same_weight_stock_q8_p50_ms": reports["q8_acc0"]["rows"][i]["p50_ms"]})
    old_rss = min(reports[n]["peak_rss_kib"] for n in ("old-before", "old-after"))
    new_rss = reports["native"]["peak_rss_kib"]
    features = reports["native"]["host"]["features"]
    target = features["avx2"] and features["fma"] and not any(features[k] for k in ("avx512f", "avx512_vnni", "avx_vnni"))
    numerical = all(precision.values())
    performance = all(r["time_reduction_vs_previous_native"] >= .15 for r in rows)
    memory = 1-new_rss/old_rss >= .15
    result = {"host": reports["native"]["host"], "rows": rows, "precision_by_same_weight_mode": precision,
              "numerical_pass": numerical, "previous_native_conservative_peak_rss_kib": old_rss,
              "new_native_peak_rss_kib": new_rss, "rss_reduction": 1-new_rss/old_rss,
              "performance_pass": performance, "memory_pass": memory, "target_avx2_no_vnni": target,
              "target_compute_gate_passed": bool(target and numerical and performance and memory),
              "quality_gate": "separate paired fixed-text probe", "full_reports": list(reports)}
    common.dump(root / "runs" / label / "comparison.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--base-root", type=Path, required=True)
    p.add_argument("--action", choices=("prepare", "worker", "all", "compare"), default="all")
    p.add_argument("--label", default="remote")
    p.add_argument("--mode", default="native")
    p.add_argument("--run-name", default="native")
    p.add_argument("--reference", default="dense-before")
    p.add_argument("--max-as-mib", type=int, default=4096)
    args = p.parse_args()
    args.root, args.base_root = args.root.resolve(), args.base_root.resolve()
    common = setup(args)
    common.self_test()
    folder = args.root / "runs" / args.label
    folder.mkdir(parents=True, exist_ok=True)
    if args.action == "compare":
        compare(common, args.root, args.label)
        return
    if args.action in ("worker", "prepare"):
        resource.setrlimit(resource.RLIMIT_AS, (args.max_as_mib*2**20, args.max_as_mib*2**20))
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        if args.action == "prepare":
            common.prepare(args.root)
        else:
            common.run(args.root, args.label, args.mode, args.run_name, args.reference)
            path = folder / args.run_name / "result.json"
            report = json.loads(path.read_text())
            if args.mode == "old_native":
                report["comparison_scope"] = "same complete inputs; previous FP32 vocabulary; performance-only control"
            else:
                report["comparison_scope"] = "same full Q8 FFN and tied Q8 vocabulary; all logits/KV and appended slice"
            common.dump(path, report)
        return
    stages = [("prepare", "dense", "prepare", "self"),
              ("worker", "old_native", "old-before", "self"),
              ("worker", "dense", "dense-before", "self"),
              ("worker", "native", "native", "dense-before"),
              ("worker", "q8_acc0", "q8_acc0", "dense-before"),
              ("worker", "dense", "dense-after", "dense-before"),
              ("worker", "old_native", "old-after", "self")]
    for action, mode, name, ref in stages:
        common.dump(folder / "state.json", {"stage": name, "status": "running"})
        command = [sys.executable, str(Path(__file__).resolve()), "--root", str(args.root),
                   "--base-root", str(args.base_root), "--action", action, "--label", args.label,
                   "--mode", mode, "--run-name", name, "--reference", ref,
                   "--max-as-mib", str(args.max_as_mib)]
        with (folder / (name + ".log")).open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            common.dump(folder / "state.json", {"stage": name, "status": "failed", "returncode": completed.returncode})
            raise SystemExit(completed.returncode)
        print("STAGE_DONE", name, flush=True)
    result = compare(common, args.root, args.label)
    common.dump(folder / "state.json", {"stage": "complete", "status": "completed",
                                       "numerical_pass": result["numerical_pass"]})


if __name__ == "__main__":
    main()
