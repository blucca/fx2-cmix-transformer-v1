#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fit one frozen 18-weight offset head on causal features from replay.cpp."""

import argparse
import json
import os
from pathlib import Path
import time

# A streamable small fit: no full feature matrix or threaded BLAS allocation.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

TRACE = np.dtype([("p", "<u2"), ("flags", "<u2"),
                  ("neural", "<f4"), ("fxcm", "<f4"), ("mixer", "<f4")])
WIDTH, CHUNK, BLOCK = 18, 65536, 8 * 131072


def objective(weights, batches, count):
    loss, gradient = 32 * weights.dot(weights), 64 * weights
    for x, z, y in batches():
        logits = z + x @ weights
        loss += np.sum(np.logaddexp(0, logits) - y * logits)
        gradient = gradient + x.T @ (expit(logits) - y)
    return loss / count, gradient / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, action="append", metavar=("TRACE", "FEATURES"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        rng = np.random.default_rng(923)
        x, z = rng.normal(size=(64, WIDTH)), rng.normal(size=64)
        y = rng.integers(0, 2, size=64)
        w = rng.normal(size=WIDTH) / 10
        batches = lambda: iter([(x, z, y)])
        _, analytic = objective(w, batches, len(y))
        for j in range(WIDTH):
            delta = np.zeros(WIDTH); delta[j] = 1e-5
            numerical = (objective(w + delta, batches, len(y))[0] -
                         objective(w - delta, batches, len(y))[0]) / 2e-5
            assert abs(numerical - analytic[j]) < 1e-8
        print(json.dumps({"stage": "history_fit_gradient_passed", "dimensions": WIDTH}))
        return
    if not args.case or args.out is None:
        parser.error("provide --case TRACE FEATURES (repeat for each slice) and --out")
    args.out.mkdir(parents=True, exist_ok=False)
    cases, info = [], []
    count = 0
    for trace_path, feature_path in args.case:
        trace_path, feature_path = Path(trace_path), Path(feature_path)
        size = trace_path.stat().st_size
        if not size or size % (TRACE.itemsize * 8):
            raise ValueError("trace must contain whole input bytes")
        n = size // TRACE.itemsize
        if feature_path.stat().st_size != n * WIDTH * 4:
            raise ValueError("feature dimensions disagree with trace")
        records = np.memmap(trace_path, dtype=TRACE, mode="r")
        features = np.memmap(feature_path, dtype="<f4", mode="r", shape=(n, WIDTH))
        train_bits = 0
        for start in range(0, n, CHUNK):
            block = (start // BLOCK) % 8
            if block not in (3, 7):
                train_bits += np.count_nonzero((records[start:start + CHUNK]["flags"] & 2) == 0)
        cases.append((records, features))
        info.append({"trace": str(trace_path), "features": str(feature_path),
                     "bits": n, "fitted_bits": int(train_bits)})
        count += int(train_bits)
    if not count:
        raise ValueError("no non-override training bits")

    def batches():
        for records, features in cases:
            for start in range(0, len(records), CHUNK):
                if (start // BLOCK) % 8 in (3, 7):
                    continue
                r = records[start:start + CHUNK]
                mask = (r["flags"] & 2) == 0
                p = r["p"][mask].astype(np.float64) / 65536
                x = features[start:start + CHUNK][mask].astype(np.float64)
                yield x, np.log(p / (1 - p)), (r["flags"][mask] & 1).astype(np.float64)

    started, iterations = time.time(), 0

    def checkpoint(weights):
        nonlocal iterations
        iterations += 1
        pending = args.out / "fit-checkpoint.pending"
        pending.write_text(json.dumps({
            "iteration": iterations, "seconds": time.time() - started,
            "weights_f64": weights.tolist()}) + "\n")
        pending.replace(args.out / "fit-checkpoint.json")

    zero = np.zeros(WIDTH)
    baseline, _ = objective(zero, batches, count)
    result = minimize(objective, zero, args=(batches, count), jac=True,
                      method="L-BFGS-B", bounds=[(-2, 2)] * WIDTH,
                      options={"maxiter": 64, "ftol": 1e-12, "gtol": 1e-9},
                      callback=checkpoint)
    frozen = result.x.astype("<f4")
    frozen.tofile(args.out / "weights.f32")
    frozen_objective, gradient = objective(frozen.astype(np.float64), batches, count)
    report = {
        "schema": "history18-ema-f32-v1", "cases": info,
        "holdout_rule": "128 KiB input-byte blocks 3 and 7 of each eight; history keeps running",
        "fitted_bits": count, "ridge_hessian": 64, "coefficient_bound": 2,
        "max_iterations": 64, "iterations": int(result.nit),
        "evaluations": int(result.nfev), "success": bool(result.success),
        "message": str(result.message), "seconds": time.time() - started,
        "weights_bytes": int(frozen.nbytes), "weights_f32": frozen.tolist(),
        "zero_objective": baseline, "frozen_objective": frozen_objective,
        "frozen_gradient_max_abs": float(np.max(np.abs(gradient))),
        "measurement": "Regularized training NLL only. Use strict integer replay for coded bytes."
    }
    (args.out / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
