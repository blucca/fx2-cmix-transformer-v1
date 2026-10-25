#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""One fixed 27/73-weight fit on the first 768 KiB of two screening traces."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

TRACE = np.dtype([("p", "<u2"), ("flags", "<u2"),
                  ("neural", "<f4"), ("fxcm", "<f4"), ("mixer", "<f4")])
WIDTH, CHUNK, BITS = 73, 65536, 8 * 1048576
DIMENSIONS = {"logit27": 27, "support73": 73}


def objective(weights, batches, count):
    # Same convex offset-NLL/ridge as fit-history.py. No delta clipping.
    loss, gradient = 32 * weights.dot(weights), 64 * weights
    for x, z, y in batches():
        logits = z + x @ weights
        loss += np.sum(np.logaddexp(0, logits) - y * logits)
        gradient = gradient + x.T @ (expit(logits) - y)
    return loss / count, gradient / count


def save(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def self_test():
    rng = np.random.default_rng(923)
    x, z = rng.normal(size=(64, WIDTH)), rng.normal(size=64)
    y, w = rng.integers(0, 2, size=64), rng.normal(size=WIDTH) / 10
    batches = lambda: iter([(x, z, y)])
    _, analytic = objective(w, batches, len(y))
    for j in range(WIDTH):
        delta = np.zeros(WIDTH); delta[j] = 1e-5
        numeric = (objective(w + delta, batches, len(y))[0] -
                   objective(w - delta, batches, len(y))[0]) / 2e-5
        assert abs(numeric - analytic[j]) < 1e-8
    assert 6 * 131072 == 3 * 1048576 // 4
    # Synthetic-only rate measurement; independent of any research labels.
    n = 262144
    x = rng.normal(size=(n, WIDTH)).astype("<f4")
    z, y = rng.normal(size=n), rng.integers(0, 2, size=n)
    def chunks():
        for start in range(0, n, CHUNK):
            yield x[start:start + CHUNK].astype(np.float64), z[start:start + CHUNK], y[start:start + CHUNK]
    started = time.monotonic()
    loss, _ = objective(np.zeros(WIDTH), chunks, n)
    elapsed = time.monotonic() - started
    assert np.isfinite(loss)
    print(json.dumps({"stage": "mixer_fit_gradient_passed", "dimensions": WIDTH,
                      "synthetic_rows": n, "objective_seconds": elapsed,
                      "estimated_screen_seconds_per_evaluation": elapsed * (2 * BITS * .75 / n)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, action="append", metavar=("TRACE", "FEATURES"))
    parser.add_argument("--model", choices=DIMENSIONS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if not args.case or len(args.case) != 2 or not args.model or args.out is None:
        parser.error("provide two --case TRACE FEATURES, --model and --out")
    dimensions = DIMENSIONS[args.model]
    args.out.mkdir(parents=True, exist_ok=False)
    cases, info, count = [], [], 0
    for trace_path, feature_path in args.case:
        trace_path, feature_path = Path(trace_path), Path(feature_path)
        if trace_path.stat().st_size != BITS * TRACE.itemsize:
            raise ValueError("each fixed screen must contain exactly 1 MiB input")
        if feature_path.stat().st_size != BITS * WIDTH * 4:
            raise ValueError("feature dimensions disagree with the fixed trace")
        records = np.memmap(trace_path, dtype=TRACE, mode="r")
        features = np.memmap(feature_path, dtype="<f4", mode="r", shape=(BITS, WIDTH))
        train_end = BITS * 3 // 4
        train_bits = sum(int(np.count_nonzero((records[start:start + CHUNK]["flags"] & 2) == 0))
                         for start in range(0, train_end, CHUNK))
        cases.append((records, features))
        info.append({"trace": str(trace_path), "features": str(feature_path),
                     "training_rows_end_exclusive": train_end, "fitted_rows": train_bits})
        count += train_bits
    if not count:
        raise ValueError("no non-override training rows")

    def batches():
        for records, features in cases:
            for start in range(0, BITS * 3 // 4, CHUNK):
                r = records[start:start + CHUNK]
                mask = (r["flags"] & 2) == 0
                p = r["p"][mask].astype(np.float64) / 65536
                x = features[start:start + CHUNK, :dimensions][mask].astype(np.float64)
                yield x, np.log(p / (1 - p)), (r["flags"][mask] & 1).astype(np.float64)

    started, iterations = time.monotonic(), 0
    def checkpoint(weights):
        nonlocal iterations
        iterations += 1
        save(args.out / "checkpoint.json", {"model": args.model, "iteration": iterations,
             "seconds": time.monotonic() - started, "weights_f64": weights.tolist()})

    zero = np.zeros(dimensions)
    baseline, _ = objective(zero, batches, count)
    result = minimize(objective, zero, args=(batches, count), jac=True,
                      method="L-BFGS-B", bounds=[(-16, 16)] * dimensions,
                      options={"maxiter": 64, "maxfun": 96, "ftol": 1e-12, "gtol": 1e-9},
                      callback=checkpoint)
    frozen = result.x.astype("<f4")
    frozen.tofile(args.out / "weights.f32")
    fitted, gradient = objective(frozen.astype(np.float64), batches, count)
    projected = gradient.copy()
    projected[(frozen <= -16) & (gradient > 0)] = 0
    projected[(frozen >= 16) & (gradient < 0)] = 0
    report = {"schema": "hutter.mixer-support-fit.v1", "model": args.model,
              "cases": info, "fitted_rows": count, "dimensions": dimensions,
              "holdout": "last 256 KiB of each 1 MiB screen; never read by optimizer",
              "warmup_rows": 0, "ridge_hessian": 64, "coefficient_bound": 16,
              "delta_clip": False, "max_iterations": 64, "max_evaluations": 96,
              "iterations": int(result.nit), "evaluations": int(result.nfev),
              "success": bool(result.success), "message": str(result.message),
              "seconds": time.monotonic() - started, "weights_bytes": int(frozen.nbytes),
              "weights_f32": frozen.tolist(), "zero_objective": baseline,
              "frozen_objective": fitted,
              "projected_gradient_max_abs": float(np.max(np.abs(projected))),
              "measurement": "Bounded regularized training fit; exact integer tail replay decides screening."}
    save(args.out / "fit.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
