#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""One fixed residual MLP16 fit on two 768 KiB training prefixes."""
import argparse
import hashlib
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
WIDTH, HIDDEN, CHUNK, BITS = 88, 16, 65536, 8 * 1048576
TRAIN_END = BITS * 3 // 4
MATRIX = WIDTH * HIDDEN
PARAMETERS = MATRIX + 2 * HIDDEN + 1
OPTIONS = {"maxiter": 64, "maxfun": 96, "ftol": 1e-12, "gtol": 1e-9}


def unpack(weights):
    return (weights[:MATRIX].reshape(HIDDEN, WIDTH),
            weights[MATRIX:MATRIX + HIDDEN],
            weights[MATRIX + HIDDEN:-1], weights[-1])


def initial_weights():
    weights = np.zeros(PARAMETERS)
    weights[:MATRIX] = np.random.default_rng(236).normal(
        0, 1 / np.sqrt(WIDTH), size=MATRIX)
    return weights


def objective(weights, batches, count):
    matrix, bias, output, intercept = unpack(weights)
    loss, gradient = 32 * weights.dot(weights), 64 * weights
    for x, offset, y in batches():
        hidden = np.tanh(x @ matrix.T + bias)
        logits = offset + hidden @ output + intercept
        residual = expit(logits) - y
        loss += np.sum(np.logaddexp(0, logits) - y * logits)
        dh = (residual[:, None] * output) * (1 - hidden * hidden)
        gradient[:MATRIX] += (dh.T @ x).ravel()
        gradient[MATRIX:MATRIX + HIDDEN] += dh.sum(axis=0)
        gradient[MATRIX + HIDDEN:-1] += hidden.T @ residual
        gradient[-1] += residual.sum()
    return loss / count, gradient / count


def projected_gradient(weights, gradient):
    projected = gradient.copy()
    projected[(weights <= -16) & (gradient > 0)] = 0
    projected[(weights >= 16) & (gradient < 0)] = 0
    return float(np.max(np.abs(projected)))


def save(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def self_test():
    rng = np.random.default_rng(923)
    x = rng.normal(size=(64, WIDTH)).astype("<f4").astype(np.float64)
    offset, y = rng.normal(size=64), rng.integers(0, 2, size=64)
    weights = rng.normal(size=PARAMETERS) / 10
    batches = lambda: iter([(x, offset, y)])
    _, gradient = objective(weights, batches, len(y))
    indices = [0, MATRIX - 1, MATRIX, MATRIX + HIDDEN - 1,
               MATRIX + HIDDEN, PARAMETERS - 2, PARAMETERS - 1]
    error = 0.0
    for index in indices:
        direction = np.zeros(PARAMETERS); direction[index] = 1
        step = direction * 1e-5
        numeric = (objective(weights + step, batches, len(y))[0] -
                   objective(weights - step, batches, len(y))[0]) / 2e-5
        error = max(error, abs(numeric - gradient[index]))
    for _ in range(3):
        direction = rng.normal(size=PARAMETERS)
        direction /= np.linalg.norm(direction)
        step = direction * 1e-5
        numeric = (objective(weights + step, batches, len(y))[0] -
                   objective(weights - step, batches, len(y))[0]) / 2e-5
        error = max(error, abs(numeric - gradient @ direction))
    assert error < 1e-8, error
    initial = initial_weights()
    matrix, bias, output, intercept = unpack(initial)
    assert np.array_equal(offset + np.tanh(x @ matrix.T + bias) @ output + intercept, offset)
    assert PARAMETERS == 1441 and initial.astype("<f4").nbytes == 5764
    assert TRAIN_END == 6 * 131072 * 8 and TRAIN_END % CHUNK == 0
    initial_loss, _ = objective(initial, batches, len(y))
    fit = minimize(objective, initial, args=(batches, len(y)), jac=True,
                   method="L-BFGS-B", bounds=[(-16, 16)] * PARAMETERS,
                   options={"maxiter": 8, "maxfun": 16})
    assert fit.fun < initial_loss
    print(json.dumps({"stage": "mixer_mlp_fit_gradient_passed", "features": WIDTH,
                      "hidden": HIDDEN, "parameters": PARAMETERS,
                      "gradient_max_error": error, "zero_output_exact": True,
                      "synthetic_initial_objective": initial_loss,
                      "synthetic_fitted_objective": float(fit.fun)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, action="append", metavar=("TRACE", "FEATURES"))
    parser.add_argument("--theta27", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if not args.case or len(args.case) != 2 or args.theta27 is None or args.out is None:
        parser.error("provide two --case TRACE FEATURES, --theta27 and --out")
    theta_bytes = args.theta27.read_bytes()
    if len(theta_bytes) != 27 * 4:
        raise ValueError("theta27 needs exactly 27 little-endian f32 coefficients")
    theta = np.frombuffer(theta_bytes, dtype="<f4").astype(np.float64)
    if not np.isfinite(theta).all() or np.any(np.abs(theta) > 16):
        raise ValueError("invalid frozen theta27 coefficient")
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
        train_bits = 0
        for start in range(0, TRAIN_END, CHUNK):
            r = records[start:start + CHUNK]
            if np.any(r["p"] == 0) or np.any(r["flags"] & ~np.uint16(3)):
                raise ValueError("invalid training trace probability/flags")
            train_bits += int(np.count_nonzero((r["flags"] & 2) == 0))
        cases.append((records, features))
        info.append({"trace": str(trace_path), "features": str(feature_path),
                     "training_rows_end_exclusive": TRAIN_END, "fitted_rows": train_bits})
        count += train_bits
    if not count:
        raise ValueError("no non-override training rows")

    def batches():
        for records, features in cases:
            for start in range(0, TRAIN_END, CHUNK):
                r = records[start:start + CHUNK]
                mask = (r["flags"] & 2) == 0
                p = r["p"][mask].astype(np.float64) / 65536
                x = features[start:start + CHUNK][mask].astype(np.float64)
                # x[:26] is phi27[1:]; theta27 remains frozen throughout.
                offset = np.log(p / (1 - p)) + (theta[0] + x[:, :26] @ theta[1:])
                yield x, offset, (r["flags"][mask] & 1).astype(np.float64)

    started, cpu_started = time.monotonic(), time.process_time()
    iterations = evaluations = 0
    last = {}
    initial_objective = None

    def evaluate(weights):
        nonlocal evaluations, initial_objective, last
        value, gradient = objective(weights, batches, count)
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise ValueError("nonfinite MLP objective/gradient")
        evaluations += 1
        if initial_objective is None:
            initial_objective = value
        last = {"weights": weights.copy(), "objective": value, "gradient": gradient.copy()}
        return value, gradient

    def checkpoint(weights):
        nonlocal iterations
        iterations += 1
        if not np.array_equal(weights, last["weights"]):
            raise RuntimeError("optimizer checkpoint does not match evaluated weights")
        save(args.out / "checkpoint.json", {
            "model": "mlp16", "optimizer": "L-BFGS-B", "options": OPTIONS,
            "iteration": iterations, "evaluations": evaluations,
            "seconds": time.monotonic() - started,
            "cpu_seconds": time.process_time() - cpu_started,
            "weights_f64": weights.tolist(), "objective": last["objective"],
            "gradient_f64": last["gradient"].tolist(),
            "projected_gradient_max_abs": projected_gradient(weights, last["gradient"])})

    baseline, _ = objective(np.zeros(PARAMETERS), batches, count)
    result = minimize(evaluate, initial_weights(), jac=True, method="L-BFGS-B",
                      bounds=[(-16, 16)] * PARAMETERS, options=OPTIONS,
                      callback=checkpoint)
    optimizer = {"method": "L-BFGS-B", "options": OPTIONS,
                 "status": int(result.status), "success": bool(result.success),
                 "message": str(result.message), "iterations": int(result.nit),
                 "evaluations": int(result.nfev), "gradient_evaluations": int(result.njev),
                 "weights_f64": result.x.tolist(), "gradient_f64": result.jac.tolist(),
                 "objective": float(result.fun),
                 "projected_gradient_max_abs": projected_gradient(result.x, result.jac),
                 "seconds": time.monotonic() - started,
                 "cpu_seconds": time.process_time() - cpu_started}
    save(args.out / "optimizer.json", optimizer)
    frozen = result.x.astype("<f4")
    pending = args.out / "weights.pending"
    frozen.tofile(pending)
    pending.replace(args.out / "weights.f32")
    fitted, gradient = objective(frozen.astype(np.float64), batches, count)
    penalty = 32 * frozen.astype(np.float64).dot(frozen.astype(np.float64)) / count
    report = {"schema": "hutter.mixer-mlp-fit.v1", "model": "mlp16",
              "cases": info, "fitted_rows": count, "features": WIDTH,
              "hidden": HIDDEN, "parameters": PARAMETERS, "activation": "tanh",
              "weight_order": "W[16,88] row-major,b[16],v[16],c[1]",
              "theta27": str(args.theta27),
              "theta27_sha256": hashlib.sha256(theta_bytes).hexdigest(),
              "theta27_f32": theta.tolist(), "offset": "base logit plus frozen theta27 dot phi27",
              "holdout": "last 256 KiB of each 1 MiB screen; never read by optimizer",
              "warmup_rows": 0, "seed": 236, "initial_W_std": 1 / np.sqrt(WIDTH),
              "initial_other_parameters": 0, "ridge_hessian": 64,
              "coefficient_bound": 16, "delta_clip": False,
              "max_iterations": 64, "max_evaluations": 96,
              "iterations": int(result.nit), "evaluations": int(result.nfev),
              "total_objective_evaluations": evaluations + 2,
              "success": bool(result.success), "message": str(result.message),
              "seconds": time.monotonic() - started,
              "cpu_seconds": time.process_time() - cpu_started,
              "weights_bytes": int(frozen.nbytes), "weights_f32": frozen.tolist(),
              "zero_objective": baseline, "initial_objective": initial_objective,
              "frozen_objective": fitted, "frozen_train_nll": fitted - penalty,
              "frozen_ridge_penalty_per_row": penalty,
              "projected_gradient_max_abs": projected_gradient(frozen, gradient),
              "measurement": "Fixed-budget nonconvex training fit; zero objective includes theta27; exact integer tail replay decides screening."}
    save(args.out / "fit.json", report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
