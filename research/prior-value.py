#!/usr/bin/env python3
"""One streaming prior-only scan of the original causal-v1 token-index inputs.

Rows hold 205 probabilities, encoded as IEEE binary16, little endian. Within
each original piece, row i scores token i+1; the piece-last row has no target.
All reported definitions share those targets and the original NN piece losses.
"""
import os
import resource
import signal
import time

START_WALL = time.monotonic()
START_CPU = time.process_time()
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
os.nice(10)
_soft, _hard = resource.getrlimit(resource.RLIMIT_CPU)
_cap = 120 if _hard == resource.RLIM_INFINITY else min(120, _hard)
resource.setrlimit(resource.RLIMIT_CPU, (_cap, _cap))


def deadline(_signal, _frame):
    raise TimeoutError("120-second wall budget reached")


signal.signal(signal.SIGALRM, deadline)
signal.alarm(120)

import argparse
import io
import json
import math
from pathlib import Path
import struct

import numpy as np

V = 205
CHUNK = 4096
FLOOR = float(np.float32(1e-6))
METRICS = ("f16_target_raw", "f16_normalized", "f16_floor1e6_normalized",
           "predictor_f16_tail_floor1e6_normalized")


def load_json(path):
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path), bytes=len(raw))


def load_bounds(path, length):
    raw = path.read_bytes()
    assert len(raw) >= 8 and len(raw) % 4 == 0, "bounds format failure"
    b = list(struct.unpack("<" + "I" * (len(raw) // 4), raw))
    assert b[0] == 0 and b[-1] == length, "bounds endpoint failure"
    assert all(0 < z - a <= 131072 for a, z in zip(b, b[1:])), "piece size failure"
    return b, dict(path=str(path), bytes=len(raw))


def loss(count, zero, finite_sum):
    return dict(targets=count, zero_targets=zero, finite_sum_nats=finite_sum,
                sum_nats=finite_sum if zero == 0 else None,
                mean_nats=finite_sum / count if zero == 0 and count else None)


def pair(priors, nn):
    if nn is None:
        return dict(status="absent_in_original_receipt")
    result = {}
    for name, p in priors.items():
        if p["sum_nats"] is None:
            result[name] = dict(status="zero_target_probability")
            continue
        result[name] = dict(
            raw_logits_gain_sum_nats=p["sum_nats"] - nn["logits_sum"],
            raw_logits_gain_mean_nats=(p["sum_nats"] - nn["logits_sum"]) / p["targets"],
            nn_probs_gain_sum_nats=p["sum_nats"] - nn["probs_sum"],
            nn_probs_gain_mean_nats=(p["sum_nats"] - nn["probs_sum"]) / p["targets"])
    return result


def scan(stream, tokens, bounds, identities, nn_pieces=None):
    """Read every f16 row exactly once, including each unused last row."""
    assert len(identities) == len(bounds) - 1
    assert int(tokens.max()) < V, "token-index range failure"
    if nn_pieces is not None:
        assert len(nn_pieces) == len(identities), "NN piece count failure"
    pieces = []
    for index, (begin, end) in enumerate(zip(bounds, bounds[1:])):
        sums = {name: [] for name in METRICS}
        count = zero = 0
        mass_sums = []
        mass_min, mass_max = math.inf, -math.inf
        rounded_zero_values = floored_values = legacy_tail_subnormal_values = 0
        for lo in range(begin, end, CHUNK):
            hi = min(end, lo + CHUNK)
            raw = stream.read((hi - lo) * V * 2)
            assert len(raw) == (hi - lo) * V * 2, "short f16 input"
            h = np.frombuffer(raw, dtype="<f2").reshape(hi - lo, V)
            p = h.astype(np.float64)
            assert float(p.min()) >= 0 and float(p.max()) <= 1, "prior probability range failure"
            n = hi - lo - (hi == end)
            if n == 0:
                continue
            p = p[:n]
            h = h[:n]
            row = np.arange(n)
            target = tokens[lo + 1:lo + n + 1]
            probability = p[row, target]
            positive = probability > 0
            count += n
            zero += n - int(np.count_nonzero(positive))
            rounded_zero_values += int(np.count_nonzero(p == 0))
            floored_values += int(np.count_nonzero(p < FLOOR))
            mass = p.sum(axis=1, dtype=np.float64)
            assert float(mass.min()) > 0, "zero row mass"
            mass_sums.append(float(mass.sum(dtype=np.float64)))
            mass_min = min(mass_min, float(mass.min()))
            mass_max = max(mass_max, float(mass.max()))
            raw_nll = np.zeros(n, dtype=np.float64)
            np.log(probability, where=positive, out=raw_nll)
            raw_nll *= -1
            sums[METRICS[0]].append(float(raw_nll.sum(dtype=np.float64)))
            sums[METRICS[1]].append(float(
                (raw_nll[positive] + np.log(mass[positive])).sum(dtype=np.float64)))

            # Predictor's file output guard is a float32 >=1e-6 test. Ratios
            # downstream define the normalized distribution represented here.
            np.maximum(p, FLOOR, out=p)
            sums[METRICS[2]].append(float((
                np.log(p.sum(axis=1, dtype=np.float64)) - np.log(p[row, target])
            ).sum(dtype=np.float64)))

            # F16C decodes columns 0..199 exactly. Predictor's scalar tail
            # decoder halves subnormal values in columns 200..204, then guards.
            tail = h[:, 200:205].astype(np.float64)
            subnormal = (tail > 0) & (tail < 2.0 ** -14)
            legacy_tail_subnormal_values += int(np.count_nonzero(subnormal))
            tail[subnormal] *= .5
            p[:, 200:205] = np.maximum(tail, FLOOR)
            sums[METRICS[3]].append(float((
                np.log(p.sum(axis=1, dtype=np.float64)) - np.log(p[row, target])
            ).sum(dtype=np.float64)))
        assert count == end - begin - 1 and count > 0, "piece target mask failure"
        priors = {name: loss(count, zero if i < 2 else 0, math.fsum(sums[name]))
                  for i, name in enumerate(METRICS)}
        nn = nn_pieces[index]["loss"] if nn_pieces is not None else None
        if nn is not None:
            assert nn["count"] == count, "paired NN target count failure"
        pieces.append(dict(
            **identities[index], piece_index=index, targets=count,
            local_token_range=[begin, end], local_target_range=[begin + 1, end],
            local_prior_row_range=[begin, end - 1], prior=priors,
            nn_baseline=nn, nn_gain_prior_minus_nn=pair(priors, nn),
            quantization=dict(f16_row_mass_sum=math.fsum(mass_sums),
                              f16_row_mass_min=mass_min, f16_row_mass_max=mass_max,
                              f16_rounded_zero_values=rounded_zero_values,
                              f16_below_guard_values=floored_values,
                              predictor_tail_subnormal_values=legacy_tail_subnormal_values)))
    assert stream.read(1) == b"", "extra f16 input bytes"
    return pieces


def aggregate(pieces):
    count = sum(p["targets"] for p in pieces)
    assert count > 0, "empty scope"
    priors = {name: loss(count, sum(p["prior"][name]["zero_targets"] for p in pieces),
                        math.fsum(p["prior"][name]["finite_sum_nats"] for p in pieces))
              for name in METRICS}
    nns = [p["nn_baseline"] for p in pieces]
    nn = None
    if all(n is not None for n in nns):
        nn = dict(count=count,
                  logits_sum=math.fsum(n["logits_sum"] for n in nns),
                  probs_sum=math.fsum(n["probs_sum"] for n in nns))
        nn.update(logits_mean=nn["logits_sum"] / count,
                  probs_mean=nn["probs_sum"] / count)
    q = dict(f16_row_mass_mean=math.fsum(p["quantization"]["f16_row_mass_sum"] for p in pieces) / count,
             f16_row_mass_min=min(p["quantization"]["f16_row_mass_min"] for p in pieces),
             f16_row_mass_max=max(p["quantization"]["f16_row_mass_max"] for p in pieces))
    for key in ("f16_rounded_zero_values", "f16_below_guard_values", "predictor_tail_subnormal_values"):
        q[key] = sum(p["quantization"][key] for p in pieces)
    return dict(pieces=len(pieces), article_clusters=len({p["article"] for p in pieces}),
                targets=count, prior=priors, nn_baseline=nn,
                nn_gain_prior_minus_nn=pair(priors, nn), quantization=q)


def fixture():
    # One six-row fixture exercises alignment, independent piece boundaries,
    # binary16 subnormal tail decoding, zero targets, row mass and row guards.
    h = np.full((6, V), 1 / 256, dtype="<f2")
    tokens = np.array([7, 2, 200, 8, 204, 3], dtype=np.uint8)
    bounds = [0, 3, 6]
    h[0, 2] = .5
    h[1, 200] = 2.0 ** -16
    h[3, 204] = 0
    h[4, 3] = .25
    h[2] = .75
    h[5] = .125
    identities = [dict(article=0, source_token_range=[100, 103], role="validation"),
                  dict(article=1, source_token_range=[200, 203], role="validation")]
    nn_pieces = [{"loss": dict(count=2, logits_sum=3., probs_sum=3.1,
                               logits_mean=1.5, probs_mean=1.55)} for _ in identities]
    pieces = scan(io.BytesIO(h.tobytes()), tokens, bounds, identities, nn_pieces)
    total = aggregate(pieces)
    assert total["targets"] == 4 and total["prior"][METRICS[0]]["zero_targets"] == 1
    assert total["prior"][METRICS[0]]["mean_nats"] is None
    manual = {name: [] for name in METRICS}
    for i in (0, 1, 3, 4):
        row = [float(v) for v in h[i]]
        t = int(tokens[i + 1])
        if row[t] > 0:
            manual[METRICS[0]].append(-math.log(row[t]))
            manual[METRICS[1]].append(math.log(math.fsum(row)) - math.log(row[t]))
        guarded = [max(x, FLOOR) for x in row]
        manual[METRICS[2]].append(math.log(math.fsum(guarded)) - math.log(guarded[t]))
        for j in range(200, 205):
            x = row[j]
            guarded[j] = max(x * .5 if 0 < x < 2.0 ** -14 else x, FLOOR)
        manual[METRICS[3]].append(math.log(math.fsum(guarded)) - math.log(guarded[t]))
    for name in METRICS:
        assert abs(total["prior"][name]["finite_sum_nats"] - math.fsum(manual[name])) < 1e-12
    assert pieces[0]["nn_baseline"]["count"] == 2
    assert total["nn_gain_prior_minus_nn"][METRICS[2]]["raw_logits_gain_sum_nats"] == total["prior"][METRICS[2]]["sum_nats"] - 6
    return dict(status="passed", rows=6, pieces=2, targets=4,
                scalar_fixture_tolerance_nats=1e-12)


def main(args):
    global CHUNK
    CHUNK = args.block_rows
    resource.setrlimit(resource.RLIMIT_AS, (args.as_mib << 20, args.as_mib << 20))
    if args.self_test:
        print(json.dumps(dict(fixture=fixture(), resources=resources()), allow_nan=False))
        return
    assert args.job is not None and args.output is not None, "job and output required"
    assert args.output.parent.is_dir() and not args.output.exists(), "fresh existing output directory required"
    job = args.job
    plan, plan_id = load_json(job / "data/plan.json")
    manifest, manifest_id = load_json(job / "data-manifest.json")
    baseline, baseline_id = load_json(job / "baseline.json")
    assert plan["method"] == "causal-whole-piece-v1" and plan["seed"] == 414
    assert baseline["status"] == "passed", "original baseline status failure"
    originals = {name: value["result"] for name, value in baseline["steps"].items()}
    for name, count, mean in (("causal", 1082505, .9662483387944264),
                              ("long-audit", 131071, .03898642613150546)):
        assert originals[name]["total"]["count"] == count
        assert abs(originals[name]["total"]["logits_mean"] - mean) < 1e-14, "original W0 identity failure"
    vocab_path = args.vocab or job.parent / "prepared/vocab.bin"
    packed = vocab_path.read_bytes()
    assert len(packed) == 32 and sum(x.bit_count() for x in packed) == V, "205-byte vocabulary failure"
    inputs, all_pieces = {}, {}
    for name, role in (("train", "train"), ("held", "validation"), ("max-audit", "audit")):
        if role == "audit":
            directory = job / "long-audit"
            token_path, prior_path = directory / "tokens.u8", directory / "priors.f16"
            bound_path = directory / "bounds.i32"
            audit = manifest["independent_long_audit"]
            selected = [dict(article=audit["article"], begin=audit["source_begin"],
                             end=audit["source_end"], role="audit", targets=audit["targets"])]
            nn = originals["long-audit"]["piece_losses"]
        else:
            token_path, prior_path = job / f"capture/{role}.u8", job / f"capture/{role}.f16"
            bound_path = job / f"data/{role}.i32"
            selected = [p for p in plan["intervals"] if p["role"] == role]
            nn = originals["causal"]["piece_losses"] if role == "validation" else None
        raw_tokens = token_path.read_bytes()
        tokens = np.frombuffer(raw_tokens, dtype=np.uint8)
        bounds, bound_id = load_bounds(bound_path, len(tokens))
        assert len(selected) == len(bounds) - 1, "source/local piece count failure"
        identities = []
        for row, a, b in zip(selected, bounds, bounds[1:]):
            assert row["end"] - row["begin"] == b - a and row["targets"] == b - a - 1
            identities.append(dict(article=row["article"], role=row["role"],
                                   source_token_range=[row["begin"], row["end"]]))
        size = prior_path.stat().st_size
        assert size == len(tokens) * V * 2, "f16 file length failure"
        with prior_path.open("rb") as stream:
            pieces = scan(stream, tokens, bounds, identities, nn)
        all_pieces[name] = pieces
        inputs[name] = dict(tokens=dict(path=str(token_path), bytes=len(raw_tokens)),
                            priors=dict(path=str(prior_path), bytes=size),
                            bounds=bound_id)
        expected = plan["roles"][role]["targets"] if role != "audit" else audit["targets"]
        assert sum(p["targets"] for p in pieces) == expected, "scope target count failure"
    held = all_pieces["held"]
    threshold = plan["train_piece_limit"]
    short = [p for p in held if p["local_token_range"][1] - p["local_token_range"][0] <= threshold]
    long = [p for p in held if p["local_token_range"][1] - p["local_token_range"][0] > threshold]
    scores = {s["name"]: s["score"] for s in manifest["samples"]}
    assert [p["piece_index"] for p in long] == scores["causal"]["long_piece_indices"] == [127, 157, 181]
    scopes = {name: aggregate(pieces) for name, pieces in all_pieces.items()}
    scopes.update({"held-short": aggregate(short), "held-long": aggregate(long)})
    assert scopes["held-short"]["targets"] + scopes["held-long"]["targets"] == scopes["held"]["targets"]
    for name, baseline_name in (("held", "causal"), ("max-audit", "long-audit")):
        for field in ("logits_sum", "probs_sum"):
            assert abs(scopes[name]["nn_baseline"][field] - originals[baseline_name]["total"][field]) < 1e-6
        scopes[name]["nn_baseline"] = originals[baseline_name]["total"]
        scopes[name]["nn_gain_prior_minus_nn"] = pair(scopes[name]["prior"], scopes[name]["nn_baseline"])
    result = dict(schema="hutter.prior-value.v1", status="passed",
                  method="original-causal-v1-prior-only-input-scan", inputs=inputs,
                  identity=dict(sample_plan=plan_id, manifest=manifest_id, baseline=baseline_id,
                                vocabulary=dict(path=str(vocab_path), bytes=32,
                                                columns=V, ordering="ascending set real-byte positions")),
                  scope=dict(seed=plan["seed"], probability=plan["probability"],
                             article_holdout_probability=plan["article_holdout_probability"],
                             source_bytes=plan["source_bytes"], full_nn_targets=plan["full_targets"],
                             held_short_max_piece_bytes=threshold,
                             held_long_piece_indices=[127, 157, 181],
                             independent_max_audit=manifest["independent_long_audit"]),
                  definitions=dict(primary="f16_normalized", target_mask="row i -> token i+1 within each original piece; skip piece-last row / first-byte target",
                                   prior_native="PPMd::ByteUpdate: sqp >=1, vocabulary mask, fp32 normalization; WritePpmdRow: RNE IEEE binary16",
                                   f16_target_raw="-ln(exact IEEE f16 target value); rounded row mass retained; zero targets use null total",
                                   f16_normalized="ln(sum205 exact IEEE f16 values) - ln(target value); softmax(log prior) in real arithmetic",
                                   floor_value_float32=FLOOR,
                                   f16_floor1e6_normalized="ln(sum205 max(exact IEEE f16 value,float32(1e-6))) - ln(guarded target)",
                                   predictor_f16_tail_floor1e6_normalized="same ratio distribution with predictor F16C bulk exact decode and scalar-tail subnormal x0.5 in columns200..204 before guard",
                                   nn_raw="original activation-probe double logsumexp of native softcapped fp32 last_logits",
                                   nn_probs="original activation-probe -ln(double(native fp32 NN softmax probability))",
                                   nn_gain_sign="prior NLL minus original NN NLL; positive is incremental NN improvement",
                                   accumulation="numpy.float64 per-row sums and NLL, math.fsum chunk and piece totals",
                                   conclusion_scope="sampled original NN target mask and input-rounded prior distributions; independent max-audit is report-only; native bit-mixer and coded package value use their measured stream scope"),
                  scopes=scopes, pieces=all_pieces, resources=resources())
    assert result["resources"]["max_rss_kib"] <= 128 * 1024, "128 MiB RSS budget exceeded"
    with args.output.open("x") as out:
        os.chmod(args.output, 0o600)
        json.dump(result, out, allow_nan=False, separators=(",", ":"))
        out.write("\n")
    print(json.dumps(dict(status=result["status"], output=str(args.output),
                          scopes=result["scopes"], resources=result["resources"]),
                     allow_nan=False, separators=(",", ":")))


def resources():
    return dict(cpu_seconds=time.process_time() - START_CPU,
                wall_seconds=time.monotonic() - START_WALL,
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                affinity=list(os.sched_getaffinity(0)), numpy_version=np.__version__,
                chunk_rows=CHUNK, cpu_limit_seconds=_cap, wall_limit_seconds=120,
                as_limit_mib=resource.getrlimit(resource.RLIMIT_AS)[0] // (1 << 20),
                niceness=os.nice(0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--out", "--output", dest="output", type=Path)
    parser.add_argument("--cpu-seconds", type=int, choices=[120], default=120)
    parser.add_argument("--block-rows", type=int, choices=[4096], default=4096)
    parser.add_argument("--as-mib", type=int, choices=[256, 512], default=512)
    parser.add_argument("--self-test", action="store_true")
    main(parser.parse_args())
