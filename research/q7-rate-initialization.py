#!/usr/bin/env python3
"""Read-only q7 rate diagnostic from the published exact INT4 histogram.

The round241 map is elementwise, so its initialized per-tensor histogram and
the soft-rate derivative need neither a model load nor another native run.
This checks initialization; the admitted 32-epoch recipe remains unchanged.
"""
import json
import math
from pathlib import Path

from q7_teacher_only import quant7_value

LEVELS = tuple(range(-3, 4))
TAU = .5
N = 586459020


def kernel(z):
    values = [math.exp(-(z - j) ** 2 / TAU) for j in LEVELS]
    total = math.fsum(values)
    return [v / total for v in values]


def hard_entropy(counts):
    n = sum(counts)
    return math.fsum(c * math.log(n / c) for c in counts if c)


def soft_distribution(counts):
    return [math.fsum(c * kernel(s)[j] for c, s in zip(counts, LEVELS)) / sum(counts)
            for j in range(7)]


def soft_derivative(p, s):
    """One coordinate's dH/dz, with its tensor histogram held as the baseline."""
    k = kernel(s)
    mean = math.fsum(v * j for v, j in zip(k, LEVELS))
    return -math.fsum(2 / TAU * (j - mean) * v * math.log(pj)
                      for j, v, pj in zip(LEVELS, k, p))


def static_payload_bits(counts):
    """Exact 15-symbol/4096-total frequency rule; ideal payload, no framing."""
    counts = [0] * 4 + list(counts) + [0] * 4
    n = sum(counts)
    freq = [1 + c * 4081 // n for c in counts]
    remainder = [c * 4081 % n for c in counts]
    while sum(freq) < 4096:
        i = max(range(15), key=lambda i: remainder[i])
        freq[i] += 1
        remainder[i] = 0
    assert sum(freq) == 4096 and min(freq) >= 1
    return math.fsum(c * math.log2(4096 / f) for c, f in zip(counts, freq) if c)


def check():
    assert hard_entropy([2, 2]) == 4 * math.log(2)
    counts = [2, 3, 4, 16, 5, 6, 2]
    p, n, s, eps = soft_distribution(counts), sum(counts), 0, 1e-5
    old = kernel(s)
    def perturbed_entropy(delta):
        shifted = kernel(s + delta)
        changed = [pj + (new - base) / n for pj, new, base in zip(p, shifted, old)]
        return -n * math.fsum(v * math.log(v) for v in changed)
    finite = (perturbed_entropy(eps) - perturbed_entropy(-eps)) / (2 * eps)
    assert math.isclose(finite, soft_derivative(p, s), abs_tol=1e-8)
    assert static_payload_bits(counts) >= hard_entropy(counts) / math.log(2)


def main():
    check()
    root = Path(__file__).with_name("results")
    census = json.loads((root / "weights-census.json").read_text())
    admitted = json.loads((root / "q7-teacher-only-preflight.json").read_text())
    hard = soft = payload_bits = 0.
    coordinates = tensors = disagreement_count = directed_count = 0
    weighted_gradient = disagreement_gradient = max_gradient = 0.
    examples, disagreements_by_symbol = [], {}
    for row in census["per_tensor"]:
        counts = [0] * 7
        for i, c in enumerate(row["histogram"]):
            counts[quant7_value(i - 7) + 3] += c
        n, p = sum(counts), soft_distribution(counts)
        assert n == row["elements"]
        coordinates += n
        tensors += 1
        hard += hard_entropy(counts)
        soft -= n * math.fsum(v * math.log(v) for v in p)
        payload_bits += static_payload_bits(counts)
        for s, c in zip(LEVELS, counts):
            if c == 0:
                continue
            g = soft_derivative(p, s)
            max_gradient = max(max_gradient, abs(g))
            target = s - (1 if g > 0 else -1)
            if abs(g) < 1e-12 or target not in LEVELS:
                continue
            moved = list(counts)
            moved[s + 3] -= 1
            moved[target + 3] += 1
            delta = hard_entropy(moved) - hard_entropy(counts)
            directed_count += c
            weighted_gradient += c * abs(g)
            if delta > 0:
                disagreement_count += c
                disagreement_gradient += c * abs(g)
                group = disagreements_by_symbol.setdefault(s, dict(coordinates=0, absolute_gradient_weight=0.))
                group["coordinates"] += c
                group["absolute_gradient_weight"] += c * abs(g)
                examples.append(dict(tensor=row["name"], source=s, target=target,
                                     count=c, soft_dH_dz=g, hard_step_delta_nats=delta))
    assert tensors == 111 and coordinates == 5868864
    recorded = admitted["initial_export"]["current_soft_entropy_nats"]
    # The published trainer accumulates FP32 softmax means; this census uses
    # binary64 and groups exact integer centers. The 1-nat allowance covers
    # the observed sub-nat reduction error across all 111 tensors.
    assert abs(soft - recorded) < 1.
    assert payload_bits >= hard / math.log(2)
    result = dict(schema="hutter.q7.initial-rate.v1", scope="exact mapped initialization only",
                  tensors=tensors, coordinates=coordinates,
                  hard_entropy_nats=hard, soft_entropy_nats_binary64=soft,
                  admitted_soft_entropy_nats_fp32=recorded,
                  soft_minus_hard_one_copy_bytes=(soft - hard) / (8 * math.log(2)),
                  double_copy_soft_minus_hard_nll=2 * (soft - hard) / N,
                  static4096_payload_bytes=payload_bits / 8,
                  static4096_payload_excess_over_hard_bytes=(payload_bits - hard / math.log(2)) / 8,
                  nominal_table_bytes=111 * 28,
                  exported_file_bytes=admitted["initial_export"]["weight_bytes"],
                  local_direction=dict(directed_count=directed_count,
                      hard_step_rate_increase_count=disagreement_count,
                      hard_step_rate_increase_fraction=disagreement_count / directed_count,
                      absolute_gradient_weighted_fraction=disagreement_gradient / weighted_gradient,
                      maximum_abs_two_copy_objective_gradient=2 * max_gradient / N,
                      adam_epsilon=admitted["recipe"]["eps"],
                      disagreements_by_initial_symbol=disagreements_by_symbol,
                      examples=sorted(examples, key=lambda r: r["count"] * abs(r["soft_dH_dz"]), reverse=True)[:5]),
                  operations=dict(model_loads=0, training_runs=0, native_runs=0, codec_runs=0, network_calls=0))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
