#!/usr/bin/env python3
"""Fixed-W observed article breadth and bounded cluster-composition sensitivity."""
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE / "results/causal-epoch01-paired-pieces.json"
j = json.loads(source.read_text())
assert j["schema"] == "hutter.causal-e1-paired-pieces.v1"
pairs, intervals = j["pairs"], j["validation_intervals"]
assert len(pairs) == len(intervals) == 473
rows, clusters = [], defaultdict(lambda: [0, 0.0, 0])
for i, (p, interval) in enumerate(zip(pairs, intervals)):
    assert p["piece"] == i and interval["role"] == "validation"
    assert p["targets"] == interval["targets"] == interval["end"] - interval["begin"] - 1
    assert isinstance(interval["article"], int) and interval["article"] >= 0
    assert i == 0 or intervals[i-1]["end"] <= interval["begin"]
    assert math.isclose(p["e1_raw_nll"] - p["baseline_raw_nll"], p["delta"], abs_tol=1e-14)
    row = dict(p, article=interval["article"], source_begin=interval["begin"], source_end=interval["end"],
               delta_sum_nats=p["targets"] * p["delta"])
    rows.append(row)
    c = clusters[row["article"]]
    c[0] += row["targets"]
    c[1] += row["delta_sum_nats"]
    c[2] += 1
N, D = sum(x["targets"] for x in rows), sum(x["delta_sum_nats"] for x in rows)
assert N == j["baseline_total"]["count"] == j["e1_total"]["count"] == 1082505
for key, total in (("baseline_raw_nll", j["baseline_total"]), ("e1_raw_nll", j["e1_total"])):
    assert math.isclose(sum(x["targets"] * x[key] for x in rows)/N, total["logits_mean"], abs_tol=1e-12)
assert math.isclose(D/N, j["e1_total"]["logits_mean"]-j["baseline_total"]["logits_mean"], abs_tol=1e-12)
assert len(clusters) == 473 and all(c[2] == 1 for c in clusters.values())

def group(a):
    return {"pieces": len(a), "piece_fraction": len(a)/len(rows),
            "targets": sum(x["targets"] for x in a), "target_fraction": sum(x["targets"] for x in a)/N,
            "delta_sum_nats": sum(x["delta_sum_nats"] for x in a)}

groups = {name: group([x for x in rows if test(x["delta"])]) for name, test in
          (("improving", lambda d: d < 0), ("degrading", lambda d: d > 0), ("equal", lambda d: d == 0))}
assert sum(x["pieces"] for x in groups.values()) == len(rows)
assert sum(x["targets"] for x in groups.values()) == N
maximum = max(rows, key=lambda x: abs(x["delta_sum_nats"]))
assert maximum == min(rows, key=lambda x: x["delta_sum_nats"])
ranked = sorted(rows, key=lambda x: x["delta_sum_nats"])

def removal(a):
    count, delta = sum(x["targets"] for x in a), sum(x["delta_sum_nats"] for x in a)
    return {"removed_pieces": [x["piece"] for x in a], "removed_targets": count,
            "removed_net_raw_gain_fraction": delta/D, "remaining_targets": N-count,
            "remaining_weighted_delta_raw_nll": (D-delta)/(N-count)}

strata, lower = [], 1
for cap in (2048, 4096, 8192, 16384, 32768, 65536, 131072):
    a = [x for x in rows if lower <= x["targets"]+1 <= cap]
    count = sum(x["targets"] for x in a)
    strata.append({"piece_bytes": [lower, cap], "pieces": len(a), "targets": count,
        "improving_pieces": sum(x["delta"] < 0 for x in a),
        "weighted_delta_raw_nll": sum(x["delta_sum_nats"] for x in a)/count if count else None})
    lower = cap+1
assert strata[-1]["pieces"] == 0 and sum(x["targets"] for x in strata) == N
assert {x["piece"] for x in rows if x["targets"]+1 > 32768} == set(j["long_piece_indices"])
assert all(x["weighted_delta_raw_nll"] < 0 for x in strata if x["targets"])
sample_rows, rng, boot = list(clusters.values()), random.Random(425), []
for _ in range(1000):
    sampled = rng.choices(sample_rows, k=len(sample_rows))
    boot.append(sum(x[1] for x in sampled)/sum(x[0] for x in sampled))
boot.sort()
r = {"status": "passed", "source_file": source.name,
    "source_read_at_beijing": datetime.fromtimestamp(j["read_at"], timezone(timedelta(hours=8))).isoformat(),
    "mapping": {"source": "original validation intervals aligned to native piece index; roles/coordinates/targets checked",
                "pieces": len(rows), "different_article_ids": len(clusters), "pieces_per_observed_article": 1},
    "paired_total": {"targets": N, "weighted_delta_raw_nll": D/N, "delta_sum_nats": D},
    "piece_direction_groups": groups,
    "target_fraction_definition": "fraction of held targets inside pieces grouped by that piece's mean raw-NLL direction",
    "maximum_absolute_influence_piece": maximum, "remove_maximum_absolute_influence_piece": removal([maximum]),
    "remove_top5_gaining_pieces": removal(ranked[:5]), "remove_top10_gaining_pieces": removal(ranked[:10]),
    "length_strata": strata,
    "article_cluster_composition_bootstrap": {"seed": 425, "replicates": 1000,
        "unit": "observed article ID; one piece per article in this sample",
        "statistic": "resampled sum(targets*paired_delta)/resampled sum(targets)",
        "percentile_rule": "nearest-rank: 25th and 975th ordered draws",
        "percentile_95_band_delta_raw_nll": [boot[24], boot[974]],
        "negative_delta_draws": sum(x < 0 for x in boot),
        "scope": "fixed W0/e1 pair, composition sensitivity among observed articles and sampled length range; 64-128KiB has zero observed held pieces"},
    "decision_readout": "observed article breadth and leave-largest-piece stability support completing the original bounded methods; fullstream spending follows the original actual mixed/record gate",
    "analysis_new_network_calls": 0, "analysis_new_model_calls": 0}
print(json.dumps(r, indent=2))
