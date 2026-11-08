#!/usr/bin/env python3
"""Check cached q7 native-stage arithmetic with the public full12 reference."""
import json
import math
import sys
from pathlib import Path

results = Path(__file__).with_name("results")
path = Path(sys.argv[1]) if len(sys.argv) > 1 else results / "q7-native-e4.json"
current = json.loads(path.read_text())
reference = json.loads((results / "causal-first-eleven-epochs.json").read_text())
base = reference["baseline"]
old = next(row for row in reference["epochs"] if row["epoch"] == current["epoch"])

def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), (actual, expected)

assert current["schema"] == "hutter.q7.teacher-only.native-stage.v1"
assert current["shared_contract"]["online_phase"] == "hard from first update"
rows = current["domains"]
assert [(r["scope"], r["targets"], r["pieces"]) for r in rows] == [
    ("full_held", 1082505, 473), ("held_long", 107647, 3), ("giant", 131071, 1)]
for row, prefix in zip(rows, ("all", "long", "audit")):
    close(row["baseline_W0_raw_nll"], base[prefix + "_nll"])
    close(row["old_q15_CE_KL_raw_nll"], old[prefix + "_nll"])
full, long, giant = rows
short = current["derived_fullheld_remainder"]
assert short["targets"] == full["targets"] - long["targets"] == 974858
assert short["pieces"] == full["pieces"] - long["pieces"] == 470
assert full["targets"] + giant["targets"] == current["shared_contract"]["independent_scored_targets"] == 1213576
for key in ("baseline_W0_raw_nll", "old_q15_CE_KL_raw_nll", "q7_teacher_only_raw_nll"):
    close(short[key], (full[key]*full["targets"] - long[key]*long["targets"])/short["targets"])
for row in rows + [short]:
    q, o, b = [row[k] for k in ("q7_teacher_only_raw_nll", "old_q15_CE_KL_raw_nll", "baseline_W0_raw_nll")]
    close(row["q7_minus_old_raw_nll"], q-o)
    close(row["q7_minus_W0_raw_nll"], q-b)
    close(row["old_minus_W0_raw_nll"], o-b)
w = current["weights"]
assert w["baseline_W0_bytes"] == base["weight_bytes"] == 2904325
assert w["old_q15_bytes"] == old["weight_bytes"]
assert w["q7_growth_from_initial_bytes"] == w["q7_bytes"] - w["initial_q7_bytes"]
assert w["q7_double_weight_saved_bytes"] == 2*(w["baseline_W0_bytes"]-w["q7_bytes"])
assert w["old_double_weight_saved_bytes"] == 2*(w["baseline_W0_bytes"]-w["old_q15_bytes"])
assert w["q7_double_weight_charge_improvement_over_old_bytes"] == 2*(w["old_q15_bytes"]-w["q7_bytes"])
p = current["primary_selection_proxy"]
assert p["target_denominator"] == 586215331 and p["kappa"] == 1
close(p["current_native_delta_nats_per_target"], full["q7_minus_W0_raw_nll"])
close(p["delta_budget_for_zero_proxy_nats_per_target"], w["q7_double_weight_saved_bytes"]*8*math.log(2)/p["target_denominator"])
expected = w["q7_double_weight_saved_bytes"] - full["q7_minus_W0_raw_nll"]*p["target_denominator"]/(8*math.log(2))
assert math.isclose(p["proxy_saved_bytes"], expected, rel_tol=0, abs_tol=1e-6)
assert all(v == 0 for v in current["reporting_operations"].values())
print(json.dumps(dict(status="passed", epoch=current["epoch"], independent_native_targets=1213576,
    remainder_targets=974858, q7_double_weight_saved_bytes=w["q7_double_weight_saved_bytes"], heavy_calls=0)))
