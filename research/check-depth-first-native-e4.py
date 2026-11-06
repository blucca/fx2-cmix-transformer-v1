#!/usr/bin/env python3
"""Replay the first depth e4 scope and size sensitivity with the stdlib only."""
import json
import math
from pathlib import Path


def main():
    result = json.loads((Path(__file__).with_name("results") / "depth-first-native-e4.json").read_text())
    h = result["native"]["held"]
    l = result["native"]["held_long"]
    s = result["native"]["held_short"]
    a = result["native"]["maximum_length_audit"]
    assert (h["targets"], l["targets"], s["targets"], a["targets"]) == (1082505, 107647, 974858, 131071)
    for row in (h, l, s, a):
        assert math.isclose(row["student_nll"] - row["original_nll"], row["delta_nll"], abs_tol=1e-14)
    for field in ("student_nll", "original_nll", "delta_nll"):
        weighted = (l["targets"] * l[field] + s["targets"] * s[field]) / h["targets"]
        assert math.isclose(weighted, h[field], abs_tol=1e-14)
    phase = result["phase"]
    assert (phase["continuous_epochs_completed"], phase["hard_qat_epochs_completed"], phase["goal_epochs"]) == (4, 0, 32)
    assert phase["completed_full_epoch_targets"] == 4 * 2941307
    p = result["actual_package"]
    dw = 2 * (p["original_weight_bytes"] - p["e4_weight_bytes"])
    db = 2 * (p["original_binary_bytes"] - p["student_binary_bytes"])
    assert (dw, db, dw + db) == (1892704, 9024, 1901728)
    assert dw == p["double_weight_saved_bytes"] and db == p["double_binary_saved_bytes"]
    assert 2 * (p["e4_weight_bytes"] - p["initialized_weight_bytes"]) == 5572
    scenario = result["named_size_sensitivity"]
    assert (scenario["full_native_targets"], scenario["historical_record_gap_bytes"]) == (586215331, 952225)
    extra = scenario["full_native_targets"] * h["delta_nll"] / (8 * math.log(2))
    space = dw + db
    historical_allowance = space - scenario["historical_record_gap_bytes"]
    assert historical_allowance == 949503
    assert math.isclose(extra, scenario["kappa1_payload_increase_bytes"], abs_tol=1e-6)
    assert math.isclose(space / extra, scenario["break_even_kappa"], abs_tol=1e-14)
    assert math.isclose(historical_allowance / extra, scenario["historical_record_kappa"], abs_tol=1e-14)
    # Each threshold satisfies its own scoped integer-byte gate exactly.
    assert math.isclose(scenario["break_even_kappa"] * extra - space, 0, abs_tol=1e-6)
    assert math.isclose(scenario["historical_record_kappa"] * extra - space, -952225, abs_tol=1e-6)
    print(json.dumps(dict(status="passed", continuous_epochs=4, hard_qat_epochs=0,
                          double_weight_plus_binary_saved_bytes=space,
                          kappa1_payload_increase_bytes=extra,
                          break_even_kappa=space / extra,
                          historical_record_kappa=historical_allowance / extra), indent=2))


if __name__ == "__main__":
    main()
