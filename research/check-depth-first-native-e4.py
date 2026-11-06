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
    epochs = result["continuous_epoch_training"]
    assert [r["epoch"] for r in epochs] == list(range(1, 8))
    assert all((r["targets"], r["pieces"]) == (2941307, 1313) for r in epochs)
    assert sum(r["optimizer_steps"] for r in epochs[:4]) == 315
    assert all(a["ce_mean"] > b["ce_mean"] and a["teacher_kl_mean"] > b["teacher_kl_mean"]
               for a, b in zip(epochs, epochs[1:]))
    fit = result["observed_fit_change"]
    for metric, output in (("ce_mean", "ce"), ("teacher_kl_mean", "kl"), ("final_hidden_mse_mean", "hidden_mse")):
        reduction = 1 - epochs[3][metric] / epochs[0][metric]
        assert math.isclose(reduction, fit[f"epoch_one_to_four_{output}_reduction_fraction"], abs_tol=1e-14)
    g = result["e4_coordinate_geometry"]
    assert (g["int4_values"], g["changed_z_coordinates"], g["changed_weight_symbols"]) == (3951936, 3921246, 716512)
    assert g["changed_z_coordinates"] - g["changed_weight_symbols"] == g["displaced_z_still_rounding_to_initial_symbol"] == 3204734
    assert math.isclose(g["displaced_z_same_initial_round_fraction"], 3204734 / 3951936, abs_tol=1e-14)
    assert (g["changed_raw_values"], g["raw_fp32_elements"], g["max_scale_delta"]) == (18853, 18854, 0)
    gradients = result["recorded_gradients"]
    assert (gradients["first_four_epoch_optimizer_steps"], gradients["snapshot_optimizer_steps"]) == (315, 610)
    assert gradients["finite_tensors_per_step"] == gradients["nonzero_tensors_per_step"] == 134
    pair = result["e4_piece_pairing"]
    assert pair["worse_pieces"] + pair["better_pieces"] == pair["pieces"] == 473
    assert pair["worse_targets"] + pair["better_targets"] == pair["targets"] == h["targets"]
    assert math.isclose(pair["weighted_delta_nats"], h["delta_nll"], abs_tol=1e-12)
    cache = result["teacher_cache_coverage"]
    assert cache["misses_at_epoch_one_end"] == cache["misses_at_snapshot"] == cache["training_pieces"] == 1313
    snapshot = result["recorded_training_snapshot"]
    assert (snapshot["completed_continuous_epochs"], snapshot["hard_qat_epochs_completed"], snapshot["latest_native_evaluation_epoch"]) == (7, 0, 4)
    assert 7 * 2941307 + snapshot["partial_epoch_targets"] == snapshot["recorded_seen_targets"] == 22691163
    assert sum(r["optimizer_steps"] for r in epochs) + snapshot["partial_epoch_optimizer_steps"] == snapshot["step"] == 610
    cpu = result["actual_experiment_cpu"]
    assert math.isclose(cpu["e4_evaluated_reported_total_seconds"] - cpu["e4_trained_reported_total_seconds"],
                        cpu["e4_export_and_both_native_evaluations_seconds"], abs_tol=1e-9)
    assert math.isclose(cpu["causal_native_reported_cpu_seconds"] + cpu["maximum_length_audit_reported_cpu_seconds"],
                        cpu["both_native_reported_cpu_seconds"], abs_tol=1e-12)
    comparison = result["native_cpu_comparison"]
    for scope in ("causal_held", "maximum_length_audit"):
        row = comparison[scope]
        assert math.isclose(row["student_program_cpu_seconds"] / row["original_program_cpu_seconds"],
                            row["raw_program_cpu_ratio"], abs_tol=1e-14)
    print(json.dumps(dict(status="passed", continuous_epochs=4, hard_qat_epochs=0,
                          double_weight_plus_binary_saved_bytes=space,
                          kappa1_payload_increase_bytes=extra,
                          break_even_kappa=space / extra,
                          historical_record_kappa=historical_allowance / extra,
                          continuous_epochs_in_metrics_snapshot=7,
                          displaced_coordinates_still_rounding_to_initial=3204734,
                          worse_held_pieces=pair["worse_pieces"]), indent=2))


if __name__ == "__main__":
    main()
