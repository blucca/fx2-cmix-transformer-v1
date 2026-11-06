#!/usr/bin/env python3
"""Replay frozen length-cell sensitivities from existing native outputs (stdlib).

Within-cell population deltas are named assumptions. Empty cells keep explicit
inputs; the original model selection and actual mixed measurements are retained.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "results/native-length-support.json"


def calculate(data):
    causal = json.loads((HERE / "results" / data["sources"][0]).read_text())["pairs"]
    depth = json.loads((HERE / "results" / data["sources"][1]).read_text())
    paired = depth["paired_pieces"]
    assert len(causal) == len(paired) == 473
    assert len({p["article"] for p in causal}) == 473
    for c, d in zip(causal, paired):
        assert c["targets"] == d[0] == c["piece_bytes"] - 1
        assert c["source_end"] - c["source_begin"] == c["piece_bytes"]
        assert math.isclose(c["nll_by_epoch"]["0"] * d[0], d[1], abs_tol=1e-7)

    cells = []
    for cell in data["census"]:
        lo, hi = cell["piece_bytes"]
        indices = [i for i, c in enumerate(causal) if lo <= c["piece_bytes"] <= hi]
        n = sum(causal[i]["targets"] for i in indices)
        assert len(indices) == cell["observed_pieces"] and n == cell["observed_targets"]
        deltas = {}
        if n:
            for epoch in (1, 2):
                deltas[f"causal_e{epoch}"] = math.fsum(
                    (causal[i]["nll_by_epoch"][str(epoch)] - causal[i]["nll_by_epoch"]["0"])
                    * causal[i]["targets"] for i in indices) / n
            for name, column in (("depth_e8", 2), ("depth_e12", 3)):
                deltas[name] = math.fsum(paired[i][column] - paired[i][1] for i in indices) / n
            deltas["depth_e12_minus_e8"] = deltas["depth_e12"] - deltas["depth_e8"]
        cells.append(dict(cell, native_delta_nats_per_target=deltas or None))

    total = sum(c["population_targets"] for c in cells)
    supported = sum(c["population_targets"] for c in cells if c["observed_targets"])
    missing = total - supported
    assert total == 586215331 and supported == 547402930 and missing == 38812401
    assert sum(c["observed_targets"] for c in cells) == 1082505
    assert cells[-1]["observed_targets"] == 0 and all(c["observed_targets"] for c in cells[:-1])
    divisor = 8 * math.log(2)
    models = {}
    for model in ("causal_e1", "causal_e2", "depth_e8", "depth_e12"):
        summed = math.fsum(c["population_targets"] * c["native_delta_nats_per_target"][model]
                          for c in cells if c["observed_targets"])
        models[model] = dict(supported_conditional_delta_nats=summed / supported,
                             full_native_support_contribution_nats=summed / total,
                             supported_native_byte_coefficient=summed / divisor)

    largest = data["largest_article"]
    assert largest["end"] - largest["begin"] == largest["bytes"]
    cap = largest["native_piece_limit"]
    pieces = (largest["bytes"] + cap - 1) // cap
    last_bytes = largest["bytes"] - (pieces - 1) * cap
    assert 65536 < last_bytes <= cap
    giant_targets = largest["bytes"] - pieces
    p_zero_given_held = (1 - largest["piece_sampling_probability"]) ** pieces
    p_held = largest["article_holdout_probability"]
    cluster = dict(pieces=pieces, last_piece_bytes=last_bytes, targets=giant_targets,
                   missing_target_share=giant_targets / missing,
                   other_missing_pieces=cells[-1]["population_pieces"] - pieces,
                   other_missing_targets=missing - giant_targets,
                   design_zero_observations_given_held=p_zero_given_held,
                   design_zero_held_observations=1 - p_held + p_held * p_zero_given_held)

    package = depth["actual_package"]["double_weight_plus_binary_saved_bytes"]
    gap = data["scenario_inputs"]["historical_record_gap_bytes"]
    margin = package - gap
    a = models["depth_e12"]["supported_native_byte_coefficient"]
    scenarios = []
    for name, delta in (("zero_missing_native_delta", 0),
                        ("missing_equals_supported_mean", models["depth_e12"]["supported_conditional_delta_nats"]),
                        ("missing_equals_local_audit_delta", depth["native"]["maximum_length_audit"]["delta_w0"])):
        coefficient = a + missing * delta / divisor
        scenarios.append(dict(assumption=name, missing_native_delta_nats=delta,
                              full_native_byte_coefficient=coefficient,
                              uniform_kappa_at_historical_record=margin / coefficient))
    kappa = data["scenario_inputs"]["kappa_supported"]
    q_budget = (margin - kappa * a) / (missing / 8)
    causal_sizes = json.loads((HERE / "results" / data["sources"][2]).read_text())["weights"]
    extra_w = 2 * (causal_sizes["epoch2_bytes"] - causal_sizes["epoch1_bytes"])
    e2_advantage = (models["causal_e1"]["supported_native_byte_coefficient"]
                    - models["causal_e2"]["supported_native_byte_coefficient"])
    assert extra_w == 84 and math.isclose(e2_advantage, 2134.04896261578, abs_tol=1e-7)
    assert package == 1895710 and margin == 943485
    assert math.isclose(a, 17479700.927613, abs_tol=1e-5)
    assert math.isclose(kappa * a + missing / 8 * q_budget, margin, abs_tol=1e-8)
    return dict(cells=cells, population_targets=total, supported_targets=supported,
                missing_targets=missing, supported_target_share=supported / total,
                models=models, largest_article=cluster,
                causal_e2_over_e1_sensitivity=dict(supported_native_gain_bytes=e2_advantage,
                    extra_double_weight_bytes=extra_w,
                    net_gain_at_kappa1_equal_missing_payload_bytes=e2_advantage - extra_w),
                payload_sensitivity=dict(double_W_B_saved_bytes=package,
                    historical_record_gap_bytes=gap, payload_allowance_bytes=margin,
                    missing_payload_coefficient_bytes_per_bit_per_target=missing / 8,
                    giant_payload_coefficient=giant_targets / 8,
                    other_missing_payload_coefficient=(missing - giant_targets) / 8,
                    kappa_supported=kappa, missing_payload_excess_budget_bits_per_target=q_budget),
                uniform_kappa_scenarios=scenarios)


if __name__ == "__main__":
    report = json.loads(REPORT.read_text())
    result = calculate(report)
    assert result == report["computed"]
    print(json.dumps(dict(status="passed", **{k: v for k, v in result.items() if k != "cells"}), indent=2))
