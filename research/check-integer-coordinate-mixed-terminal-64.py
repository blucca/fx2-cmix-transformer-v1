#!/usr/bin/env python3
"""Stdlib-only replay of the original terminal mixed measurements and roles.

Usage: python3 -B check-integer-coordinate-mixed-terminal-64.py [RESULT.json]
Default: the sibling result JSON, or research/results when installed there.
This check replays the published counters and exact subset accounting.
The existing C++ reader performs the actual trace/range/token verification.
"""
import json
import math
from pathlib import Path
import sys

base = Path(__file__).resolve().parent
name = "integer-coordinate-mixed-terminal-64.json"
path = Path(sys.argv[1]) if len(sys.argv) > 1 else base/name
if not path.is_file() and len(sys.argv) == 1:
    path = base/"results"/name
r = json.loads(path.read_text())


def close(a, b, tolerance=1e-6):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=0, abs_tol=tolerance), (a, b)


assert r["status"] == "passed" and r["schema"] == "hutter.integer-coordinate.mixed-terminal-64.v1"
w = r["measured_export"]
assert (w["epoch"], w["step"], w["training_status"], w["mixed_status"]) == (64, 1904, "complete", "complete")
assert w["same_final_export_train_native_mixed"] and w["fixed_scale_and_tiny_row_bits_preserved"]
assert (w["baseline_weight_bytes"], w["candidate_weight_bytes"]) == (2904325, 2914354)
assert w["double_weight_growth_bytes"] == 2*w["single_weight_growth_bytes"] == 2*(w["candidate_weight_bytes"]-w["baseline_weight_bytes"]) == 20058
assert r["common_runtime"]["inference_code_added_bytes"] == 0 and r["common_runtime"]["baseline_reference_relink_bytes_equal"]
fields = ("neural_bits", "pre_sse_logit_bits", "final_u16_bits", "override_bits")

for case, offset, count, expected in (("prefix8", 0, 8388608, (1072279, 1158043)),
                                      ("q2", 293232482, 1048576, (208769, 151680))):
    for side, coded in zip(("baseline", "candidate"), expected):
        p = r["mixed_pairs"][case][side]
        e, i = p["encode"], p["interval"]
        assert p["status"] == i["status"] == "passed"
        assert (e["offset"], e["input_bytes"], e["output_bytes"]) == (offset, count, coded)
        assert i["full_input_bytes"] == count and i["full_coded_bytes"] == coded
        assert i["end_closed_bytes"]-i["begin_closed_bytes"] == i["interval_coded_bytes"]
        close(i["interval_coded_bytes"]-i["all"]["final_u16_bits"]/8, i["interval_range_residual_bytes"])
        assert i["all"]["bytes"] == i["score_end"]-i["score_begin"]
        assert i["all"]["bytes"]-i["neural_targets"]["bytes"] == i["pieces"]
        if side == "baseline":
            assert p["baseline_golden_bytes_equal"] and p["decode"]["reused_original_passed_decode"]
        else:
            assert p["candidate_roundtrip_bytes_equal"]
            assert (p["decode"]["mode"], p["decode"]["offset"], p["decode"]["input_bytes"], p["decode"]["output_bytes"]) == ("decode", offset, count, count)

for scope, s in r["native_scores"].items():
    assert s["same_full_piece_bounds"]
    for side in ("baseline", "candidate"):
        t = s[side]
        assert t["count"] == s["targets"]
        close(t["logits_sum"]/t["count"], t["logits_mean"], 1e-14)
    close(s["candidate"]["logits_mean"]-s["baseline"]["logits_mean"], s["delta_raw_nll"], 1e-14)
    close((s["candidate"]["logits_sum"]-s["baseline"]["logits_sum"])/math.log(2), s["delta_R_bits"])
    close(-s["delta_R_bits"]/8, s["raw_nn_saved_bytes"])

for scope, p in r["propagation"].items():
    a, b = p["baseline"], p["candidate"]
    assert a["all"]["bytes"] == b["all"]["bytes"] and a["neural_targets"]["bytes"] == b["neural_targets"]["bytes"]
    assert p["piece_start_fallback_count"] == a["all"]["bytes"]-a["neural_targets"]["bytes"]
    if scope != "q2-training-containing-complement":
        for side, x in (("baseline", a), ("candidate", b)):
            assert x["status"] == "passed"
            assert x["end_closed_bytes"]-x["begin_closed_bytes"] == x["interval_coded_bytes"]
    for side, x in (("baseline", a), ("candidate", b)):
        close(x["interval_coded_bytes"]-x["all"]["final_u16_bits"]/8, x["interval_range_residual_bytes"])
        for k in fields:
            close(x["all"][k]-x["neural_targets"][k], p["piece_start_fallback"][side][k])
    for k in fields:
        for g in ("all", "neural_targets"):
            close(b[g][k]-a[g][k], p["candidate_minus_baseline"][g][k])
        close(p["piece_start_fallback"]["candidate"][k]-p["piece_start_fallback"]["baseline"][k], p["candidate_minus_baseline_fallback"][k])
    assert b["interval_coded_bytes"]-a["interval_coded_bytes"] == p["delta_interval_coded_bytes"] == -p["interval_saved_bytes"]
    close(b["interval_range_residual_bytes"]-a["interval_range_residual_bytes"], p["delta_range_residual_bytes"])
    d = p["candidate_minus_baseline"]
    close(d["all"]["final_u16_bits"]/8, p["final_all_bit_delta_bytes"])
    close(d["neural_targets"]["final_u16_bits"]/8, p["neural_target_final_bit_delta_bytes"])
    close(p["candidate_minus_baseline_fallback"]["final_u16_bits"]/8, p["fallback_final_bit_delta_bytes"])
    close(p["neural_target_final_bit_delta_bytes"]+p["fallback_final_bit_delta_bytes"]+p["delta_range_residual_bytes"], p["delta_interval_coded_bytes"])
    close(d["neural_targets"]["pre_sse_logit_bits"]/d["neural_targets"]["neural_bits"], p["mixer_G_transfer"], 1e-14)
    close(d["neural_targets"]["final_u16_bits"]/d["neural_targets"]["neural_bits"], p["u16_G_transfer"], 1e-14)

roles = r["q2_manifest_roles"]
subsets = ("train", "cross_mid_piece", "independent_held", "eof_piece")
assert [(roles[x]["begin"], roles[x]["end"]) for x in subsets] == [(0,521436),(521436,531039),(531039,1047533),(1047533,1048576)]
assert sum(roles[x]["targets"] for x in subsets) == roles["full"]["targets"] == 1048367
assert sum(roles[x]["pieces"] for x in subsets) == roles["full"]["pieces"] == 209
for x in roles.values():
    assert x["end"]-x["begin"]-x["pieces"] == x["targets"]
g = r["q2_piece_geometry"]
assert len(g["full_bounds"])-1 == 209 and len(g["held_bounds"])-1 == 103
assert [x-531039 for x in g["full_bounds"] if 531039 <= x <= 1047533] == g["held_bounds"]
assert g["full_and_held_candidate_bounds_identical"]
for side in ("baseline", "candidate"):
    f, h, c = (r["propagation"][k][side] for k in ("q2-full", "q2-held", "q2-training-containing-complement"))
    assert (h["score_begin"], h["score_end"], h["pieces"], h["all"]["bytes"], h["neural_targets"]["bytes"]) == (531039,1047533,103,516494,516391)
    assert h["full_coded_bytes"] == f["full_coded_bytes"]
    assert f["interval_coded_bytes"]-h["interval_coded_bytes"] == c["interval_coded_bytes"]
    for group in ("all", "neural_targets"):
        assert f[group]["bytes"]-h[group]["bytes"] == c[group]["bytes"]
        for k in fields:
            close(f[group][k]-h[group][k], c[group][k])
    fn, hn = r["native_scores"]["q2-full"][side], r["native_scores"]["q2-held"][side]
    assert fn["count"]-hn["count"] == c["raw_targets"] == 531976
    close(fn["logits_sum"]-hn["logits_sum"], c["raw_logits_sum"])
    close(c["raw_logits_sum"]/c["raw_targets"], c["raw_nll"], 1e-14)
q = r["q2_role_accounting"]
assert q["full_payload_saved_bytes"] == q["held_interval_saved_bytes"]+q["training_containing_complement_interval_saved_bytes"] == 57089
assert (q["held_interval_saved_bytes"], q["training_containing_complement_interval_saved_bytes"]) == (-7085,64174)
assert q["complement_training_targets"]+q["complement_cross_eof_targets"] == q["complement_targets"] == 531976
close(q["complement_training_targets"]/q["complement_targets"], q["complement_training_fraction"], 1e-14)
history = r["q2_online_history"]
assert history["held_source_begin"] == history["probe_source_offset"]+history["held_q2_begin"] == 293763521
assert history["held_source_end"] == history["probe_source_offset"]+history["held_q2_end"] == 294280015
assert history["ppmd_and_mixer_history_bytes_before_held"] == 531039
assert history["range_replayed_from_q2_byte_zero"] and history["selected_tokens_and_vocab_bytes_checked"]
m = r["mature_scope"]
assert (m["begin"], m["end"], m["input_bytes"], m["pieces"], m["targets"], m["training_overlap_targets"]) == (7344758,8386448,1041690,293,1041397,0)
assert r["propagation"]["prefix8-mature"]["delta_interval_coded_bytes"] == 20835
expected_margins = {"prefix8-full":(-85764,-105822), "prefix8-mature-interval":(-20835,-40893),
                    "q2-full-training-containing":(57089,37031), "q2-independent-held-interval":(-7085,-27143)}
for scope, (saved, net) in expected_margins.items():
    a = r["same_component_package_margins"][scope]
    assert a["payload_or_interval_saved_bytes"] == saved
    assert a["double_weight_growth_bytes"] == 20058
    assert a["payload_or_interval_saved_bytes"]-a["double_weight_growth_bytes"] == a["common_component_margin_bytes"] == net
pc = r["package_contract"]
assert pc["formula"] == "S=P+2B+2D+O+2W+69"
assert pc["whole_prepared_stream_bytes"] == 586459020
assert not pc["full_release_S_measured"] and not pc["full_release_run_started"]
assert (pc["original_static15_verified_net_component_saved_bytes"], pc["original_minimum_prize_saved_bytes"]) == (52022,1004247)
c = r["actual_cpu"]
close(c["probe_encode_decode_seconds_included"]+c["q2_full_native_seconds_included"]+c["mixed_other_reader_wrapper_seconds_included"], c["original_mixed_queue_seconds"])
close(c["original_mixed_queue_seconds"]+c["held_readonly_seconds"], c["original_mixed_bucket_used_seconds"])
close(c["original_mixed_bucket_used_seconds"]+c["original_mixed_bucket_remaining_seconds"], c["original_mixed_bucket_limit_seconds"])
assert 0 < c["held_readonly_seconds"] < c["held_cpu_limit_seconds"] == 120
assert c["held_single_core"] == [1] and c["original_mixed_bucket_used_seconds"] < 28800
close(sum(c[k] for k in ("training_terminal_seconds", "mature_queue_seconds", "original_mixed_queue_seconds", "held_readonly_seconds")), c["independent_phase_total_seconds"])
event = r["this_event_collection"]
assert (event["get_requests"], event["authenticated_exec_requests"], event["new_trace_reader_calls"], event["full_trace_reads"]) == (5,1,2,2)
assert all(event[k] == 0 for k in ("new_nn_runs", "new_native_runs", "new_encode_runs", "new_decode_runs", "new_accounts", "new_tokens", "machine_restarts", "metadata_polls", "retries"))
d = r["decision"]
assert d["fixed_recipe_new_engineering"] == d["fixed_recipe_full_stream_advance"] == "STOP"
assert d["mature_quality_gate"] == d["independent_q2_held_gate"] == d["continuous_prefix_component_gate"] == "FAIL"
assert d["q2_full_sample_component_gate"] == "PASS" and d["sole_mixed_remaining_work"] == "NONE"
assert d["original_training_native_mixed_complete"] and d["new_verified_positive_sample_payload_saved_bytes"] == 57089
assert d["new_verified_full_stream_net_saved_bytes"] == d["new_prize_cash_eur"] == d["new_compute_cash_eur"] == 0
print(json.dumps(dict(status="passed", epoch=64, step=1904, q2_held_growth_bytes=7085,
                     mature_interval_growth_bytes=20835, q2_full_sample_saved_bytes=57089,
                     q2_training_containing_complement_saved_bytes=64174,
                     double_weight_growth_bytes=20058, full_stream_advance="STOP")))
