#!/usr/bin/env python3
"""Read the admitted q7 trajectory and its own selected-weight codec results.

Standard library only. Inputs are the once-collected small originals; this
script performs no training, native evaluation, codec run, or network request.
The full12 W0 reference and the separately measured q7 e0 control are reused.
"""
import argparse
import collections
import json
import math
from pathlib import Path

from q7_teacher_only import data_objective

JOB = "/data/workspace/temp/hutter/round484/q7-teacher"
METHOD = "q7-teacher-only-fixed-scale-integer-coordinate-v1"
TARGETS = 2941307
REFERENCE = Path(__file__).with_name("results") / "causal-first-eleven-epochs.json"
INITIAL_CONTROL = REFERENCE.with_name("q7-initial-native.json")


def load(path):
    return json.loads(Path(path).read_text())


def online_epochs(records, recipe):
    steps = [r for r in records if r["kind"] == "optimizer_step"]
    result = []
    for epoch in sorted({r["epoch"] for r in steps}):
        part = [r for r in steps if r["epoch"] == epoch]
        n = sum(r["targets"] for r in part)
        means = {key: math.fsum(r[key] * r["targets"] for r in part) / n
                 for key in ("ce_mean", "teacher_kl_mean", "soft_entropy_nats", "objective")}
        expected = data_objective(means["ce_mean"], means["teacher_kl_mean"], recipe)
        expected += 2 * means["soft_entropy_nats"] / recipe["full_targets"]
        assert math.isclose(expected, means["objective"], rel_tol=0, abs_tol=1e-9)
        assert all(r["finite_gradient_tensors"] == 111 and r["teacher_cache_misses_this_run"] == 0 for r in part)
        pieces = sum(r["pieces"] for r in part)
        assert n <= TARGETS and pieces <= 1313
        result.append(dict(epoch=epoch + 1, phase="hard", complete=n == TARGETS,
                           targets=n, pieces=pieces, optimizer_steps=len(part), **means))
    return result


def native_log(path):
    rows = [json.loads(line.split("ACTIVATION_RESULT ", 1)[1])
            for line in Path(path).read_text().splitlines() if line.startswith("ACTIVATION_RESULT ")]
    assert len(rows) == 1
    return rows[0]


def native_scopes(causal, giant, long_indices):
    def total(pieces):
        n = sum(p["loss"]["count"] for p in pieces)
        loss = math.fsum(p["loss"]["logits_sum"] for p in pieces)
        return dict(pieces=len(pieces), targets=n, raw_nll=loss / n)
    values = dict(full_held=total(causal["piece_losses"]),
                  held_long=total([causal["piece_losses"][i] for i in long_indices]),
                  held_short=total([p for i, p in enumerate(causal["piece_losses"]) if i not in long_indices]),
                  giant=total(giant["piece_losses"]))
    assert [values[k]["targets"] for k in values] == [1082505, 107647, 974858, 131071]
    return values


def initial_native():
    # This fixed file belongs to this admitted job, not a future recipe/run.
    if not INITIAL_CONTROL.exists():
        return None, None
    control = load(INITIAL_CONTROL)
    assert control["schema"] == "hutter.q7.initial-native.v1"
    assert control["status"] == "complete" and control["source_job"] == JOB and control["epoch"] == 0
    assert control["weight_bytes"] == 2044814 and control["initial_reference_bytes_equal"]
    assert control["native_calls"] == 2 and control["independent_scored_targets"] == 1213576
    causal, giant = [control["scopes"][name]["native"] for name in ("causal", "long-audit")]
    values = native_scopes(causal, giant, control["held_long"]["piece_indices"])
    return control, values


def readout(out, attribution_path=None):
    raw = out / "raw"
    receipt, identity, index = [load(out / name) for name in
                               ("collection-receipt.json", "selected-identity.json", "native-source-index.json")]
    assert receipt["status"] == index["status"] == "passed"
    assert receipt["ordinal"] == 2 and receipt["job_root"] == identity["job_root"] == JOB
    mixed, training, pipeline, preflight, prep = [load(raw / name) for name in
        ("mixed-state.json", "training-state.json", "pipeline-state.json", "preflight.json", "prep-ledger.json")]
    recipe = training["identity"]["recipe"]
    assert recipe == preflight["recipe"] and recipe["method"] == METHOD and recipe["epochs"] == 32
    assert recipe["label_ce"] == recipe["final_hidden_mse"] == recipe["raw_trainable_tensors"] == 0
    assert recipe["teacher_kl_coefficient"] == 2 and recipe["hard_integer_range"] == [-3, 3]
    assert preflight["status"] == "passed" and preflight["exact_round241_quant7_export"]
    assert training["identity"]["teacher_cache_mode"] == "read-only"
    assert training["cpu_limit_seconds"] == pipeline["cpu_limit_seconds"] == 172800
    records = [json.loads(line) for line in (raw / "metrics.jsonl").read_text().splitlines() if line.strip()]
    online = online_epochs(records, recipe)
    assert sum(r["targets"] for r in online) == training["seen_targets"]
    completed = training["epoch"] + int(training["next_piece"] == 1313)
    assert training["seen_targets"] == training["epoch"] * TARGETS + training["epoch_targets"]
    assert training["status"] != "complete" or (completed == 32 and training["seen_targets"] == 32 * TARGETS)
    reference = load(REFERENCE)
    base, long_ix = reference["baseline"], reference["held_long_piece_indices"]
    baseline = dict(full_held=base["all_nll"], held_long=base["long_nll"],
                    held_short=base["short_nll"], giant=base["audit_nll"])
    initial_control, initial_values = initial_native()
    if initial_values:
        assert initial_control["weight_file"] == training["zero_step"]["weight_file"]
        for name, value in initial_values.items():
            value["delta_nll_w0"] = value["raw_nll"] - baseline[name]
    old_pieces = reference["paired_pieces"]
    bounds = [0]
    for p in old_pieces:
        bounds.append(bounds[-1] + p[0] + 1)
    selection = mixed.get("selection", {})
    selected = selection.get("selected")
    exports, pieces = [], [[p[0], p[1] * p[0]] for p in old_pieces]
    if initial_control:
        zero = initial_control["scopes"]["causal"]["native"]
        assert zero["piece_bounds"] == bounds
        for row, p in zip(pieces, zero["piece_losses"]):
            assert row[0] == p["loss"]["count"]
            row.append(p["loss"]["logits_sum"])
    for item in index["native_originals"]:
        epoch = item["epoch"]
        export = next(r for r in training["evaluations"] if r["epoch"] == epoch)
        causal = native_log(raw / f"epoch-{epoch:02d}-causal.log")
        giant = native_log(raw / f"epoch-{epoch:02d}-long-audit.log")
        assert causal["piece_bounds"] == bounds
        assert giant["piece_bounds"] == [0, 131072]
        assert [p["loss"]["count"] for p in causal["piece_losses"]] == [p[0] for p in old_pieces]
        assert export["max_scale_delta"] == 0 and export["hard_integer_limit"] == 3
        assert -3 <= export["hard_integer_range"][0] <= export["hard_integer_range"][1] <= 3
        values = native_scopes(causal, giant, long_ix)
        for name, value in values.items():
            value["delta_nll_w0"] = value["raw_nll"] - baseline[name]
            if initial_values:
                value["delta_nll_q7_e0"] = value["raw_nll"] - initial_values[name]["raw_nll"]
        assert math.isclose(values["full_held"]["raw_nll"], export["scores"]["causal"]["raw_nll"], abs_tol=1e-12)
        fixed = 2 * (2904325 - export["weight_bytes"])
        proxy = fixed - values["full_held"]["delta_nll_w0"] * 586215331 / (8 * math.log(2))
        exports.append(dict(epoch=epoch, step=export["step"], selected=bool(selected and selected["epoch"] == epoch),
            weight_bytes=export["weight_bytes"], changed_weight_symbols=export["changed_weight_symbols"],
            double_weight_saved_bytes=fixed, held_kappa1_proxy_saved_bytes=proxy, native=values))
        for row, p in zip(pieces, causal["piece_losses"]):
            row.append(p["loss"]["logits_sum"])
    same_w = {k: mixed.get(k) for k in ("selection", "baseline_weight", "candidate_weight",
        "double_weight_saved_bytes", "dual_code_saved_bytes", "paired", "actual_mixed", "rates", "full_s")}
    status = "passed_local" if mixed["status"] == "complete" else "stopped_mixed_" + mixed["status"]
    if selected:
        chosen = next(r for r in exports if r["selected"])
        assert selected["epoch"] > 0 and selected["step"] > 0 and selected["initial_student_byte_equal"] is False
        assert chosen["weight_bytes"] == selected["weight_bytes"] == mixed["candidate_weight"]["bytes"]
        assert selected["weight_file"] == mixed["candidate_weight"]["path"]
        assert mixed["candidate_weight"]["sha256"] == selection["selected_weight_sha256"]
        assert chosen["double_weight_saved_bytes"] == mixed["double_weight_saved_bytes"] and mixed["dual_code_saved_bytes"] == 0
        if mixed["status"] == "complete":
            if attribution_path is None:
                status = "waiting_same_W_attribution"
            else:
                attr = load(attribution_path)
                assert attr["status"] == "passed" and attr["job_root"] == JOB
                assert attr["same_W"]["selected_epoch"] == selected["epoch"]
                assert attr["same_W"]["selected_weight_identity"] == selection["selected_weight_sha256"]
                same_w["excluded_input_scopes"] = {name: row["paired"] for name, row in attr["scopes"].items()}
                same_w["mixed_budget_with_four_readers"] = attr["original_mixed_budget"]
    else:
        same_w["candidate_attribution"] = dict(status="stopped", reason="no selected trained hard-change weight")
    return dict(schema="hutter.q7.teacher-only.terminal.v1", status=status, ordinal=2, job_root=JOB,
        recipe=recipe, training_status=training["status"], mixed_status=mixed["status"],
        coverage=dict(target_epochs=32, completed_epochs=completed, complete32=training["status"] == "complete",
            partial_epoch_next_piece=training["next_piece"], partial_epoch_targets=training["epoch_targets"],
            seen_targets=training["seen_targets"], native_epochs=index["evaluated_epochs"]),
        objective_contract="2KL + 2H/586459020; CE report-only; all epochs hard",
        metric_counts=dict(collections.Counter(r["kind"] for r in records)), online_epochs=online,
        initial_student=dict(epoch=0, weight_bytes=training["zero_step"]["weight_bytes"],
            native_scope="complete separate e0 control" if initial_values else "unmeasured initialization in this run",
            native=initial_values, quant7_initialization=training["quant7_initialization"]),
        baseline_native=baseline, exports=exports,
        selected_epoch=selected["epoch"] if selected else None,
        final_evaluated_epoch=max(index["evaluated_epochs"], default=None),
        paired_piece_columns=["targets", "W0_logits_sum"] + (["e0_logits_sum"] if initial_values else [])
            + [f"e{r['epoch']}_logits_sum" for r in exports],
        paired_pieces=pieces, same_W_mixed=same_w,
        resource=dict(training_cpu_seconds_including_prep=training["total_cpu_seconds"],
            prep_cpu_seconds=prep["cpu_seconds"], prep_ledger=prep, pipeline_child_runs=pipeline.get("child_runs", []),
            separate_initial_native_cpu_seconds=initial_control["total_cpu_seconds"] if initial_control else 0,
            original_mixed_cpu_seconds=mixed.get("cpu_seconds"), reused_baseline_extra_cpu_seconds=0),
        operations=dict(training_runs=0, native_runs=0, codec_runs=0, network_calls=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collection", type=Path)
    mode.add_argument("--check", action="store_true", help="check the admitted scalar and a partial all-hard epoch locally")
    p.add_argument("--attribution", type=Path)
    a = p.parse_args()
    if a.check:
        admitted = load(Path(__file__).with_name("results") / "q7-teacher-only-preflight.json")
        check = admitted["preflight"]
        row = dict(kind="optimizer_step", epoch=0, targets=check["targets"], pieces=1,
            teacher_cache_misses_this_run=check["teacher_cache_misses"],
            soft_entropy_nats=admitted["initial_export"]["current_soft_entropy_nats"],
            **{k: check[k] for k in ("ce_mean", "teacher_kl_mean", "objective", "finite_gradient_tensors")})
        result = online_epochs([row, dict(row, targets=3, ce_mean=99.)], admitted["recipe"])[0]
        assert result["phase"] == "hard" and result["targets"] == 1202 and not result["complete"]
        assert math.isclose(result["objective"], check["objective"], abs_tol=1e-12)
        control, initial = initial_native()
        if initial:
            observed = load(INITIAL_CONTROL.with_name("q7-native-e8.json"))
            for row in observed["domains"]:
                name = row["scope"]
                q = initial[name]["raw_nll"] - row["baseline_W0_raw_nll"]
                t = row["q7_teacher_only_raw_nll"] - initial[name]["raw_nll"]
                assert math.isclose(q+t, row["q7_minus_W0_raw_nll"], rel_tol=0, abs_tol=1e-12)
        print(json.dumps(dict(status="passed", scope="admitted scalar; partial all-hard epoch; actual e0 decomposition",
                              initial_control=bool(control), heavy_calls=0)))
        return
    result = readout(a.collection, a.attribution)
    (a.collection / "terminal-readout.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "coverage", "selected_epoch", "same_W_mixed")}, indent=2))


if __name__ == "__main__":
    main()
