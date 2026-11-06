#!/usr/bin/env python3
"""One trained depth8 export versus retained v22 depth12, <=8 CPU hours.

Reuse passed original sample traces/round trips. The supplied deployment must
link old non-NN objects with the actual eight-block NN objects and charge each
side's deployed UPX runner. Preceding phases supply build/capture/training receipts.
"""
import argparse
from array import array
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location(
    "causal_mixed", Path(__file__).with_name("causal-mixed-audit.py"))
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)
CPU_LIMIT = 28800
W0_SHA = "659c0767bce4146d2c457b9aaa13b3e707561ed5d37173a2a6ea1e5965e86b18"
V2_SHA = "7f4db6c8c843a7e6264b6a48ed4805e9e431f543df7a9a0ecb37a35a5e4b8860"
FIXED = {"binary": (181956, "686cb39ca80e6aeb24b558aefd1f59e56f01c6e4352cdc3304bb61256d8953ee"),
         "compressed_dictionary": (100098, "0bf16f1b29d854758c4fb3a16336a4b8fc112194b3cbfa8994c21c35a934b21a"),
         "compressed_order": (201111, "05ea75cb2cc644e004f880c8e68cac2fbd43186b4d42d77877d2d5634ce61b49")}
V22_SOURCES = {
    "src/predictor.cpp": "36610335d3b9c09de47dde06baf39269d058a9d696794829b9762bd510ff6837",
    "src/predictor.h": "c074b4621085b70f70365e526182615cc3a69a5c54d7720d449bde82df303723",
    "src/mixer/mixer.cpp": "6fb3e2df88429382452b4df6095b0604a7fe84953678b839e2ec0b01b8193a76",
    "src/mixer/mixer.h": "6a32194ad23ace57d66d0cb64b028698d405f0a3b56c53e20cd43f1fb284914e",
    "src/models/ppmd.cpp": "fe371c9f43a3d4f9cce2f24f87df0ce888858466a7aa5530523e047ea008dac1",
    "src/coder/encoder.cpp": "d91631eed50e5d520fe3e7c9b518fbebd62ffc62e5ed95209335e6b093fc1b45",
    "research/trace.h": "13c37af0dc657506b0b51636951d3526d47561e9c77047cb584d44a48eb93dc3"}
SPECS = {"prefix8": (0, 8 << 20, 7344758, 1041690, 1041397, 1072279),
         "q2": (293232482, 1 << 20, 0, 1048576, 1048367, 208769)}


def read(path):
    return json.loads(Path(path).read_text())


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": old.sha(path)}


def checked_file(row):
    path = Path(row["path"]).resolve()
    assert file_record(path) == row, str(path)
    return path


def retained_link(base, native, output, trace):
    """Exactly the successful 414 probe / 412 runner link, replacing tf_* only."""
    manifest = read(base / "build.json")
    assert manifest["status"] == "built"
    command = next(c for c in manifest["commands"] if c[-2:] == ["-o", "cmix"])
    repair = native / "repair-round420.json"
    if repair.is_file():
        r = read(repair)
        assert r["status"] == "passed"
        nn = next(row["command"] for row in r["runs"] if row["id"] == "depth8-reference-attn-relink-cmix" and row["exit_code"] == 0)
    else:
        nn = next(c for c in read(native / "build.json")["commands"] if c[-2:] == ["-o", "cmix"])
    original_nn = {Path(a).name for a in command if a.endswith(".o") and Path(a).name.startswith("tf_")}
    native_nn = [Path(a).name for a in nn if a.endswith(".o") and Path(a).name.startswith("tf_")]
    assert original_nn <= set(native_nn) and "tf_kernels_reference.o" in native_nn
    extra = [native / name for name in native_nn if name not in original_nn]
    return retained_command(command, base, native, output, trace, extra)


def retained_command(command, base, native, output, trace, extra_nn=()):
    if trace:
        command = [x for x in command[:-2] if x not in ("runner.o", "-s")]
        command += ["probe_research.o", "-o", "probe"]
    linked = []
    for arg in command[:-1]:
        if arg.endswith(".o"):
            arg = str((native if Path(arg).name.startswith("tf_") else base) / Path(arg).name)
        linked.append(arg)
    assert linked[-1] == "-o"
    return [*linked[:-1], *map(str, extra_nn), "-o", str(output)]


def build_contract(b):
    assert b["status"] == "built" and b["layers"] == 8
    trace, production, native = (Path(b[k]).resolve() for k in ("trace_base", "production_base", "native8"))
    n = read(native / "build.json")
    assert n["status"] == "built" and n["family"]["layers"] == 8
    repair = native / "repair-round420.json"
    assert b["native_receipt"] == file_record(repair if repair.is_file() else native / "build.json")
    if repair.is_file():
        r = read(repair)
        assert r["status"] == "passed" and r["native"]["status"] == "passed"
        for name, row in r["after_binaries"].items():
            actual = file_record(native / name)
            assert (actual["bytes"], actual["sha256"]) == (row["bytes"], row["sha256"])
        for name, row in r["source_files"].items():
            actual = file_record(Path(n["source"]) / name)
            assert (actual["bytes"], actual["sha256"]) == (row["bytes"], row["sha256"])
    paths = {name: checked_file(row) for name, row in b["outputs"].items()}
    assert b["upx_version"].startswith("upx 5.2.1")
    assert b["links"]["probe"] == retained_link(trace, native, paths["probe"], True)
    assert b["links"]["binary"] == retained_link(production, native, Path(b["runner_elf"]).resolve(), False)
    assert b["upx_command"] == [b["upx"], "-9", b["runner_elf"], "-o", str(paths["binary"])]
    objects = {arg for command in b["links"].values() for arg in command if arg.endswith(".o")}
    assert set(b["input_objects"]) == objects
    for path, row in b["input_objects"].items():
        assert str(checked_file(row)) == path
    for name, digest in V22_SOURCES.items():
        assert b["codec_sources"][name]["sha256"] == digest
        checked_file(b["codec_sources"][name])
    for base, traced in [(trace, True), (production, False)]:
        flags = {arg for c in read(base / "build.json")["commands"] for arg in c if arg.startswith("-D")}
        assert {"-DSEED=923", "-DUPDATE_LIMIT=3000"} <= flags
        assert ("-DHUTTER_TRACE" in flags) == traced
        assert all(not any(tag in flag for tag in ("HUTTER_MIXER_TRACE", "HUTTER_PPMD_TRACE",
                   "HUTTER_FAST_FXCM", "HUTTER_ACTIVE_AUX", "HUTTER_MX9_FULL_CONTEXT")) for flag in flags)
    assert math.isfinite(b["cpu_seconds"]) and 0 <= b["cpu_seconds"] < CPU_LIMIT


def deployment_contract(d):
    assert d["status"] == "built" and d["family"] == "old-v22-depth8-static15"
    b = read(d["build_receipt"])
    build_contract(b)
    paths = {name: checked_file(row) for name, row in d["common"].items()}
    for side in ("baseline", "candidate"):
        for name, row in d[side].items():
            paths[side + "_" + name] = checked_file(row)
    paths["interval"] = checked_file(d["interval"])
    assert b["outputs"] == {"probe": d["candidate"]["probe"], "binary": d["candidate"]["binary"]}
    assert (d["baseline"]["weights"]["bytes"], d["baseline"]["weights"]["sha256"]) == (2904325, W0_SHA)
    for name, expected in FIXED.items():
        row = d["baseline"][name] if name == "binary" else d["common"][name]
        assert (row["bytes"], row["sha256"]) == expected
    assert d["common"]["stream"]["bytes"] == old.FULL_BYTES and d["common"]["vocab"]["bytes"] == 32
    reference = Path(d["reference_job"]).resolve()
    r = read(reference / "mixed-build/build.json")
    assert r["status"] == "built" and r["reference_relink_byte_equal"] is True
    assert paths["baseline_probe"] == reference / "mixed-build/probe"
    return paths, b


def student_schema(row):
    assert tuple(row[k] for k in ("state_tensors", "quantized_weights", "activation_scales", "raw_fp32", "tensor_count")) == (289, 75, 80, 59, 294)
    assert row["max_scale_delta"] == 0 and row["original_rope_bytes_equal"] is True
    assert row["decoded_tensor_bytes_equal"] is True and row["derived_tensors"] == 5
    r = row["static15_recode"]
    assert r["status"] == "passed" and r["all_tensor_bytes_equal"] is True and r["int4_tensors"] == 75
    assert Path(row["weight_file"]).stat().st_size == row["weight_bytes"]


def terminal_contract(pipeline, trajectory, code):
    assert trajectory["status"] in ("complete", "cpu_budget") and code in (0, 124)
    assert pipeline["status"] == pipeline["training_status"] == trajectory["status"]
    assert pipeline["stage"] == "training_terminal" and pipeline["training_exit_code"] == code


def aggregate(paired, double_w, dual_b):
    payload = sum(row["payload_saved_bytes"] for row in paired.values())
    return {"scope": "full prefix8 plus fresh q2 payloads; one double-W and one dual-deployed-UPX-B delta; fixed D/O",
            "sample_count": 2, "sample_payload_saved_bytes": payload,
            "double_weight_saved_bytes": double_w, "dual_code_saved_bytes": dual_b,
            "paired_sample_component_saved_bytes": payload + double_w + dual_b}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, default=Path("/data/workspace/temp/hutter/round419"))
    p.add_argument("--deployment", type=Path, help="Actual v22-object links, two probes/UPX runners and passed reference paths")
    p.add_argument("--range-plan", type=Path, help="Transferred original data/plan.json; defaults to incoming/source-plan.json")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        self_check()
        return
    assert a.deployment is not None
    job = a.job.resolve()
    out = job / "depth-mixed-audit"
    out.mkdir(exist_ok=True)
    state_path = out / "state.json"
    assert not state_path.exists(), "inspect the single existing mixed receipt and its CPU before continuation"
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (24 << 30, 24 << 30))
    build_cpu = 0.
    state = dict(schema="hutter.depth-mixed.v1", status="checking", family="old-v22-depth8-static15",
                 started_at=time.time(), steps={}, paired={}, commands={}, cpu_limit_seconds=CPU_LIMIT,
                 address_space_limit_gib=24, trace_budget_bytes=old.TRACE_LIMIT, new_ppmd_capture_runs=0)
    deadline = time.time() + 604800

    def cpu():
        x, y = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        return build_cpu + x.ru_utime + x.ru_stime + y.ru_utime + y.ru_stime

    def save(**values):
        state.update(cpu_seconds=cpu(), updated_at=time.time(), **values)
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(state_path)

    def run(label, argv, trace=None):
        remaining = math.floor(CPU_LIMIT-cpu())-2
        assert remaining >= 30, "mixed CPU budget exhausted"
        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (remaining-1, remaining))
        command = ["timeout", "-k", "30s", str(max(1, math.floor(deadline-time.time()))), *map(str, argv)]
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   HUTTER_TRACE_PATH=str(trace) if trace else "", HUTTER_PPMD_TRACE_PATH="", HUTTER_MIXER_TRACE_PATH="")
        log, started = out / (label+".log"), time.time()
        with log.open("w") as f:
            process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=env, cwd=out, preexec_fn=limits)
            save(status="running", phase=label, command=command, command_pid=process.pid)
            _, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
        state["commands"][label] = dict(exit_code=process.returncode, cpu_seconds=usage.ru_utime+usage.ru_stime,
            max_process_rss_kib=usage.ru_maxrss, wall_seconds=time.time()-started, command=command)
        save()
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
        return log

    save()
    try:
        pipeline, trajectory = read(job / "pipeline-state.json"), read(job / "training/state.json")
        code = int((job / "training.exit").read_text())
        terminal_contract(pipeline, trajectory, code)
        recipe = trajectory["identity"]["recipe"]
        assert recipe["epochs"] == 32 and recipe["continuous_epochs"] == 8 and recipe["evaluation_every"] == 4
        assert recipe["kept_layers"] == [0, 1, 2, 3, 8, 9, 10, 11]
        assert recipe["frozen"] == "all signed BF16 weight and activation scales"
        data = read(trajectory["identity"]["data"])
        assert (data["method"], data["source_bytes"], data["training_pieces"], data["training_targets"]) == ("causal-whole-piece-v1", old.FULL_BYTES, 1313, 2941307)
        assert [s["name"] for s in data["samples"]] == ["causal", "long-audit"]
        assert [s["score"]["targets"] for s in data["samples"]] == [1082505, 131071]
        assert 0 <= trajectory["epoch"] <= 32 and 0 <= trajectory["next_piece"] <= 1313
        assert 0 <= trajectory["epoch_targets"] <= 2941307
        assert trajectory["seen_targets"] == trajectory["epoch"]*2941307+trajectory["epoch_targets"]
        assert trajectory["next_piece"] != 0 or trajectory["epoch_targets"] == 0
        completed = trajectory["epoch"] + int(trajectory["next_piece"] == 1313)
        assert completed <= 32 and (trajectory["status"] != "complete" or completed == 32)
        d = read(a.deployment)
        paths, build = deployment_contract(d)
        build_cpu = build["cpu_seconds"]
        remaining = math.floor(CPU_LIMIT-cpu())
        assert remaining > 30
        resource.setrlimit(resource.RLIMIT_CPU, (remaining-1, remaining))
        assert Path(trajectory["identity"]["original_static15"]).read_bytes() == paths["baseline_weights"].read_bytes()
        baseline = read(job / "incoming/baseline.json")
        assert baseline["status"] == "passed"
        base_total = old.native_total(baseline["steps"]["causal"]["result"], 1082505)
        zero = trajectory["zero_step"]
        student_schema(zero)
        zero_sha = old.sha(Path(zero["weight_file"]))
        assert zero_sha != W0_SHA
        exports = []
        for row in trajectory["evaluations"]:
            epoch = row["epoch"]
            assert 0 < epoch <= completed and epoch % 4 == 0 and row["step"] > 0
            student_schema(row)
            for name, targets in [("causal", 1082505), ("long-audit", 131071)]:
                held = old.native_total(old.activation_result(job / f"training/epoch-{epoch:02d}-{name}.log"), targets)
                assert (held["count"], held["logits_mean"]) == (row["scores"][name]["targets"], row["scores"][name]["raw_nll"])
            digest = old.sha(Path(row["weight_file"]))
            exports.append(dict(epoch=epoch, step=row["step"], weight_file=row["weight_file"], weight_bytes=row["weight_bytes"],
                held_total={"count": 1082505, "logits_mean": row["scores"]["causal"]["raw_nll"]},
                original_byte_equal=digest == W0_SHA, initial_student_byte_equal=digest == zero_sha, report_only=row["scores"]))
        assert exports, "terminal trajectory has zero completed trained native exports"
        best, rows = old.choose_export(base_total, old.FULL_TARGETS, 2904325, exports)
        dual_b = 2*(d["baseline"]["binary"]["bytes"]-d["candidate"]["binary"]["bytes"])
        for row in rows:
            row.update(dual_code_saved_bytes=dual_b, proxy_net_saved_bytes=row["proxy_net_saved_bytes"]+dual_b)
        best = next(row for row in rows if row["epoch"] == best["epoch"])
        weight = Path(best["weight_file"])
        plan = read(a.range_plan or job / "incoming/source-plan.json")
        boundaries = array("i")
        boundaries.frombytes((paths["stream"].parent / "article_boundaries.i32").read_bytes())
        assert plan["source_bytes"] == old.FULL_BYTES and plan["full_targets"] == old.FULL_TARGETS
        assert plan["seed"] == 414 and plan["roles"]["validation"]["targets"] == 1082505
        assert (plan["roles"]["train"]["pieces"], plan["roles"]["train"]["targets"]) == (1313, 2941307)
        state.update(baseline_weight=d["baseline"]["weights"], candidate_weight=file_record(weight),
            baseline_probe=d["baseline"]["probe"], candidate_probe=d["candidate"]["probe"],
            B0=d["baseline"]["binary"], B1=d["candidate"]["binary"], deployment_sha256=old.sha(a.deployment),
            build_receipt=build, prior_mixed_build_cpu_seconds=build_cpu, original_weight_bytes_equal=False,
            double_weight_saved_bytes=best["double_weight_saved_bytes"], dual_code_saved_bytes=dual_b,
            pipeline_terminal=dict(status=trajectory["status"], training_exit=code, target_epochs=32,
                completed_epochs=completed, completed_continuous_epochs=min(8, completed), completed_hard_epochs=max(0, completed-8),
                partial_epoch_next_piece=trajectory["next_piece"], seen_targets=trajectory["seen_targets"],
                training_cpu_seconds=trajectory["total_cpu_seconds"], large_python_child_reaped=True),
            selection=dict(status="frozen", scope="one positive-epoch trained hard/native export; full realized held proxy",
                criterion="(held_NLL0-held_NLL1)*586215331/(8*ln2)+2*(W0-W1)+2*(B0-B1)", kappa=1,
                held_targets=1082505, baseline_held_total=base_total, original_weight_sha256=W0_SHA,
                selected_weight_sha256=old.sha(weight), selected=best, completed_exports=rows,
                initial_student=dict(epoch=0, original_byte_equal=False, selected=False, weight_file=zero["weight_file"], weight_bytes=zero["weight_bytes"], sha256=zero_sha),
                tie_breaker="earliest epoch", independent_max_audit_role="report-only", held_long3_role="report-only",
                realized_sampling_range=old.selection_range(plan, boundaries)))
        assert state["selection"]["selected"]["epoch"] > 0 and best["original_byte_equal"] is False
        save(status="frozen")
        token_dir = out / "data"
        token_dir.mkdir(exist_ok=True)
        packed = paths["vocab"].read_bytes()
        values = [b for b in range(256) if (packed[b//8] >> (b%8)) & 1]
        assert len(values) == 205
        mapping = bytearray([255]*256)
        for token, byte in enumerate(values):
            mapping[byte] = token
        reference = Path(d["reference_job"])
        state["cases"] = []
        trace_bytes = sum(spec[1]*128 for spec in SPECS.values())
        assert trace_bytes <= old.TRACE_LIMIT and shutil.disk_usage(out).free > old.TRACE_LIMIT
        save(trace_planned_bytes=trace_bytes, required_free_disk_bytes=old.TRACE_LIMIT,
             original_encode_runs=0, original_decode_runs=0)
        for name, (offset, length, begin, count, targets, expected) in SPECS.items():
            with paths["stream"].open("rb") as f:
                f.seek(offset)
                raw = f.read(length)
            tokens = raw[begin:begin+count].translate(mapping)
            assert 255 not in tokens and old.target_count(tokens) == (targets, count-targets)
            token_path = token_dir / (name+".tokens.u8")
            token_path.write_bytes(tokens)
            original_dir = reference / "mixed-audit" / name / "baseline"
            golden = paths["stream"].parent.parent / ("runs/long-prefix/prefix-8m/baseline" if name == "prefix8" else "runs/first-ab/article-q2-1m/baseline")
            original, g = read(original_dir / "result.json"), read(golden / "result.json")
            assert original["status"] == g["status"] == "passed" and g["encoded_bytes"] == expected
            assert (g["identity"]["sample"]["offset"], g["identity"]["sample"]["length"]) == (offset, length)
            for key in ("stream", "vocab", "dictionary"):
                assert g["identity"][key+"_sha256"] == d["common"][key]["sha256"]
            assert g["identity"]["weights_sha256"] == V2_SHA
            raw_sha = hashlib.sha256(raw).hexdigest()
            assert g["original_slice_sha256"] == g["restored_slice_sha256"] == raw_sha
            assert g["decode"]["mode"] == "decode" and g["decode"]["output_bytes"] == length
            assert (original_dir / "encoded.bin").stat().st_size == expected
            assert (original_dir / "encoded.bin").read_bytes() == (golden / "encoded.bin").read_bytes()
            assert original["encode"]["mode"] == "encode" and original["encode"]["input_bytes"] == length
            assert original["encode"]["output_bytes"] == expected
            accounting = original["interval"]
            assert accounting["status"] == "passed" and accounting["full_input_bytes"] == length and accounting["full_coded_bytes"] == expected
            assert tuple(accounting[k] for k in ("score_begin", "score_end", "pieces")) == (begin, begin+count, count-targets)
            assert accounting["all"]["bytes"] == count and accounting["neural_targets"]["bytes"] == targets
            original["reused_encoded_sha256"] = old.sha(original_dir / "encoded.bin")
            original["decode"] = dict(original_golden_decode_reused=True, original_mixed_result_reused=True,
                source_result=str(original_dir / "result.json"), golden_result=str(golden / "result.json"), restored_slice_sha256=raw_sha)
            original["reuse_identity"] = dict(weights=d["baseline"]["weights"], probe=d["baseline"]["probe"],
                golden_identity=g["identity"], prior_golden_decode=g["decode"], encoded_sha256=original["reused_encoded_sha256"])
            destination = out / name / "candidate"
            destination.mkdir(parents=True, exist_ok=True)
            encoded, trace = destination / "encoded.bin", destination / "trace.bin"
            e = old.probe_result(run(name+"-candidate-encode", [paths["candidate_probe"], "encode", paths["stream"], encoded,
                paths["vocab"], paths["dictionary"], weight, offset, length], trace))
            assert e["mode"] == "encode" and e["input_bytes"] == length and e["output_bytes"] == encoded.stat().st_size
            assert trace.stat().st_size == length*128
            decoded = destination / "decoded.bin"
            r = old.probe_result(run(name+"-candidate-decode", [paths["candidate_probe"], "decode", encoded, decoded,
                paths["vocab"], paths["dictionary"], weight, offset, length]))
            assert r["mode"] == "decode" and r["output_bytes"] == length and decoded.read_bytes() == raw
            r["restored_slice_sha256"] = old.sha(decoded)
            interval = read(run(name+"-candidate-interval", [paths["interval"], trace, token_path, paths["vocab"], begin, encoded.stat().st_size]))
            assert interval["status"] == "passed" and interval["full_input_bytes"] == length and interval["full_coded_bytes"] == encoded.stat().st_size
            assert tuple(interval[k] for k in ("score_begin", "score_end", "pieces")) == (begin, begin+count, count-targets)
            assert interval["all"]["bytes"] == count and interval["neural_targets"]["bytes"] == targets
            candidate = dict(status="passed", encode=e, decode=r, interval=interval, encoded_sha256=old.sha(encoded), trace_bytes=trace.stat().st_size)
            (destination / "result.json").write_text(json.dumps(candidate, indent=2)+"\n")
            state["steps"][name+"-baseline"], state["steps"][name+"-candidate"] = original, candidate
            state["paired"][name] = old.paired_record(original, candidate)
            state["cases"].append(dict(name=name, offset=offset, length=length, score_begin=begin, score_bytes=count,
                neural_targets=targets, first_token_fallback_bytes=count-targets,
                statistical_start="stream0_continuous" if name == "prefix8" else "fresh_cold"))
            save()
        shared_delta = best["double_weight_saved_bytes"]+dual_b
        state["actual_mixed"] = aggregate(state["paired"], best["double_weight_saved_bytes"], dual_b)
        state["rates"] = {name: dict(scope="same selected W and actual measured interval; fullstream rate scenario",
            score_bytes=count, actual_payload_saved_bytes=saved,
            projected_full_net_saved_bytes=old.FULL_BYTES*saved/count+shared_delta)
            for name, saved, count in [("mature", state["paired"]["prefix8"]["interval_coded_saved_bytes"], SPECS["prefix8"][3]),
                                       ("prefix8", state["paired"]["prefix8"]["payload_saved_bytes"], 8 << 20)]}
        state["full_s"] = dict(status="awaiting_actual_full_stream_package", source_bytes=old.FULL_BYTES,
            native_targets=old.FULL_TARGETS, known_double_weight_saved_bytes=best["double_weight_saved_bytes"],
            known_dual_code_saved_bytes=dual_b, actual_full_s_saved_bytes=None,
            score="P+2B+2D+O+2W+69", fixed_aux={k: d["common"][k] for k in ("compressed_dictionary", "compressed_order")})
        save(status="complete", finished_at=time.time())
    except Exception as error:
        save(status="cpu_budget" if cpu() >= CPU_LIMIT-30 else "failed", error=repr(error), finished_at=time.time())
        raise


def self_check():
    # Metadata only: epoch0 has a genuine eight-block identity, outside selection.
    for status, code in (("complete", 0), ("cpu_budget", 124)):
        terminal_contract(dict(status=status, stage="training_terminal", training_status=status,
                               training_exit_code=code), dict(status=status), code)
    baseline = {"count": 8, "logits_mean": 2.5}
    exports = [dict(epoch=e, step=1, weight_bytes=90, held_total={"count": 8, "logits_mean": 3.},
                   original_byte_equal=False) for e in (8, 4)]
    zero = dict(exports[0], epoch=0, step=0, weight_bytes=1, held_total={"count": 8, "logits_mean": 0.})
    best, _ = old.choose_export(baseline, 80*math.log(2), 100, [zero, *exports])
    assert best["epoch"] == 4 and best["proxy_net_saved_bytes"] == 15
    negative = dict(exports[1], held_total={"count": 8, "logits_mean": 10.})
    best, _ = old.choose_export(baseline, 80*math.log(2), 100, [zero, negative])
    assert best["epoch"] == 4 and best["proxy_net_saved_bytes"] < 0
    assert aggregate({"prefix8": {"payload_saved_bytes": 10}, "q2": {"payload_saved_bytes": -2}}, 20, -6)["paired_sample_component_saved_bytes"] == 22
    source = ["clang++-17", "-m64", "predictor.o", "fxcmv1.o", "mixer.o", "tf_model_opt.o", "runner.o", "-s", "-o", "cmix"]
    extra = [Path("/native8/tf_kernels_reference.o")]
    trace = retained_command(source, Path("/old-trace"), Path("/native8"), Path("/out/probe"), True, extra)
    runner = retained_command(source, Path("/old-prod"), Path("/native8"), Path("/out/cmix"), False, extra)
    assert trace == ["clang++-17", "-m64", "/old-trace/predictor.o", "/old-trace/fxcmv1.o", "/old-trace/mixer.o", "/native8/tf_model_opt.o", "/old-trace/probe_research.o", "/native8/tf_kernels_reference.o", "-o", "/out/probe"]
    assert runner[-6:] == ["/native8/tf_model_opt.o", "/old-prod/runner.o", "-s", "/native8/tf_kernels_reference.o", "-o", "/out/cmix"]
    print(json.dumps({"status": "passed", "checks": ["current complete/cpu_budget terminal", "positive trained epoch with earliest tie and negative proxy", "same held target mask", "double-W and real dual-B once", "depth epoch0 outside W0 selection", "v22 non-NN retained plus canonical-attention object"], "heavy_calls": 0}))


if __name__ == "__main__":
    main()
