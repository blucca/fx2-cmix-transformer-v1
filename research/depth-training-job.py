#!/usr/bin/env python3
"""One fixed eight-block training pipeline on the admitted free CPU worker.

The caller builds native8/recode, completes the uniform preflight, and maps
the original causal inputs from worker 2. This runner waits once, launches
one complete trainer, and records its real exit and isolated child CPU.
Native VM pause keeps these same processes; interrupted jobs require review.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


CPU_SECONDS = 172800
INPUT_WAIT_SECONDS = 86400
WALL_SECONDS = 604800
PREPARED_BYTES = 586459020
TRAIN_PIECES = 1313
TRAIN_TARGETS = 2941307
HELD_TARGETS = 1082505
AUDIT_TARGETS = 131071


def read_json(path):
    return json.loads(Path(path).read_text())


def mapped_inputs(incoming, ready):
    """Read the actual mapped manifest, its metadata sizes and baseline counts."""
    assert ready["status"] == "ready" and isinstance(ready["source2"], dict)
    data, baseline = Path(ready["data_manifest"]), Path(ready["baseline"])
    assert data.resolve() == (incoming / "data-manifest.json").resolve()
    assert baseline.resolve() == (incoming / "baseline.json").resolve()
    manifest, reference = read_json(data), read_json(baseline)
    assert manifest["method"] == "causal-whole-piece-v1"
    assert manifest["source_bytes"] == PREPARED_BYTES
    assert manifest["training_pieces"] == TRAIN_PIECES
    assert manifest["training_targets"] == TRAIN_TARGETS
    assert reference["status"] == "passed"
    samples = manifest["samples"]
    assert [s["name"] for s in samples] == ["causal", "long-audit"]
    assert samples[0]["score"]["targets"] == HELD_TARGETS
    assert samples[1]["score"]["targets"] == AUDIT_TARGETS
    assert samples[1]["train_pieces"] == []
    assert len(samples[0]["train_pieces"]) == TRAIN_PIECES
    assert sum(hi-lo-1 for lo, hi in samples[0]["train_pieces"]) == TRAIN_TARGETS
    assert all(1 < hi-lo <= 32768 for lo, hi in samples[0]["train_pieces"])
    for sample in samples:
        for record in (sample, sample["score"]):
            tokens, priors = Path(record["tokens"]), Path(record["priors"])
            assert tokens.resolve().is_relative_to(incoming.resolve())
            assert priors.resolve().is_relative_to(incoming.resolve())
            assert priors.stat().st_size == tokens.stat().st_size * 410
        score = sample["score"]
        bounds = Path(score["boundaries"])
        assert bounds.resolve().is_relative_to(incoming.resolve()) and bounds.is_file()
        assert reference["steps"][sample["name"]]["result"]["total"]["count"] == score["targets"]
    return data, baseline, manifest, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/workspace/temp/hutter"))
    args = parser.parse_args()
    root = args.root.resolve()
    job = root / "round419"
    job.mkdir(parents=True, exist_ok=True)
    state_path = job / "pipeline-state.json"
    assert not state_path.exists(), "inspect the existing pipeline; native pause retains its live process"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (7 << 30, 7 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS-2, CPU_SECONDS))
    started = time.time()
    deadline = started + WALL_SECONDS
    inputs_deadline = started + INPUT_WAIT_SECONDS
    prep_cpu = 0.
    state = dict(schema="hutter.depth-training.pipeline.v1", status="starting",
                 stage="preflight_receipt", started_at=started, root=str(root),
                 job=str(job), pipeline_pid=os.getpid(), cpu_limit_seconds=CPU_SECONDS,
                 input_wait_limit_seconds=INPUT_WAIT_SECONDS,
                 input_deadline_at=inputs_deadline, deadline_at=deadline,
                 deadline_utc=datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
                 address_space_limit_gib=7, cpu_affinity=list(os.sched_getaffinity(0)),
                 threads_per_library=1, child_runs=[])

    def total_cpu():
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        return prep_cpu + time.process_time() + child.ru_utime + child.ru_stime

    def save(stage, **updates):
        state.update(stage=stage, updated_at=time.time(), total_cpu_seconds=total_cpu(), **updates)
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(state_path)

    def finish(status, stage, code, **updates):
        save(stage, status=status, completed_at=time.time(), pipeline_exit_code=code, **updates)
        (job / "pipeline.exit").write_text(str(code) + "\n")
        print(json.dumps(state), flush=True)
        return code

    save("preflight_receipt")
    try:
        preflight_path = job / "check/preflight.json"
        preflight = read_json(preflight_path)
        assert preflight["status"] == "passed"
        ledger_path = job / "prep-ledger.json"
        ledger = read_json(ledger_path)
        prep_cpu = float(ledger["cpu_seconds"])
        assert math.isfinite(prep_cpu) and prep_cpu >= 0
        save("waiting_for_inputs", status="running", preflight=str(preflight_path),
             prep_ledger=str(ledger_path), prep_cpu_seconds=prep_cpu)
        incoming = job / "incoming"
        ready_path = incoming / "ready.json"
        while not ready_path.is_file():
            now = time.time()
            if now >= inputs_deadline or now >= deadline:
                return finish("stopped", "input_wait_stopped", 124, stop_reason="input_wait_one_day")
            time.sleep(min(30., inputs_deadline-now, deadline-now))
        ready = read_json(ready_path)
        data, baseline, manifest, reference = mapped_inputs(incoming, ready)
        save("inputs_ready", incoming_ready=str(ready_path), source2=ready["source2"],
             data_manifest=str(data), baseline=str(baseline),
             training_pieces=manifest["training_pieces"], training_targets=manifest["training_targets"],
             held_targets=HELD_TARGETS, independent_audit_targets=AUDIT_TARGETS)
        # Input transfer finishes during the wait; charge its completed prep
        # from the caller's final ledger before freezing trainer CPU credit.
        ledger = read_json(ledger_path)
        prep_cpu = float(ledger["cpu_seconds"])
        assert math.isfinite(prep_cpu) and prep_cpu >= 0
        save("prep_accounted", prep_cpu_seconds=prep_cpu)

        source = job / "source"
        files = dict(trainer=source / "research/depth-distillation.py",
                     checkpoint=root / "round416/source/models/6m-q4-fp32.tch",
                     kernel=root / "round416/cpu-kda.so",
                     weights=root / "candidate/models/6m-q4-fp32.tfwc2",
                     original_static15=job / "assets/original-static15.tfwc3",
                     evaluator=job / "native/depth8/activation-probe",
                     recode=job / "bin/recode")
        assert all(path.is_file() for path in files.values()), {k: str(v) for k, v in files.items()}
        training = job / "training"
        assert not (training / "state.json").exists(), "inspect the original admitted training trajectory"
        prior_cpu = total_cpu()
        remaining = math.floor(CPU_SECONDS-prior_cpu)
        if remaining < 180:
            return finish("cpu_budget", "prep_cpu_budget", 124)
        wall_remaining = math.floor(deadline-time.time())
        if wall_remaining <= 0:
            return finish("stopped", "wall_deadline", 124, stop_reason="seven_day_deadline")
        trainer_command = [sys.executable, "-B", "-u", str(files["trainer"])]
        for name in ("checkpoint", "kernel", "weights", "evaluator", "recode"):
            trainer_command += ["--" + name, str(files[name])]
        trainer_command += ["--data", str(data), "--baseline", str(baseline),
                            "--original-static15", str(files["original_static15"]),
                            "--output", str(training), "--cpu-seconds", str(CPU_SECONDS),
                            "--prior-cpu-seconds", str(prior_cpu)]
        # Reuse the queue's native seven-day timeout; one command group includes
        # the trainer's recode/evaluator descendants and natural VM pause time.
        command = ["timeout", "-k", "30s", str(wall_remaining), *trainer_command]
        pythonpath = str(source / "research") + ":" + str(source)
        if os.environ.get("PYTHONPATH"):
            pythonpath += ":" + os.environ["PYTHONPATH"]
        env = dict(os.environ, PYTHONPATH=pythonpath, PYTHONUNBUFFERED="1")

        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (remaining-2, remaining))

        log_path = job / "training.log"
        child_started = time.time()
        with log_path.open("w") as log:
            child = subprocess.Popen(command, cwd=job, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, preexec_fn=limits)
            save("training", status="running", command=command, trainer_command=trainer_command,
                 child_pid=child.pid, log=str(log_path), prior_cpu_seconds=prior_cpu,
                 child_cpu_limit_seconds=remaining, child_started_at=child_started)
            _, wait_status, usage = os.wait4(child.pid, 0)
            code = os.waitstatus_to_exitcode(wait_status)
            child.returncode = code
        (job / "training.exit").write_text(str(code) + "\n")
        state["child_runs"].append(dict(exit_code=code, cpu_seconds=usage.ru_utime+usage.ru_stime,
            max_process_rss_kib=usage.ru_maxrss, wall_elapsed_seconds=time.time()-child_started,
            resource_scope="wait4 command and reaped descendants", log=str(log_path)))
        terminal_path = training / "state.json"
        terminal = read_json(terminal_path) if terminal_path.is_file() else {}
        terminal_status = terminal.get("status")
        details = dict(training_exit_code=code, training_state=str(terminal_path),
                       training_status=terminal_status,
                       trainer_total_cpu_seconds=terminal.get("total_cpu_seconds"))
        if code == 0 and terminal_status in ("complete", "cpu_budget", "stopped"):
            return finish(terminal_status, "training_terminal", code, **details)
        if code == 124 and terminal_status == "cpu_budget":
            return finish("cpu_budget", "training_terminal", code, **details)
        if code in (124, 137) and time.time() >= deadline-1:
            return finish("stopped", "wall_deadline", code, stop_reason="seven_day_deadline", **details)
        return finish("failed", "training_failed", code or 1, **details)
    except Exception as exc:
        return finish("failed", "failed", 1, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
