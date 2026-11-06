#!/usr/bin/env python3
"""Copy the already completed causal inputs once, directly between CPU workers.

The source descriptor is a private, short-lived file with the admitted machine
URL and account headers. Sampling, PPMd history, native baseline and tensors are
reused. An actual interrupted download is retained for operator review.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import shutil
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from importlib.util import module_from_spec, spec_from_file_location


def map_files(manifest, source, incoming):
    mapped, files = copy.deepcopy(manifest), {}
    for sample in mapped["samples"]:
        for record in (sample, sample["score"]):
            for key in ("tokens", "priors", "boundaries"):
                if key not in record:
                    continue
                original = Path(record[key])
                destination = incoming / original.relative_to(source)
                files[str(original)] = destination
                record[key] = str(destination)
    return mapped, files


def copy_stream(response, destination, expected, prefix=0):
    """Keep an inspected partial file; this API serves whole files, not Range.

    The original validation stream ended early. Advancing past its saved prefix
    costs network bytes only; completed inputs are reused without another GET.
    """
    assert 0 <= prefix <= expected, (str(destination), prefix, expected)
    remaining = prefix
    while remaining:
        block = response.read(min(1 << 20, remaining))
        assert block, f"source ended before saved prefix: {destination}"
        remaining -= len(block)
    with destination.open("ab" if prefix else "wb") as output:
        shutil.copyfileobj(response, output, length=1 << 20)
    actual = destination.stat().st_size
    assert actual == expected, f"{destination}: expected {expected} bytes, received {actual}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--resume-failed", action="store_true",
                        help="reuse the inspected original failed transfer and its files")
    args = parser.parse_args()
    job, config = args.job.resolve(), args.source_json.resolve()
    incoming = job / "incoming"
    incoming.mkdir(exist_ok=True)
    state_path = incoming / "transfer-state.json"
    previous = {}
    if args.resume_failed:
        previous = json.loads(state_path.read_text())
        assert previous["status"] == "failed", previous["status"]
        assert not Path(f"/proc/{previous['pid']}").exists(), "original transfer is still live"
    else:
        assert not state_path.exists(), "inspect this original transfer before restarting"
    assert json.loads((job / "check/preflight.json").read_text())["status"] == "passed"
    source = json.loads(config.read_text())
    source_job = Path(source["job"])
    assert not previous or previous["source_job"] == str(source_job)
    prior_cpu = float(previous.get("total_cpu_seconds", previous.get("cpu_seconds", 0.)))
    started = time.time()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
    remaining_cpu = int(180-prior_cpu)
    assert remaining_cpu > 2, "original transfer CPU allowance exhausted"
    resource.setrlimit(resource.RLIMIT_CPU, (remaining_cpu-2, remaining_cpu))
    state = dict(schema="hutter.depth-input-transfer.v1", status="running",
                 stage="source_baseline", pid=os.getpid(), started_at=started,
                 source_job=str(source_job), files=copy.deepcopy(previous.get("files", [])),
                 prior_cpu_seconds=prior_cpu,
                 reused_completed_bytes=sum(row["bytes"] for row in previous.get("files", [])))

    def save(stage, **values):
        cpu = time.process_time()
        state.update(stage=stage, updated_at=time.time(), cpu_seconds=cpu,
                     total_cpu_seconds=prior_cpu+cpu, **values)
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(state_path)

    def request(path):
        url = source["base"] + "/files/content?" + urlencode({"path": str(path)})
        return urlopen(Request(url, headers=source["headers"]), timeout=120)

    def remote_json(path):
        with request(path) as response:
            assert response.status == 200
            return json.load(response)

    code = 1
    save("source_baseline")
    try:
        baseline = remote_json(source_job / "baseline.json")
        assert baseline["status"] == "passed"
        original = remote_json(source_job / "data-manifest.json")
        plan = remote_json(source_job / "data/plan.json")
        assert plan["seed"] == 414 and plan["method"] == original["method"]
        mapped, files = map_files(original, source_job, incoming)
        (incoming / "source-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        completed = {row["source"]: row for row in state["files"]}
        for original_path, destination in files.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if original_path in completed:
                row = completed[original_path]
                assert str(destination) == row["destination"]
                assert destination.stat().st_size == row["bytes"], str(destination)
                save("file_reused", current_file=original_path)
                continue
            prefix = 0
            if destination.exists():
                assert args.resume_failed and previous.get("current_file") == original_path, str(destination)
                prefix = destination.stat().st_size
            save("copying", current_file=original_path)
            with request(original_path) as response:
                assert response.status == 200
                expected = int(response.headers["Content-Length"])
                save("copying", expected_bytes=expected, reused_prefix_bytes=prefix)
                copy_stream(response, destination, expected, prefix)
            state["files"].append(dict(source=original_path, destination=str(destination),
                                      bytes=expected, reused_prefix_bytes=prefix))
            save("file_complete")
        manifest_path, baseline_path = incoming / "data-manifest.json", incoming / "baseline.json"
        manifest_path.write_text(json.dumps(mapped, indent=2) + "\n")
        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")
        ready = dict(status="ready", data_manifest=str(manifest_path), baseline=str(baseline_path),
                     source2=dict(job=str(source_job), method=original["method"], sampling_seed=plan["seed"],
                                  source_bytes=original["source_bytes"], training_pieces=original["training_pieces"],
                                  training_targets=original["training_targets"], held_targets=1082505,
                                  independent_audit_targets=131071))
        spec = spec_from_file_location("depth_pipeline", Path(__file__).with_name("depth-training-job.py"))
        pipeline = module_from_spec(spec)
        spec.loader.exec_module(pipeline)
        pipeline.mapped_inputs(incoming, ready)
        ledger_path = job / "prep-ledger.json"
        ledger = json.loads(ledger_path.read_text())
        item = dict(id="causal-input-transfer", cpu_seconds=prior_cpu+time.process_time(),
                    prior_failed_cpu_seconds=prior_cpu,
                    scope="destination transfer attempts; source capture and baseline reused")
        assert not any(row["id"] == item["id"] for row in ledger["items"])
        ledger["items"].append(item)
        ledger["cpu_seconds"] = sum(row["cpu_seconds"] for row in ledger["items"])
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
        save("inputs_ready", status="passed", total_bytes=sum(row["bytes"] for row in state["files"]))
        tmp = incoming / "ready.tmp"
        tmp.write_text(json.dumps(ready, indent=2) + "\n")
        tmp.replace(incoming / "ready.json")
        code = 0
        print(json.dumps(state), flush=True)
    except Exception as error:
        save("failed", status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (incoming / "transfer.exit").write_text(str(code) + "\n")
        config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
