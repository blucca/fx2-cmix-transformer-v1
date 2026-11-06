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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    args = parser.parse_args()
    job, config = args.job.resolve(), args.source_json.resolve()
    incoming = job / "incoming"
    incoming.mkdir(exist_ok=True)
    state_path = incoming / "transfer-state.json"
    assert not state_path.exists(), "inspect this original transfer before restarting"
    assert json.loads((job / "check/preflight.json").read_text())["status"] == "passed"
    source = json.loads(config.read_text())
    source_job = Path(source["job"])
    started = time.time()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
    resource.setrlimit(resource.RLIMIT_CPU, (178, 180))
    state = dict(schema="hutter.depth-input-transfer.v1", status="running",
                 stage="source_baseline", pid=os.getpid(), started_at=started,
                 source_job=str(source_job), files=[])

    def save(stage, **values):
        state.update(stage=stage, updated_at=time.time(), cpu_seconds=time.process_time(), **values)
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
        for original_path, destination in files.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            assert not destination.exists(), "inspect existing download " + str(destination)
            save("copying", current_file=original_path)
            with request(original_path) as response:
                assert response.status == 200
                expected = int(response.headers["Content-Length"])
                with destination.open("wb") as output:
                    shutil.copyfileobj(response, output, length=1 << 20)
            assert destination.stat().st_size == expected
            state["files"].append(dict(source=original_path, destination=str(destination), bytes=expected))
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
        item = dict(id="causal-input-transfer", cpu_seconds=time.process_time(),
                    scope="destination transfer process; source capture and baseline reused")
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
