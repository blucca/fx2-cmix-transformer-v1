#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Run a serial, restartable A/B plan; checkpoint each verified round trip."""

import argparse
import fcntl
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path, offset=0, length=None):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        while length is None or length > 0:
            block = stream.read(1024 * 1024 if length is None else min(1024 * 1024, length))
            if not block:
                if length:
                    raise ValueError("short file while hashing requested slice")
                break
            value.update(block)
            if length is not None:
                length -= len(block)
    return value.hexdigest()


def save(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["baseline", "v26"])
    parser.add_argument("--trace", action="store_true",
                        help="Record 16-byte/bit diagnostics; binaries require HUTTER_TRACE")
    parser.add_argument("--timeout", type=int, default=18000,
                        help="Maximum seconds for one real encode/decode command")
    args = parser.parse_args()
    if len(args.variants) != len(set(args.variants)) or any(
            not name or Path(name).name != name or name in (".", "..")
            for name in args.variants):
        raise ValueError("variants must be unique plain directory names")
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "batch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads(args.plan.read_text())
    stream = root / "prepared/stream.bin"
    vocab = root / "prepared/vocab.bin"
    dictionary = root / "upstream/dictionary/english.dic"
    weights = root / "upstream/models/6m-q4-fp32.tfwc2"
    shared = {"stream_sha256": digest(stream), "vocab_sha256": digest(vocab),
              "dictionary_sha256": digest(dictionary), "weights_sha256": digest(weights)}

    def event(stage, **values):
        row = {"at": datetime.now(timezone.utc).isoformat(), "stage": stage, **values}
        with (out / "progress.jsonl").open("a") as log:
            log.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    for sample in plan:
        name = sample["name"]
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("sample names must be plain directory names")
        offset, length = sample["offset"], sample["length"]
        if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or length <= 0:
            raise ValueError("sample offset/length must be nonnegative/positive integers")
        reference = digest(stream, offset, length)
        for variant in args.variants:
            binary = root / "build" / variant / "probe"
            identity = {**shared, "binary_sha256": digest(binary), "sample": sample, "variant": variant}
            if args.trace:
                identity["trace_record_bytes"] = 16
            trial = out / name / variant
            trial.mkdir(parents=True, exist_ok=True)
            receipt = trial / "result.json"
            if receipt.exists():
                previous = json.loads(receipt.read_text())
                if previous.get("status") == "passed" and previous.get("identity") == identity:
                    event("reused_verified_trial", sample=name, variant=variant)
                    continue
                if previous.get("status") == "running" and previous.get("identity") == identity:
                    trial.rename(trial.with_name(variant + f"-interrupted-{time.time_ns()}"))
                    trial.mkdir()
                    event("resuming_interrupted_trial", sample=name, variant=variant)
                else:
                    raise ValueError(f"Existing failed or different trial: inspect {receipt}; use a new run directory")
            row = {"identity": identity, "status": "running", "started_at": time.time()}
            save(receipt, row)
            event("trial_started", sample=name, variant=variant, offset=offset, length=length)
            encoded, decoded = trial / "encoded.bin", trial / "decoded.bin"
            try:
                for mode, source, target in (("encode", stream, encoded), ("decode", encoded, decoded)):
                    command = [str(binary), mode, str(source), str(target), str(vocab),
                               str(dictionary), str(weights), str(offset), str(length)]
                    with (trial / f"{mode}.log").open("w") as log:
                        subprocess.run(command, cwd=trial, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=args.timeout, check=True,
                                       env=dict(os.environ,
                                                HUTTER_TRACE_PATH=str(trial / "trace.bin")
                                                if args.trace and mode == "encode" else "",
                                                TMPDIR=str(trial), OMP_NUM_THREADS="1",
                                                OPENBLAS_NUM_THREADS="1"))
                    text = (trial / f"{mode}.log").read_text()
                    marker = next(line.split("PROBE_RESULT ", 1)[1]
                                  for line in reversed(text.splitlines()) if line.startswith("PROBE_RESULT "))
                    row[mode] = json.loads(marker)
                    save(receipt, row)
                if decoded.stat().st_size != length or digest(decoded) != reference:
                    raise ValueError("lossless round-trip mismatch")
                if args.trace:
                    row["trace_bytes"] = (trial / "trace.bin").stat().st_size
                    if row["trace_bytes"] != length * 8 * 16:
                        raise ValueError("trace must contain exactly one record per coded bit")
                row.update(status="passed", encoded_bytes=encoded.stat().st_size,
                           bits_per_stream_byte=8 * encoded.stat().st_size / length,
                           original_slice_sha256=reference, restored_slice_sha256=digest(decoded),
                           elapsed_seconds=time.time() - row["started_at"])
                save(receipt, row)
                event("trial_passed", sample=name, variant=variant,
                      encoded_bytes=row["encoded_bytes"], seconds=row["elapsed_seconds"])
                # The verified compressed file, digests and logs retain the evidence.
                decoded.unlink()
            except Exception as error:
                row.update(status="failed", error=str(error), elapsed_seconds=time.time() - row["started_at"])
                save(receipt, row)
                event("trial_failed", sample=name, variant=variant, error=str(error))
                raise
        base = json.loads((out / name / args.variants[0] / "result.json").read_text())
        for variant in args.variants[1:]:
            candidate = json.loads((out / name / variant / "result.json").read_text())
            event("comparison", sample=name, baseline=args.variants[0], candidate=variant,
                  baseline_bytes=base["encoded_bytes"], candidate_bytes=candidate["encoded_bytes"],
                  saved_bytes=base["encoded_bytes"] - candidate["encoded_bytes"],
                  saved_fraction=1 - candidate["encoded_bytes"] / base["encoded_bytes"])
    event("batch_complete", samples=len(plan))


if __name__ == "__main__":
    main()
