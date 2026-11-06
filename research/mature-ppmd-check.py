#!/usr/bin/env python3
"""Check retained PPMd rows against one continuous 64 KiB capture.

Usage: mature-ppmd-check.py PROBE INPUT VOCAB OUTPUT_DIRECTORY
The real production PPMd runs twice; no model or probability mock is used.
"""
import json
from pathlib import Path
import resource
import subprocess
import sys


def main():
    probe, source, vocab, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, offset, length, warm in (("full", 0, 65536, []),
                                      ("retained", 32768, 32768, [0])):
        tokens, priors = output / (name + ".u8"), output / (name + ".f16")
        cmd = [probe, "capture", source, tokens, priors, vocab, offset, length, *warm]
        run = subprocess.run(list(map(str, cmd)), text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        (output / (name + ".log")).write_text(run.stdout)
        run.check_returncode()
        result, = [json.loads(line.split(" ", 1)[1]) for line in run.stdout.splitlines()
                   if line.startswith("ACTIVATION_RESULT ")]
        rows[name] = result
    assert (output / "retained.u8").read_bytes() == (output / "full.u8").read_bytes()[32768:]
    assert (output / "retained.f16").read_bytes() == (output / "full.f16").read_bytes()[32768 * 410:]
    assert rows["retained"]["prior_bytes"] == 32768 * 410
    assert rows["retained"]["warmup_bytes"] == 32768
    assert rows["retained"]["processed_bytes"] == 65536
    invalid = subprocess.run(list(map(str, [probe, "capture", source,
        output / "invalid.u8", output / "invalid.f16", vocab, 0, 2, 1])),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert invalid.returncode != 0 and not (output / "invalid.u8").exists()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = dict(status="passed", retained_rows_byte_identical=True,
                   invalid_warm_start_rejected=True, capture_results=rows,
                   child_cpu_seconds=usage.ru_utime + usage.ru_stime)
    (output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
