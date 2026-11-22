#!/usr/bin/env python3
"""Fixed complete graph + full persisted code/scale byte check, with no inference."""
import argparse
import filecmp
from pathlib import Path
import resource
import time

import convert as c


def check(source, target):
    start = time.monotonic()
    source_preflight = c.preflight_files(source)
    target_preflight = c.preflight_files(target, packed=True)
    original = c.onnx.load(source, load_external_data=False)
    rows = c.inspect_source(original)
    graph = c.rewrite(original, rows)
    saved = c.onnx.load(target, load_external_data=False)
    assert saved.SerializeToString() == original.SerializeToString(), "fixed graph/initializer/IO metadata changed"
    c.onnx.checker.check_model(str(target))
    expected_bins = set(c.fixed_source_files())
    for row in rows:
        expected_bins.remove(row["B"] + ".bin")
        expected_bins.remove(row["S"] + ".bin")
        expected_bins.update([row["P"] + ".bin", row["PS"] + ".bin"])
        row["byte_checks"] = c.check_pair(source.parent, target.parent, row)
    assert {p.name for p in target.parent.glob("*.bin")} == expected_bins, "extra or obsolete weight table"
    unchanged = []
    for tensor in saved.graph.initializer:
        if c.uses_external_data(tensor) and tensor.name not in {r[x] for r in rows for x in ("P", "PS")}:
            name, offset, length = c.ext_signature(tensor)
            assert filecmp.cmp(source.parent / name, target.parent / name, shallow=False), name
            unchanged.append({"file": name, "bytes": length, "bytewise_equal": True})
    return {"status": "PASS_complete_fixed_graph_and_persisted_weights", "source": str(source), "model": str(target),
            "preflight_before_each_model_parse": [source_preflight, target_preflight],
            "equations": c.check_equations(), "graph": graph, "all_25_pairs": rows,
            "all_P_equation_bytes": sum(r["P_bytes"] for r in rows),
            "all_PS_bitwise_bytes": sum(r["PS_bytes"] for r in rows),
            "unchanged_external_files": unchanged, "extra_full_table_or_old_B_S_files": 0,
            "quantizer_calls": 0, "native_or_model_inference_runs": 0,
            "resource_receipt": {"wall_seconds": time.monotonic() - start,
                                 "peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    result = check(args.source.resolve(), args.model.resolve())
    c.write_json(args.receipt.resolve(), result)
    print(f"{result['status']}: 25 pairs / {result['all_P_equation_bytes']} code bytes / {result['all_PS_bitwise_bytes']} scale bytes / 13 outputs")
