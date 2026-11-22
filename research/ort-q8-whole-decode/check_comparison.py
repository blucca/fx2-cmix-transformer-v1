#!/usr/bin/env python3
"""Small oracle: a failing stock mode must fail the complete numerical gate."""
from pathlib import Path
import tempfile
import native_runner as study

# TMPDIR can point to your scratch directory; no model or kernel is loaded.
with tempfile.TemporaryDirectory(prefix="q8-comparison-") as tmp:
    root = Path(tmp)
    names = ("dense-before", "native", "q8_acc0", "split8", "dense-after")
    for name in names:
        study.dump(root / "runs/check" / name / "result.json", {
            "status": "completed", "all_outputs_passed": True, "peak_rss_kib": 1,
            "host": {"features": {"avx2": True, "fma": True, "avx512f": False,
                                  "avx512_vnni": False, "avx_vnni": False}},
            "rows": [{"p50_ms": 8.0 if name == "native" else 10.0,
                      "check": {"all_outputs_passed": True}} for _ in study.PAST]})
    assert study.compare(root, "check")["target_gate_passed"]
    path = root / "runs/check/q8_acc0/result.json"
    import json
    report = json.loads(path.read_text())
    report["all_outputs_passed"] = False
    report["rows"][0]["check"]["all_outputs_passed"] = False
    study.dump(path, report)
    result = study.compare(root, "check")
    assert result["performance_pass"] and not result["numerical_pass"]
    assert not result["target_gate_passed"] and not result["precision_by_mode"]["q8_acc0"]
print("COMPARISON_ORACLE_PASS")
