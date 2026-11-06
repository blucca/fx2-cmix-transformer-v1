#!/usr/bin/env python3
"""Replay the frozen v22 output bound and two-field affine-anchor contract.

Source assertions are intentionally fixed to the measured v22 revision. A new
family requires reviewing its producer counts before reusing these conclusions.
This check runs no neural model, coder, or online FXCM state.
"""
import argparse
import json
import math
from pathlib import Path
import re
import struct
import subprocess

REV = "83f2603f3f7751da5f429fd32669807eb8510494"


def check(source):
    assert "#define VERSION 22" in source
    size_expr = re.search(r"int num_models = ([0-9+ -]+);", source)[1]
    size = sum(int(x) for x in re.findall(r"[+-]?\s*\d+", size_expr.replace(" ", "")))
    contexts = []
    for match in re.finditer(r"(cmC[12]?\[\d+\])\.Init\(([^;]+)\);", source):
        args = match[2].split(",")
        assert len(args) == 8
        count = int(args[1].split("|", 1)[0])
        skip = int(args[6])
        assert 1 <= count <= 8 and skip in (0, 1)
        contexts.append(dict(name=match[1], contexts=count, skip2=skip))
    body = source[source.index("int modelPrediction("):source.index("int rate=6;")]
    used = re.findall(r"(cmC[12]?\[\d+\])\.mix\(\)", body)
    assert len(contexts) == len(used) == 31
    assert sorted(row["name"] for row in contexts) == sorted(used)
    # Each of the three CM classes has the same output count in populated,
    # zero-state and disabled branches: 4+skip2, including the run prediction.
    for name, next_name in (("ContextMap", "ContextMap1"),
                            ("ContextMap1", "ContextMap2"),
                            ("ContextMap2", "// APM maps")):
        start = source.index("struct " + name + " {")
        stop = source.index("struct " + next_name + " {", start) if next_name.startswith("Context") else source.index(next_name, start)
        text = source[start:stop]
        assert "C=c&255" in text and "int i=cn++" in text and "i<C" in text
        assert "skip2=u" in text and "if (skip2==1)" in text
        assert text.count("prediction_index--") == 3
    nst = int(re.search(r"const int nST=(\d+);", source)[1])
    assert nst == 3 and "for( int i = 0; i < nST; i++ )" in source
    assert len(re.findall(r"scmA\[\d+\]\.mix\(sscmrate\)", body)) == 7
    assert len(re.findall(r"x\.mxInputs2\.add\(mxA\[\d+\]\.p1\(\)\)", body)) == 10
    assert "x.mxInputs1.add(stretch(lstmpr)); prediction_index--;" in body
    assert "x.mxInputs2.add(stretch(lstmpr)/2); prediction_index--;" in body
    components = dict(contexts=sum((4 + row["skip2"]) * row["contexts"] for row in contexts),
                      stationary=7, match=1 + 2 * nst, sparse_match=2,
                      run=1, bias=1, first_level_mixes=10, apm=7)
    maximum = sum(components.values())
    assert components["contexts"] == 394 and maximum == 429 and size == 431
    # The initialization is 0.5, and writes always start from index zero.
    assert "model_predictions(0.5f, num_models)" in source
    assert "model_predictions[prediction_index++]" in source
    assert "prediction_index = 0;" in source
    assert "return fxcmv1::model_predictions;" in source

    # Two stored floats suffice when they are (intercept, slope). x0 is read
    # while the collector has the actual native bit probability, then folded.
    logit = lambda p: math.log(p / (1 - p))
    def anchor(p0, z0, slope, override=False):
        if override:
            return z0, 0.0
        assert 0 < p0 < 1
        return z0 - slope * logit(p0), slope
    for probability in (0.0, 1.0):
        assert anchor(probability, 1.1, -.8, override=True) == (1.1, 0.0)
    x0, z0, slope = logit(.25), 1.1, -.8
    intercept, slope = anchor(.25, z0, slope)
    assert abs((intercept + slope * x0) - z0) < 1e-15
    packed_b, packed_s = struct.unpack("<ff", struct.pack("<ff", intercept, slope))
    residual = abs((packed_b + packed_s * x0) - z0)
    assert residual < 1e-7
    # (Z0, slope) requires its x0 or a same-W surrogate reference recomputation.
    ambiguous = abs(slope * (logit(.2) - logit(.7)))
    assert ambiguous > 1
    count = 4_023_812
    stats_bytes = count * 8 * struct.calcsize("<ff")
    assert stats_bytes == 257_523_968
    return dict(schema="hutter.v22-local-slope-source.v1", source_revision=REV,
                output_array_size=size, maximum_written=maximum,
                constant_padding_indices=list(range(maximum, size)),
                prediction_components=components, context_arrays=contexts,
                smooth_affine_anchor=dict(fields=["intercept", "slope"],
                  formula="eta=intercept+slope*x", intercept="Z0-slope*x0",
                  fixture_fp32_anchor_error=residual,
                  fixture_z0_slope_only_intercept_spread=ambiguous,
                  selected_targets=count, eight_bit_stats_bytes=stats_bytes),
                model_calls=0, codec_calls=0, passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.source:
        source = args.source.read_text()
    else:
        repo = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(["git", "show", REV + ":src/models/fxcmv1.cpp"], cwd=repo, text=True)
    print(json.dumps(check(source), indent=2))


if __name__ == "__main__":
    main()
