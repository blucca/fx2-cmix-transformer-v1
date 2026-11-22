#!/usr/bin/env python3
"""Pure NumPy layout oracle for the fixed tied-vocab mechanism; no ORT/ONNX.

Runs the existing converter's quant8/dequant function bodies in an isolated
NumPy-only namespace, then checks their layout against independent Gather,
native-pack address, zero-padding and logit-slice equations. No source model,
native library, download, framework construction or product writes.
"""
import ast
import json
from pathlib import Path
import time

import numpy as np

N, NPAD, K, BLOCK = 50257, 50272, 768, 32
KB = K // BLOCK
CONVERTER = Path(__file__).resolve().parent.parent / "ort-q8-whole-decode/converter.py"


def live_quant_functions():
    tree = ast.parse(CONVERTER.read_text())
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name in {"quant8", "dequant"}]
    assert len(selected) == 2
    namespace = {"np": np, "BLOCK": BLOCK}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(CONVERTER), "exec"), namespace)
    return namespace["quant8"], namespace["dequant"]


def main():
    began = time.perf_counter()
    quant8, dequant = live_quant_functions()
    assert NPAD % 16 == 0 and K % BLOCK == 0 and NPAD - N == 15

    # Small exact-dyadic, column/block-distinct weights. A zero row and a zero
    # block exercise the converter's positive absmax floor without a full W.
    ns = 17
    inner = np.arange(BLOCK, dtype=np.int32)[None, :, None]
    block = np.arange(KB, dtype=np.int32)[:, None, None]
    column = np.arange(ns, dtype=np.int32)[None, None, :]
    q = ((inner * 19 + block * 7 + column * 11) % 255 - 127).astype(np.float32)
    q[:, 0, :] = -127
    q[:, -1, :] = 127
    wanted_scale = ((np.arange(KB, dtype=np.float32)[:, None] % 4 + 1)
                    * (np.arange(ns, dtype=np.float32)[None, :] % 8 + 1)
                    / np.float32(1024))
    w = np.ascontiguousarray((q * wanted_scale[:, None, :]).reshape(K, ns))
    w[:, 2] = 0
    w[5 * BLOCK:6 * BLOCK, 3] = 0
    qb, qs, qdeq = quant8(w)
    floor = np.float32(1e-8) / np.float32(127)
    assert np.all(qb[2] == 128) and np.all(qs.reshape(ns, KB)[2] == floor)
    assert np.all(qb[3, 5] == 128) and qs.reshape(ns, KB)[3, 5] == floor

    # N padding is independent of per-column/K-block quantization.
    small_pad_w = np.pad(w, ((0, 0), (0, 32 - ns)))
    pb, ps, pdeq = quant8(small_pad_w)
    np.testing.assert_array_equal(pb[:ns], qb)
    np.testing.assert_array_equal(ps.reshape(32, KB)[:ns], qs.reshape(ns, KB))
    np.testing.assert_array_equal(pdeq[:, :ns], qdeq)
    assert np.all(pb[ns:] == 128) and np.all(ps.reshape(32, KB)[ns:] > 0)
    assert np.all(pdeq[:, ns:] == 0)

    # Full target index/shape but only 43.4 MB B/S, and no dense full-vocab W.
    b = np.full((NPAD, KB, BLOCK), 128, dtype=np.uint8)
    s = np.full(NPAD * KB, floor, dtype=np.float32)
    placements = {0: 0, 1: 1, 2: 2, 15: 3, 16: 4, 37: 5, N - 2: 6, N - 1: 7}
    for row, source_column in placements.items():
        b[row] = qb[source_column]
        s.reshape(NPAD, KB)[row] = qs.reshape(ns, KB)[source_column]

    def gather(ids):
        # ONNX Gather(data, ids, axis=0); both Cast/Sub and Mul are FLOAT.
        assert ids.dtype == np.int64 and ids.shape == (1, 1)
        codes = np.take(b, ids, axis=0).astype(np.float32) - np.float32(128)
        scales = np.take(s.reshape(NPAD, KB, 1), ids, axis=0)
        assert codes.shape == (1, 1, KB, BLOCK)
        assert scales.shape == (1, 1, KB, 1)
        result = codes * scales
        assert result.dtype == np.float32
        return result.reshape(1, 1, K)

    checked = []
    for row, source_column in placements.items():
        ids = np.array([[row]], dtype=np.int64)
        assert 0 <= row < N
        actual = gather(ids)
        expected = np.ascontiguousarray(qdeq[:, source_column]).reshape(1, 1, K)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        checked.append(row)
    for pad_row in (N, NPAD - 1):
        assert np.all(gather(np.array([[pad_row]], dtype=np.int64)) == 0)
    assert np.all(b[N:] == 128) and np.all(s.reshape(NPAD, KB)[N:] > 0)

    # Fault-sensitive: flattened-scale Gather and uint8 subtraction both
    # change this actual selected row. These are layout algebra diagnostics.
    ids = np.array([[N - 1]], dtype=np.int64)
    correct = gather(ids)
    wrong_scalar = ((np.take(b, ids, axis=0).astype(np.float32) - np.float32(128))
                    * np.take(s, ids, axis=0)[..., None, None]).reshape(1, 1, K)
    assert not np.array_equal(correct, wrong_scalar)
    wrong_uint8 = (((np.take(b, ids, axis=0) - np.uint8(128)).astype(np.float32))
                   * np.take(s.reshape(NPAD, KB, 1), ids, axis=0)).reshape(1, 1, K)
    assert not np.array_equal(correct, wrong_uint8)

    # Mirror only the existing kernel's index equations for first and final
    # 16-lane tiles, not its execution or float32/FMA error. Dyadic operands
    # make the mathematical comparison exact in float64.
    a = ((np.arange(K, dtype=np.int32) % 13 - 6) / 32).astype(np.float32)
    max_dot_error = 0.0
    for tile in (0, NPAD // 16 - 1):
        packed = np.empty((KB, BLOCK, 16), dtype=np.int8)
        packed_s = np.empty((KB, 16), dtype=np.float32)
        for block_index in range(KB):
            for lane in range(16):
                row = tile * 16 + lane
                for r in range(BLOCK):
                    source = (row * KB + block_index) * BLOCK + r
                    packed[block_index, r, lane] = int(b.reshape(-1)[source]) - 128
                packed_s[block_index, lane] = s[row * KB + block_index]
        native_equation = np.zeros(16, dtype=np.float64)
        for block_index in range(KB):
            dot = a.reshape(KB, BLOCK)[block_index].astype(np.float64) @ packed[block_index].astype(np.float64)
            native_equation += dot * packed_s[block_index].astype(np.float64)
        dense_equation = np.array([
            a.astype(np.float64) @ gather(np.array([[tile * 16 + lane]], dtype=np.int64)).reshape(K).astype(np.float64)
            for lane in range(16)
        ])
        np.testing.assert_array_equal(native_equation, dense_equation)
        max_dot_error = max(max_dot_error, float(np.max(np.abs(native_equation - dense_equation))))
        if tile == NPAD // 16 - 1:
            assert tile * 16 == N - 1
            assert np.all(native_equation[1:] == 0)  # Exactly 15 padding lanes.

    # Output Slice: zero padding must leave the original vocab before token
    # selection/NLL. With negative real logits, an omitted slice selects pad.
    y2d = -np.ones((1, NPAD), dtype=np.float32)
    y2d[:, N:] = 0
    logits = y2d[:, 0:N:1].reshape(1, 1, N)
    assert logits.shape == (1, 1, N) and int(np.argmax(logits[0, 0])) < N
    assert int(np.argmax(y2d[0])) >= N

    result = {
        "status": "PASS", "scope": "NumPy quant/layout/padding/Gather/Slice/index equations only",
        "N": N, "Npad": NPAD, "padding_rows": NPAD - N,
        "K": K, "K_blocks": KB,
        "B_shape": list(b.shape), "S_shape": list(s.shape),
        "B_S_payload_bytes": b.nbytes + s.nbytes,
        "gather_legal_rows_checked": checked,
        "padding_rows_checked": [N, NPAD - 1],
        "gather_dequant_prefix_bits_equal": True,
        "padding_does_not_change_original_quantization": True,
        "flattened_scale_and_uint8_subtraction_faults_detected": True,
        "native_index_equation_max_abs_error_exact_dyadic_case": max_dot_error,
        "pad_logits_excluded_before_argmax": True,
        "actual_ONNX_or_native_execution": False,
        "elapsed_seconds": time.perf_counter() - began,
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
