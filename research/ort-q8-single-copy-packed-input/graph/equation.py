#!/usr/bin/env python3
"""The fixed block32/tile16 layout; scalar equations are the independent check."""
import json
from pathlib import Path
import argparse

import numpy as np

BLOCK, TILE = 32, 16


def pack_codes(b):
    n, kb, inner = b.shape
    assert b.dtype == np.uint8 and inner == BLOCK and n % TILE == 0
    return np.ascontiguousarray((b.astype(np.int16) - 128).astype(np.int8)
                               .reshape(n // TILE, TILE, kb, BLOCK)
                               .transpose(0, 2, 3, 1))


def pack_scales(s, n, kb):
    assert s.dtype == np.float32 and s.shape == (n * kb,)
    return np.ascontiguousarray(s.reshape(n // TILE, TILE, kb).transpose(0, 2, 1))


def packed_embedding(p, ps, token):
    """Exactly the scalar-id, two-level Gather graph, without a full-table view."""
    tile, lane = int(token) // TILE, int(token) % TILE
    codes = np.take(np.take(p, tile, axis=0), lane, axis=2)
    scales = np.take(np.take(ps, tile, axis=0), lane, axis=1)
    return (codes.astype(np.float32) * scales.reshape(-1, 1)).reshape(1, 1, -1)


def check_equations():
    n, kb = 64, 3
    real_n = n - 15
    b = (np.arange(n * kb * BLOCK, dtype=np.int64) % 256).astype(np.uint8).reshape(n, kb, BLOCK)
    b[real_n:] = 128
    s = ((np.arange(n * kb, dtype=np.float32) + 1) / np.float32(128)).reshape(-1)
    assert len(np.unique(b[:real_n])) == 256
    p, ps = pack_codes(b), pack_scales(s, n, kb)
    assert p.dtype == np.int8 and p.flags.c_contiguous
    assert ps.dtype == np.float32 and ps.flags.c_contiguous
    # This loop is deliberately independent of reshape/transpose implementation.
    for tile in range(n // TILE):
        for block in range(kb):
            for lane in range(TILE):
                row = tile * TILE + lane
                scale_offset = (tile * kb + block) * TILE + lane
                assert ps.reshape(-1).view(np.uint32)[scale_offset] == s.view(np.uint32)[row * kb + block]
                for inner in range(BLOCK):
                    packed_offset = ((tile * kb + block) * BLOCK + inner) * TILE + lane
                    assert int(p.reshape(-1)[packed_offset]) == int(b[row, block, inner]) - 128
    tokens = [0, 1, 15, 16, 17, 31, 32, real_n - 1, real_n, n - 1]
    for token in tokens:
        old = ((b[token].astype(np.float32) - np.float32(128))
               * s.reshape(n, kb)[token, :, None]).reshape(1, 1, kb * BLOCK)
        new = packed_embedding(p, ps, token)
        assert np.array_equal(old.view(np.uint32), new.view(np.uint32)), token
    for token in range(real_n, n):
        assert np.all(packed_embedding(p, ps, token) == 0), token
    assert np.isfinite(ps).all() and np.all(ps > 0)
    assert {int(x) for x in p.reshape(-1)} == set(range(-128, 128))
    return {
        "status": "PASS_equation_sign_scales_embedding_padding",
        "synthetic_N": n, "synthetic_K": kb * BLOCK,
        "codes_checked": int(b.size), "all_256_source_codes": True,
        "signed_range": [-128, 127], "zero_128_maps_to_0": True,
        "scales_distinct_tile_block_lane_and_bits_preserved": True,
        "scalar_native_offsets_checked": True,
        "embedding_checked_tokens": tokens, "embedding_FP32_bits_equal": True,
        "padding_rows": 15, "padding_embedding_zero": True,
        "weight_quantizer_calls": 0, "model_inference_runs": 0,
        "P_equation": "P[t,b,i,l] = INT8(INT16(B[t*16+l,b,i])-128)",
        "PS_equation": "PS[t,b,l] = S[(t*16+l)*KB+b]",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = check_equations()
    data = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.receipt:
        args.receipt.write_text(data)
    print(data, end="")
