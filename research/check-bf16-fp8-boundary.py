#!/usr/bin/env python3
"""One finite E4M3FN rounding cell, RNE; a precision-contract counterexample.

The local FP8 rule covers [1.0, 1.125]. Use a device converter for broader
format/overflow coverage. This check needs only the standard library.
"""
import json
import struct

x = struct.unpack("<f", struct.pack("<f", 1.0626))[0]
u = struct.unpack("<I", struct.pack("<f", x))[0]
# BF16 round-to-nearest-even: discarded 16 bits plus retained least bit.
bits = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
b = struct.unpack("<f", struct.pack("<I", bits))[0]

def fp8_cell(y):
    assert 1.0 <= y <= 1.125
    # E4M3FN codes 0x38 and 0x39; the midpoint tie rounds to even 0x38.
    return 1.125 if y > 1.0625 else 1.0

direct, via_bf16 = fp8_cell(x), fp8_cell(b)
assert (b, direct, via_bf16) == (1.0625, 1.125, 1.0)
print(json.dumps({"fp32_input": x, "bf16": b, "fp32_to_fp8": direct,
                  "fp32_to_bf16_to_fp8": via_bf16,
                  "scope": "finite positive E4M3FN cell; round-to-nearest-even"}, indent=2))
