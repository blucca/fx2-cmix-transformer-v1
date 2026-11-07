#!/usr/bin/env python3
"""RNE precision control for positive finite E4M3FN values in [1, 2].

The band converter uses its nine exact FP8 points. Use a device converter
for other bands/overflow coverage. No model or package dependency.
"""
import json
import struct


def bf16_rne(x):
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def fp8_band(y):
    assert 1.0 <= y <= 2.0
    # Nine E4M3FN points, 1+k/8. Python round supplies the even midpoint tie.
    return 1.0 + round(8 * (y - 1.0)) / 8


x = struct.unpack("<f", struct.pack("<f", 1.0626))[0]
b = bf16_rne(x)
direct, via_bf16 = fp8_band(x), fp8_band(b)
assert (b, direct, via_bf16) == (1.0625, 1.125, 1.0)

# Every 1+i/65536 is exactly FP32. This is a stated uniform control band.
n = 1 << 16
changed = sum(fp8_band(1 + i/n) != fp8_band(bf16_rne(1 + i/n)) for i in range(n))
assert changed == n // 32
print(json.dumps({"fp32_input": x, "bf16": b, "fp32_to_fp8": direct,
                  "fp32_to_bf16_to_fp8": via_bf16,
                  "uniform_band": {"samples": n, "changed": changed,
                                   "fraction": changed/n},
                  "scope": "finite positive E4M3FN [1,2]; round-to-nearest-even"}, indent=2))
