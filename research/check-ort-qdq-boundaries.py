#!/usr/bin/env python3
"""Rebuild the recorded fixture's FP32 → nearest-even UINT8 → DQ boundaries.

Usage: python3 -B research/check-ort-qdq-boundaries.py research/results/ort-qdq-*.json
The fixture's power-of-two scale gives an exact division in Python float.
"""
import json
import struct
import sys
from pathlib import Path

assert len(sys.argv) > 1
count = 0
for filename in sys.argv[1:]:
    data = json.loads(Path(filename).read_text())
    scale, zero = data['scale'], data['zero_point']
    assert scale == 0.03125 and zero == 128
    assert all(row['original_endpoint_equal'] for row in data['probe_sessions'].values())
    for point in data['quantization_boundaries']:
        for route in ('reference', 'all'):
            value = point[f'pre_q_{route}']
            bits = struct.unpack('<I', struct.pack('<f', value))[0]
            assert bits == int(point[f'{route}_fp32_hex'], 16)
            quantized = round(value / scale) + zero
            assert 0 <= quantized <= 255  # These recorded crossings are inside the clamp.
            assert quantized == point[f'{route}_uint8']
            assert (quantized - zero) * scale == point[f'{route}_output']
        assert abs(point['all_output'] - point['reference_output']) == scale
        count += 1
    print(data['machine'], data['runtime_version'], len(data['quantization_boundaries']), 'boundaries passed')
print('Total checked boundaries:', count)
