#!/usr/bin/env python3
"""One eight-byte exact range fixture; usage: PROBE OUTPUT_DIRECTORY."""
import json
from pathlib import Path
import struct
import subprocess
import sys

probe, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
vocab = bytearray(32)
for byte in range(205):
    vocab[byte // 8] |= 1 << (byte % 8)
(out / "vocab.bin").write_bytes(vocab)
(out / "tokens.u8").write_bytes(bytes([2, 3, 4, 5]))
with (out / "trace.bin").open("wb") as f:
    for byte in range(8):
        for shift in range(7, -1, -1):
            f.write(struct.pack("<HHfff", 32768, (byte >> shift) & 1, .5, .5, 0.))
result = json.loads(subprocess.check_output(list(map(str, [probe, out / "trace.bin",
    out / "tokens.u8", out / "vocab.bin", 2, 9])), text=True))
assert result["full_coded_bytes"] == 9 and result["interval_coded_bytes"] == 4
assert result["neural_targets"]["bytes"] == 3
assert result["neural_targets"]["neural_bits"] == 24
assert result["neural_targets"]["final_u16_bits"] == 24
(out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print("mixed interval fixture passed: full=9B, retained=4B, NN targets=3")
