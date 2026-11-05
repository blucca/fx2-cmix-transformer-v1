#!/usr/bin/env python3
"""One tiny fixture for native section layout, headers and complete two-copy S."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--work', type=Path, required=True)
root = p.parse_args().work.resolve()
root.mkdir(parents=True, exist_ok=False)
parts = {'binary': b'B' * 11, 'compressed-dictionary': b'D' * 13,
         'compressed-order': b'O' * 17, 'weights': b'W' * 19,
         'raw-payload': b'P' * 23, 'vocab': bytes(range(32))}
command = [sys.executable, str(Path(__file__).with_name('archive-package.py').resolve())]
for name, data in parts.items():
    path = root / name
    path.write_bytes(data)
    command += ['--' + name, str(path)]
command += ['--prepared-bytes', '10000', '--out', str(root / 'package')]
subprocess.run(command, check=True)
cmix = (root / 'package/cmix').read_bytes()
archive = (root / 'package/archive9').read_bytes()
assert cmix == b'B' * 11 + b'D' * 13 + b'O' * 17 + b'W' * 19 + struct.pack('<4i', 13, 17, 0, 19)
header = (10000 | (1 << 39)).to_bytes(5, 'big') + bytes(range(32))
assert archive == b'B' * 11 + b'D' * 13 + b'W' * 19 + header + b'P' * 23 + struct.pack('<4i', 13, 17, 60, 19)
result = json.loads((root / 'package/package.json').read_text())
assert result['full_s'] == len(cmix) + len(archive) == 195
assert result['stream_header_bytes'] == 37 and result['selfextract_header_bytes'] == 32
print('original-section-layout-two-copy-S-ok')
