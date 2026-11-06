#!/usr/bin/env python3
"""Materialize two real native pieces; uniform priors are resource-only."""
from array import array
from pathlib import Path
import hashlib
import json
import os
import resource
import struct
import sys
import time

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepared', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
J = args.out
J.mkdir(parents=True, exist_ok=True)
CAP = 1 << 17
EXPECTED_STREAM = 'a5cef5dfb651f7a12d3c6fec27dbe1e6290bde05ef1b893b24989081857fbead'
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_CPU, (58, 60))
resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
started = time.monotonic()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


out = J / 'inputs'
out.mkdir(exist_ok=True)
assert not (J/'inputs.json').exists(), 'inspect existing input manifest'
prepared = args.prepared
stream = prepared/'stream.bin'
assert stream.stat().st_size == 586459020
stream_sha = digest(stream)
assert stream_sha == EXPECTED_STREAM
vocab = (prepared/'vocab.bin').read_bytes()
assert len(vocab) == 32
members = [byte for byte in range(256) if (vocab[byte//8] >> (byte%8)) & 1]
assert len(members) == 205
mapping = {byte: index for index, byte in enumerate(members)}
bounds = array('i')
bounds.frombytes((prepared/'article_boundaries.i32').read_bytes())
assert sys.byteorder == 'little' and bounds.itemsize == 4
assert bounds[0] == 0 and bounds[1] == 43 and bounds[-1] == stream.stat().st_size
assert all(end > start for start, end in zip(bounds, bounds[1:]))
short = [(abs(end-start-1-1346), index, start, end)
         for index, (start, end) in enumerate(zip(bounds, bounds[1:]))
         if 1025 <= end-start-1 <= 2048]
assert short
_, short_index, short_start, short_end = min(short)
largest = max(range(len(bounds)-1), key=lambda index: bounds[index+1]-bounds[index])
article_start, article_end = bounds[largest], bounds[largest+1]
full_start, full_end = 570595966, 570727038
assert article_start <= full_start < full_end <= article_end
assert (full_start-article_start) % CAP == 0
assert full_end-full_start == CAP
assert [full_start, full_end] == [570595966, 570727038]
pieces = {'short': (short_index, short_start, short_end),
          'full': (largest, full_start, full_end)}
separator = bytes([8, 8, 0x25, 0xac, 0x65, 0x27, 5, 8, 8, 8, 8, 0x25, 0xac, 0x68, 0x27])
cases = {}
for name, (index, start, end) in pieces.items():
    with stream.open('rb') as f:
        f.seek(start)
        raw = f.read(end-start)
    assert len(raw) == end-start
    tokens = bytes(mapping[byte] for byte in raw)
    # A separator at a native piece's end belongs to that piece. Earlier
    # separator ends would create a shorter native piece and are rejected.
    assert tokens.find(separator) in (-1, len(tokens)-len(separator))
    if name == 'short':
        assert tokens.endswith(separator)
    tok_path, prior_path = out/(name+'.u8'), out/(name+'-uniform.f16')
    assert not tok_path.exists() and not prior_path.exists()
    tok_path.write_bytes(tokens)
    row = struct.pack('<e', 1/205)*205
    with prior_path.open('wb') as f:
        for lo in range(0, len(tokens), 4096):
            f.write(row*min(4096, len(tokens)-lo))
    cases[name] = dict(tokens=str(tok_path), priors=str(prior_path), targets=len(tokens)-1,
                       source_begin=start, source_end=end, source_article_index=index,
                       source_article_begin=bounds[index], source_article_end=bounds[index+1],
                       native_piece_limit=CAP,
                       prior_kind='synthetic_uniform_f16_resource_only',
                       uniform_f16_value=struct.unpack('<e', struct.pack('<e', 1/205))[0],
                       tokens_sha256=digest(tok_path), priors_sha256=digest(prior_path))
manifest = dict(schema='hutter.exact-tile-probe-inputs.v1', status='passed',
                source_stream_sha256=stream_sha, source_stream_bytes=stream.stat().st_size,
                source_vocab_sha256=digest(prepared/'vocab.bin'),
                source_bounds_sha256=digest(prepared/'article_boundaries.i32'),
                vocab_mapping='ascending source byte values selected by original membership mask',
                real_matching_prior_assets=0, ppmd_capture_calls=0, compression_quality_evaluation=False,
                cases=cases, cpu_seconds=time.process_time(),
                wall_seconds=time.monotonic()-started,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(J/'inputs.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest), flush=True)
