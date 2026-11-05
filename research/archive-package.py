#!/usr/bin/env python3
"""Assemble the original x86 enwik9 self-extractors from measured components.

Consumes an untraced full-stream probe payload and already cmix-compressed
dictionary/order files. This does no coding or decoding; archive9's real 1GB
round trip and the final program's resource measurements are separate stages.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct


def append(output, paths):
    for path in paths:
        with path.open('rb') as source:
            shutil.copyfileobj(source, output, 1 << 20)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['binary', 'compressed-dictionary', 'compressed-order', 'weights',
                 'raw-payload', 'vocab', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--prepared-bytes', type=int, required=True)
    a = p.parse_args()
    binary, dictionary, order, weights, payload = [path.resolve() for path in
        [a.binary, a.compressed_dictionary, a.compressed_order, a.weights, a.raw_payload]]
    sizes = {name: path.stat().st_size for name, path in zip(
        ['binary', 'compressed_dictionary', 'compressed_order', 'weights', 'raw_payload'],
        [binary, dictionary, order, weights, payload])}
    # The pinned x86 runner stores HeaderInfo as four little-endian int32s.
    # The full enwik9 stream uses the 37-byte WriteHeader branch (length >=10k).
    assert all(0 < size < 1 << 31 for size in sizes.values())
    assert 10000 <= a.prepared_bytes < 1 << 39
    vocab = a.vocab.read_bytes()
    assert len(vocab) == 32
    stream_header = (a.prepared_bytes | (1 << 39)).to_bytes(5, 'big') + vocab
    d, o, w = sizes['compressed_dictionary'], sizes['compressed_order'], sizes['weights']
    compressed_input_size = sizes['raw_payload'] + len(stream_header)
    assert compressed_input_size < 1 << 31
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'cmix').open('xb') as output:
        append(output, [binary, dictionary, order, weights])
        output.write(struct.pack('<4i', d, o, 0, w))
    with (out / 'archive9').open('xb') as output:
        append(output, [binary, dictionary, weights])
        output.write(stream_header)
        append(output, [payload])
        # The upstream archive preserves o in its header while omitting O in its body.
        output.write(struct.pack('<4i', d, o, compressed_input_size, w))
    for name in ['cmix', 'archive9']:
        (out / name).chmod(0o755)
    compressor_bytes = (out / 'cmix').stat().st_size
    archive_bytes = (out / 'archive9').stat().st_size
    full_s = compressor_bytes + archive_bytes
    expected = (sizes['raw_payload'] + 2 * sizes['binary'] + 2 * d + o + 2 * w + 69)
    assert full_s == expected
    result = dict(status='assembled', prepared_bytes=a.prepared_bytes, components=sizes,
                  stream_header_bytes=len(stream_header), selfextract_header_bytes=32,
                  compressor_bytes=compressor_bytes, archive_bytes=archive_bytes, full_s=full_s,
                  verification='actual archive9 decode is the next stage')
    (out / 'package.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
