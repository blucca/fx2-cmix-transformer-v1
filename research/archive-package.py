#!/usr/bin/env python3
"""Build the pinned compressor, or measure its authentic native archive.

The retained prepared-payload mode assembles both x86 self-extractors.
Compressor-only mode bootstraps a whole ``cmix -e`` run; native-output mode
checks that run's stream header, appended sections and actual two-file S.
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
                 'vocab', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--raw-payload', type=Path)
    mode.add_argument('--native-output', type=Path)
    mode.add_argument('--compressor-only', action='store_true')
    p.add_argument('--prepared-bytes', type=int, required=True)
    a = p.parse_args()
    binary, dictionary, order, weights = [path.resolve() for path in
        [a.binary, a.compressed_dictionary, a.compressed_order, a.weights]]
    sizes = {name: path.stat().st_size for name, path in zip(
        ['binary', 'compressed_dictionary', 'compressed_order', 'weights'],
        [binary, dictionary, order, weights])}
    # The pinned x86 runner stores HeaderInfo as four little-endian int32s.
    # The full enwik9 stream uses the 37-byte WriteHeader branch (length >=10k).
    assert all(0 < size < 1 << 31 for size in sizes.values())
    assert 10000 <= a.prepared_bytes < 1 << 39
    vocab = a.vocab.read_bytes()
    assert len(vocab) == 32
    stream_header = (a.prepared_bytes | (1 << 39)).to_bytes(5, 'big') + vocab
    d, o, w = sizes['compressed_dictionary'], sizes['compressed_order'], sizes['weights']
    out = a.out.resolve()
    if a.native_output:
        assert out.is_dir()
        compressed_input_size = a.native_output.stat().st_size
        sizes['raw_payload'] = compressed_input_size - len(stream_header)
        assert sizes['raw_payload'] > 0 and compressed_input_size < 1 << 31
        with a.native_output.open('rb') as source:
            assert source.read(len(stream_header)) == stream_header, 'native prepared header changed'
        with (out / 'cmix').open('rb') as source:
            for path in [binary, dictionary, order, weights]:
                with path.open('rb') as section:
                    for block in iter(lambda: section.read(1 << 20), b''):
                        assert source.read(len(block)) == block, 'native compressor section changed'
            assert source.read() == struct.pack('<4i', d, o, 0, w)
        with (out / 'archive9').open('rb') as source:
            for path in [binary, dictionary, weights, a.native_output]:
                with path.open('rb') as section:
                    for block in iter(lambda: section.read(1 << 20), b''):
                        assert source.read(len(block)) == block, 'native archive section changed'
            assert source.read() == struct.pack('<4i', d, o, compressed_input_size, w)
    else:
        out.mkdir(parents=True, exist_ok=False)
        with (out / 'cmix').open('xb') as output:
            append(output, [binary, dictionary, order, weights])
            output.write(struct.pack('<4i', d, o, 0, w))
        (out / 'cmix').chmod(0o755)
        if a.compressor_only:
            result = dict(status='compressor_ready', components=sizes,
                          compressor_bytes=(out / 'cmix').stat().st_size,
                          next_step='run ./cmix -e ORIGINAL native-output from this directory')
            print(json.dumps(result))
            return
        payload = a.raw_payload.resolve()
        sizes['raw_payload'] = payload.stat().st_size
        assert 0 < sizes['raw_payload'] < 1 << 31
        compressed_input_size = sizes['raw_payload'] + len(stream_header)
        assert compressed_input_size < 1 << 31
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
    result = dict(status='native_measured' if a.native_output else 'assembled',
                  prepared_bytes=a.prepared_bytes, components=sizes,
                  stream_header_bytes=len(stream_header), selfextract_header_bytes=32,
                  compressor_bytes=compressor_bytes, archive_bytes=archive_bytes, full_s=full_s,
                  verification='actual archive9 decode is the next stage')
    (out / 'package.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
