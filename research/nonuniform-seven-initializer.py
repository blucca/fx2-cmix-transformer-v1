"""One physical nonuniform initialization; reuse the original RAW/v2 readers.

The AST import selects the existing RAW reader/writer and dtype declarations
without importing model/framework dependencies. The original v2 reader and
original compiled C3 replacement recoder perform the actual format work.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path
import struct
import sys
import types

ROOT = Path('/data/workspace/temp/hutter')
J = ROOT/'round484/q7-teacher'
SOURCE = J/'source/pysrc'
C2 = ROOT/'candidate/models/6m-q4-fp32.tfwc2'
DERIVED = {'rope.inv_freq', 'rope.sin', 'rope.cos', 'config.ints', 'config.kimi'}
EXPECTED_HASHES = {
    'export_weights.py': 'bac19094137e6cea761bf336f178ba55dfee2a12a6cd0dd1d4a88fd95c6df817',
    'weights_compress.py': 'eb3a5404886f4cc8b67a0197385ef1f0f220721707f269fa54870f28ba53e6c8',
}


def scale_bits(bits, numerator, denominator):
    """Exact rational multiplication followed by normal-BF16 RNE."""
    exponent = (bits >> 7) & 255
    assert 0 < exponent < 255 and 0 < numerator < denominator
    n = (128 + (bits & 127))*numerator
    d = denominator
    while n < 128*d:
        n *= 2
        exponent -= 1
    while n >= 256*d:
        d *= 2
        exponent += 1
    significand, remainder = divmod(n, d)
    significand += 2*remainder > d or (2*remainder == d and significand & 1)
    if significand == 256:
        significand = 128
        exponent += 1
    assert 0 < exponent < 255 and 128 <= significand <= 255
    return (bits & 0x8000) | (exponent << 7) | (significand-128)


def actual_mapping(before, after):
    assert (before & 0x8000) == (after & 0x8000)
    e0, e1 = (before >> 7) & 255, (after >> 7) & 255
    num = (128 + (after & 127)) << max(e1-e0, 0)
    den = (128 + (before & 127)) << max(e0-e1, 0)
    positive = [min((0, 2, 4, 7), key=lambda k: (abs(q*den-k*num), k)) for q in range(8)]
    mapping = [-positive[-q] if q < 0 else positive[q] for q in range(-7, 8)]
    return mapping, num/den


def bf16(bits):
    return struct.unpack('<f', struct.pack('<I', bits << 16))[0]


def original_readers():
    import numpy as np
    for name, digest in EXPECTED_HASHES.items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == digest, name
    module = types.ModuleType('pysrc.export_weights')
    module.__dict__.update(np=np, struct=struct)
    tree = ast.parse((SOURCE/'export_weights.py').read_text())
    names = {'MAGIC', 'DTYPE_INT8', 'DTYPE_BF16_BITS', 'DTYPE_FLOAT32', 'DTYPE_INT32',
             '_DTYPE_TO_NUMPY', 'read_tensor_file', 'write_tensor_file'}
    nodes = [n for n in tree.body if
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))
             or (isinstance(n, ast.FunctionDef) and n.name in names)]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE/'export_weights.py'), 'exec'), module.__dict__)
    package = types.ModuleType('pysrc')
    package.__path__ = [str(SOURCE)]
    sys.modules['pysrc'] = package
    sys.modules['pysrc.export_weights'] = module
    spec = importlib.util.spec_from_file_location('original_compressed_reader', SOURCE/'weights_compress.py')
    compressed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compressed)
    # Shape-only derived views: the native recoder owns the five real derived
    # values and verifies them against the original C2 at its 434-tensor reload.
    compressed.cuda_rope_table = lambda inv_freq, n_pos, cos: np.broadcast_to(
        np.zeros((1, len(inv_freq)), dtype=np.float32), (n_pos, len(inv_freq)))
    return np, module, compressed


def initialize(out, manifest):
    np, raw, compressed = original_readers()
    assert C2.stat().st_size == 2930652
    assert hashlib.sha256(C2.read_bytes()).hexdigest() == '7f4db6c8c843a7e6264b6a48ed4805e9e431f543df7a9a0ecb37a35a5e4b8860'
    before = compressed.read_tensor_file_v2(str(C2))
    assert len(before) == 434 and DERIVED <= before.keys()
    fit = {t['name']: t for t in manifest['per_tensor']}
    assert len(fit) == 111 and set(fit) == {n for n, (d, _) in before.items() if d == raw.DTYPE_INT8}
    after = {n: (d, a.copy()) for n, (d, a) in before.items() if n not in DERIVED}
    assert len(after) == 429
    tensor_rows = []
    original_abs, new_abs, ratios = [], [], []
    changed_q = changed_scale = nonzero = zeros = total = 0
    mappings = set()
    histogram = np.zeros(15, dtype=np.int64)
    for name, f in fit.items():
        dtype, q = before[name]
        scale_name = name[:-2]+'.scale'
        scale_dtype, old_scale = before[scale_name]
        assert dtype == raw.DTYPE_INT8 and scale_dtype == raw.DTYPE_BF16_BITS
        assert q.shape == tuple(f['shape']) and q.size == f['elements']
        assert old_scale.shape == (q.shape[0],)
        assert np.bincount(q.ravel().astype(np.int16)+7, minlength=15).tolist() == f['original_histogram_15']
        mapped = after[name][1]
        new_scale = after[scale_name][1]
        num, den = f['scale_multiplier_exact']['numerator'], f['scale_multiplier_exact']['denominator']
        cache = {}
        local_ratios = []
        local_maps = set()
        for row, value in enumerate(old_scale):
            bits = int(value)
            if bits not in cache:
                new_bits = scale_bits(bits, num, den)
                lut, ratio = actual_mapping(bits, new_bits)
                cache[bits] = (new_bits, np.asarray(lut, dtype=np.int8), ratio)
            new_bits, lut, ratio = cache[bits]
            new_scale[row] = new_bits
            mapped[row] = lut[q[row].astype(np.int16)+7]
            assert bits >> 15 == new_bits >> 15
            original_abs.append(abs(bf16(bits)))
            new_abs.append(abs(bf16(new_bits)))
            ratios.append(ratio)
            local_ratios.append(ratio)
            mappings.add(tuple(lut.tolist()))
            local_maps.add(tuple(lut.tolist()))
        assert np.array_equal(q != 0, mapped != 0), name
        assert np.array_equal(np.sign(q), np.sign(mapped)), name
        assert set(np.unique(mapped).tolist()) <= {-7, -4, -2, 0, 2, 4, 7}
        changed_q += int(np.count_nonzero(q != mapped))
        changed_scale += int(np.count_nonzero(old_scale != new_scale))
        nonzero += int(np.count_nonzero(mapped))
        zeros += int(np.count_nonzero(mapped == 0))
        total += q.size
        histogram += np.bincount(mapped.ravel().astype(np.int16)+7, minlength=15)
        tensor_rows.append(dict(name=name, scale_name=scale_name, dtype=dtype, shape=list(q.shape),
            original_scale_bits_sha256=hashlib.sha256(old_scale.tobytes()).hexdigest(),
            new_scale_bits_sha256=hashlib.sha256(new_scale.tobytes()).hexdigest(),
            actual_ratio_min=min(local_ratios), actual_ratio_max=max(local_ratios),
            actual_mappings=[list(m) for m in sorted(local_maps)]))
    changed_names = set(fit) | {n[:-2]+'.scale' for n in fit}
    assert len(changed_names) == 222 and total == 5868864 and zeros == 631815
    for name, (dtype, a) in after.items():
        old_dtype, old = before[name]
        assert dtype == old_dtype and a.shape == old.shape
        if name not in changed_names:
            assert a.tobytes() == old.tobytes(), name
    entries = [(name, dtype, array) for name, (dtype, array) in after.items()]
    output = out/'nonuniform-payloads.bin'
    raw.write_tensor_file(str(output), entries)
    read_back = raw.read_tensor_file(str(output))
    assert len(read_back) == 429 and list(read_back) == list(after)
    for name, (dtype, array) in after.items():
        d, a = read_back[name]
        assert d == dtype and a.shape == array.shape and a.tobytes() == array.tobytes(), name
    return dict(status='passed', source_c2=str(C2), source_c2_sha256=hashlib.sha256(C2.read_bytes()).hexdigest(),
        original_reader_sources=EXPECTED_HASHES, tensor_count=434, payload_count=429,
        python_rope_shape_only=True, actual_derived_reconstruction_and_validation='existing C++ replacement recoder',
        names_dtype_shape_preserved=True, unchanged_payloads_exact=207, retained_derived=sorted(DERIVED),
        changed_allowed_tensors=222, quantized_tensors=111, paired_scale_tensors=111,
        int4_elements=total, changed_int4_values=changed_q, changed_scale_values=changed_scale,
        zero_values=zeros, nonzero_values=nonzero, nonzero_support_and_symbol_sign_preserved=True,
        actual_histogram_15=histogram.tolist(), actual_mappings=[list(m) for m in sorted(mappings)],
        original_abs_bf16_scale_min=min(original_abs), original_abs_bf16_scale_max=max(original_abs),
        new_abs_bf16_scale_min=min(new_abs), new_abs_bf16_scale_max=max(new_abs),
        actual_scale_ratio_min=min(ratios), actual_scale_ratio_max=max(ratios),
        payload_file=str(output), payload_bytes=output.stat().st_size,
        payload_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), raw_roundtrip_passed=True,
        per_tensor=tensor_rows)
