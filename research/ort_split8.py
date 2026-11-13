#!/usr/bin/env python3
"""Whole-call test: symmetric uint8/block32 weights as two native Q4 terms.

Scope: FP32 matrices, constant symmetric weights, K a multiple of 32.
The helper builds one operator graph; real model integration has its own contract.
"""
import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

BLOCK = 32


def wrap(nodes, inits, m, k, n):
    graph = helper.make_graph(nodes, 'split8',
        [helper.make_tensor_value_info('A', TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [m, n])],
        [numpy_helper.from_array(a, name) for name, a in inits.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17), helper.make_opsetid('com.microsoft', 1)], ir_version=9)
    return model.SerializeToString(), sum(a.nbytes for a in inits.values())


def packed4(q):
    # Serialized MatMulNBits storage: each byte packs two adjacent K values.
    return np.ascontiguousarray(q[..., 0::2] | (q[..., 1::2] << 4))


def split8(m, k, n, b, s, *, low_zero=True):
    assert k % BLOCK == 0 and b.shape == (n, k // BLOCK, BLOCK)
    assert b.dtype == np.uint8 and s.dtype == np.float32 and s.size == n * (k // BLOCK)
    high, low = packed4(b >> 4), packed4(b & 15)
    high_scale = s * np.float32(16)
    assert np.isfinite(high_scale).all()
    low_zp = np.zeros((n, (k // BLOCK + 1) // 2), dtype=np.uint8)
    attrs = dict(domain='com.microsoft', K=k, N=n, bits=4, block_size=BLOCK, accuracy_level=0)
    nodes = [helper.make_node('MatMulNBits', ['A', 'BH', 'SH'], ['H'], **attrs),
             helper.make_node('MatMulNBits', ['A', 'BL', 'SL', *(['ZL'] if low_zero else [])], ['L'], **attrs),
             helper.make_node('Add', ['H', 'L'], ['Y'])]
    return wrap(nodes, {'BH': high, 'BL': low, 'SH': high_scale, 'SL': s, **({'ZL': low_zp} if low_zero else {})}, m, k, n)


def q8(m, k, n, b, s, acc):
    node = helper.make_node('MatMulNBits', ['A', 'B', 'S'], ['Y'], domain='com.microsoft',
                            K=k, N=n, bits=8, block_size=BLOCK, accuracy_level=acc)
    return wrap([node], {'B': b, 'S': s}, m, k, n)


def dense(m, k, n, w):
    return wrap([helper.make_node('MatMul', ['A', 'W'], ['Y'])], {'W': w}, m, k, n)


def dequant(b, s):
    n, kb, _ = b.shape
    return np.ascontiguousarray(((b.astype(np.float32) - 128) * s.reshape(n, kb, 1)).transpose(1, 2, 0).reshape(kb * BLOCK, n))


def quant8(w):
    k, n = w.shape
    wb = w.reshape(k // BLOCK, BLOCK, n)
    scale = np.maximum(np.abs(wb).max(axis=1), np.float32(1e-8)) / np.float32(127)
    q = np.clip(np.rint(wb / scale[:, None, :]) + 128, 0, 255).astype(np.uint8)
    b, s = np.ascontiguousarray(q.transpose(2, 0, 1)), np.ascontiguousarray(scale.T).reshape(-1)
    return b, s, dequant(b, s)


def session(model):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    return ort.InferenceSession(model, opts, providers=['CPUExecutionProvider'])


def measure(model, a):
    sess = session(model)
    for _ in range(30):
        y = sess.run(None, {'A': a})[0]
    samples = []
    for _ in range(200):
        start = time.perf_counter()
        sess.run(None, {'A': a})
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return float(np.median(samples)), float(samples[189]), y


def error(y, ref):
    absolute = float(np.abs(y - ref).max())
    return absolute, absolute / (float(np.abs(ref).max()) + 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('split8-results.json'))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else platform.processor()
    model_line = next((line.split(':', 1)[1].strip() for line in raw.splitlines() if line.startswith('model name')), platform.processor())
    flags = next((line.split(':', 1)[1].split() for line in raw.splitlines() if line.startswith('flags')), [])
    state = {'status': 'running', 'startedAt': datetime.now(timezone.utc).isoformat(),
             'python': sys.version.split()[0], 'ort': ort.__version__, 'onnx': onnx.__version__, 'numpy': np.__version__,
             # /proc/cpuinfo flags are Linux-only; Windows/other hosts report unknown.
             'cpu': model_line, 'features': {name: name in flags if flags else None for name in ['avx2', 'avx512f', 'avx512_vnni', 'avx_vnni']},
             'threads': 1, 'warmup': 30, 'iterations': 200, 'packing': 'adjacent-low/high', 'controls': {}, 'rows': []}
    def save():
        args.out.write_text(json.dumps(state, indent=2) + '\n')
    save()
    try:
        # All 256 codes, with power-of-two scales, expose nibble order and zero points exactly.
        b = np.arange(256, dtype=np.uint8).reshape(8, 1, 32)
        s = np.full(8, .125, dtype=np.float32)
        a = np.eye(32, dtype=np.float32)
        ref = dequant(b, s)
        original = session(q8(32, 32, 8, b, s, 0)[0]).run(None, {'A': a})[0]
        actual = session(split8(32, 32, 8, b, s)[0]).run(None, {'A': a})[0]
        state['controls']['q8_all256_equal'] = bool(np.array_equal(original, ref))
        state['controls']['split_all256_equal'] = bool(np.array_equal(actual, ref))
        state['controls']['split_first_rows'] = actual[:4].tolist()
        save()
        assert np.array_equal(original, ref)
        assert np.array_equal(actual, ref), 'All-code native control: nibble storage or zero-point mismatch'
        wrong = session(split8(32, 32, 8, b, s, low_zero=False)[0]).run(None, {'A': a})[0]
        state['controls']['missing_low_zero_max_abs'] = error(wrong, ref)[0]
        assert np.array_equal(wrong, ref - np.float32(1)), 'Negative control should shift every weight by -8*S'
        rng_control = np.random.default_rng(561)
        b = rng_control.integers(0, 256, size=(17, 2, 32), dtype=np.uint8)
        s = np.linspace(.00013, .00471, 34, dtype=np.float32)
        a = rng_control.standard_normal((3, 64)).astype(np.float32)
        ref = a @ dequant(b, s)
        actual = session(split8(3, 64, 17, b, s)[0]).run(None, {'A': a})[0]
        state['controls']['arbitrary_scales_norm_error'] = error(actual, ref)[1]
        assert error(actual, ref)[1] <= 2e-6
        save()
        # Generate in the reporter's order/seed, then measure single-token first.
        rng = np.random.default_rng(0)
        cases = []
        for m, k, n in [(197, 768, 3072), (197, 3072, 768), (1, 768, 3072)]:
            w = rng.standard_normal((k, n)).astype(np.float32) * np.float32(.02)
            a = rng.standard_normal((m, k)).astype(np.float32)
            b, s, deq = quant8(w)
            cases.append((m, k, n, a, b, s, deq))
        for m, k, n, a, b, s, deq in sorted(cases, key=lambda c: c[0]):
            ref = a @ deq
            row = {'M': m, 'K': k, 'N': n, 'B_fp32_MiB': k * n * 4 / 2**20, 'modes': {}}
            for name, (model, weight_bytes) in [('fp32_same_weights', dense(m, k, n, deq)),
                    ('q8_acc0', q8(m, k, n, b, s, 0)), ('split_two_q4', split8(m, k, n, b, s)),
                    ('q8_acc4', q8(m, k, n, b, s, 4))]:
                p50, p95, y = measure(model, a)
                absolute, relative = error(y, ref)
                row['modes'][name] = {'p50_ms': p50, 'p95_ms': p95, 'max_abs_error': absolute,
                                       'norm_max_error': relative, 'initializer_bytes': weight_bytes}
            row['speedup_vs_q8_acc0'] = row['modes']['q8_acc0']['p50_ms'] / row['modes']['split_two_q4']['p50_ms']
            row['initializer_ratio_vs_q8'] = row['modes']['split_two_q4']['initializer_bytes'] / row['modes']['q8_acc0']['initializer_bytes']
            state['rows'].append(row)
            print(json.dumps(row), flush=True)
            save()
        single = next(row for row in state['rows'] if row['M'] == 1)
        decision = {'single_token_speedup_pass': single['speedup_vs_q8_acc0'] >= 2,
                    'precision_pass': all(row['modes']['split_two_q4']['norm_max_error'] <= 2e-6 for row in state['rows']),
                    'initializer_pass': all(row['initializer_ratio_vs_q8'] <= 1.15 for row in state['rows'])}
        decision['passed'] = all(decision.values())
        # Preserve the original all-shape gate; expose the selected M1 scope separately.
        decision['m1_scope_passed'] = (decision['single_token_speedup_pass']
                                     and single['modes']['split_two_q4']['norm_max_error'] <= 2e-6
                                     and single['initializer_ratio_vs_q8'] <= 1.15)
        state['decision'] = decision
        state['status'] = 'completed'
    except Exception as exc:
        state['status'] = 'failed'
        state['error'] = repr(exc)
        raise
    finally:
        state['completedAt'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
