#!/usr/bin/env python3
"""Single-copy Q8 study: three fresh complete sessions, fixed prior feeds.

Reuses native_runner.run without changing its 30 warmups, 200 samples,
all thirteen returned outputs, validation, or per-worker RSS measurement.
"""
import argparse
import ctypes
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
sys.dont_write_bytecode = True

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

NAMES = ('old-before', 'packed', 'old-after')


def gate(rows, old_rss, new_rss, numerical, target):
    performance = all(r['packed_p50_ms'] <= r['old_conservative_p50_ms'] * 1.05 for r in rows)
    memory = new_rss * 5 <= old_rss * 4  # Exact integer KiB threshold, including 20%.
    return {'numerical_pass': bool(numerical), 'performance_pass': performance,
            'memory_pass': memory, 'target_avx2_no_vnni': bool(target),
            'target_compute_gate_passed': bool(numerical and performance and memory and target)}


def self_test():
    row = [{'old_conservative_p50_ms': 10., 'packed_p50_ms': 10.5}]
    assert gate(row, 100, 80, True, True)['target_compute_gate_passed']
    assert not gate([{'old_conservative_p50_ms': 10., 'packed_p50_ms': 10.501}], 100, 79, True, True)['performance_pass']
    assert not gate(row, 100, 81, True, True)['memory_pass']
    assert not gate(row, 100, 79, False, True)['target_compute_gate_passed']
    assert not gate(row, 100, 79, True, False)['target_compute_gate_passed']


def setup(args):
    sys.path.insert(0, str(args.common_root))
    return importlib.import_module('native_runner')


def packed_session(common, model, library):
    ort = common.ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    opts.register_custom_ops_library(str(library))
    s = ort.InferenceSession(str(model), opts, providers=['CPUExecutionProvider'])
    assert [x.name for x in s.get_inputs()] == common.INPUTS
    assert [x.name for x in s.get_outputs()] == common.OUTPUTS
    return s


def worker(args, common):
    resource.setrlimit(resource.RLIMIT_AS, (args.max_as_mib * 2**20,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    original_session = common.session
    diagnostic = None
    if args.run_name == 'packed':
        handle = ctypes.CDLL(str(args.library))
        ctor, owned = handle.BluccaQ8M1PackedCtorCount, handle.BluccaQ8M1PackedOwnedPackedBytes
        for f in (ctor, owned):
            f.restype, f.argtypes = ctypes.c_uint64, []
        before = ctor()
        common.session = lambda root, mode: packed_session(common, args.model, args.library)
        diagnostic = (ctor, owned, before)
    else:
        common.session = lambda root, mode: original_session(args.base_root, 'native')
    reference = 'self' if args.run_name == 'old-before' else 'old-before'
    common.run(args.out, 'target', 'packed' if diagnostic else 'native', args.run_name, reference)
    path = args.out / 'runs/target' / args.run_name / 'result.json'
    report = json.loads(path.read_text())
    report.update(model=str(args.model if diagnostic else args.base_root / 'models/native.onnx'),
                  library=str(args.library if diagnostic else args.base_root / 'libq8_m1.so'),
                  fresh_session=True, feeds_reused_from=str(args.base_root / 'feeds'),
                  comparison_scope='same tied Q8 weights and complete feeds; old owner vs direct packed initializer view')
    if diagnostic:
        ctor, owned, before = diagnostic
        report.update(constructor_delta=ctor() - before, native_owned_packed_bytes=owned())
        assert report['constructor_delta'] == 25 and report['native_owned_packed_bytes'] == 0
    common.dump(path, report)


def compare(args, common):
    folder = args.out / 'runs/target'
    reports = {n: json.loads((folder / n / 'result.json').read_text()) for n in NAMES}
    assert all(r['status'] == 'completed' and len(r['rows']) == 3 and r['timed'] == 200 for r in reports.values())
    rows = []
    bits_by_past = {}
    for i, p in enumerate(common.PAST):
        assert all(r['rows'][i]['past'] == p for r in reports.values())
        old = min(reports[n]['rows'][i]['p50_ms'] for n in ('old-before', 'old-after'))
        new = reports['packed']['rows'][i]['p50_ms']
        before = common.load_npz(folder / 'old-before' / f'past-{p}-outputs.npz')
        packed = common.load_npz(folder / 'packed' / f'past-{p}-outputs.npz')
        bits_by_past[str(p)] = all(common.np.array_equal(before[n].view(common.np.uint32), packed[n].view(common.np.uint32)) for n in common.OUTPUTS)
        rows.append({'past': p, 'old_before_p50_ms': reports['old-before']['rows'][i]['p50_ms'],
                     'old_after_p50_ms': reports['old-after']['rows'][i]['p50_ms'],
                     'old_conservative_p50_ms': old, 'packed_p50_ms': new,
                     'time_reduction': 1-new/old, 'all_13_output_bits_same': bits_by_past[str(p)]})
    old_rss = min(reports[n]['peak_rss_kib'] for n in ('old-before', 'old-after'))
    new_rss = reports['packed']['peak_rss_kib']
    features = reports['packed']['host']['features']
    target = features['avx2'] and features['fma'] and not any(features[k] for k in ('avx512f', 'avx512_vnni', 'avx_vnni'))
    numerical = all(r['all_outputs_passed'] for r in reports.values())
    result = {'host': reports['packed']['host'], 'rows': rows,
              'old_conservative_peak_rss_kib': old_rss, 'packed_peak_rss_kib': new_rss,
              'rss_reduction': 1-new_rss/old_rss,
              'all_13_output_bits_same_by_past': bits_by_past,
              'packed_native_owned_bytes': reports['packed']['native_owned_packed_bytes'],
              'full_reports': {n: f'{n}/result.json' for n in NAMES},
              'quality_gate': 'separate fixed NIST2048 independent chain and 4x64 greedy'}
    result.update(gate(rows, old_rss, new_rss, numerical, target))
    common.dump(folder / 'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--base-root', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--common-root', type=Path, default=Path(__file__).resolve().parent.parent / 'ort-q8-whole-decode')
    p.add_argument('--action', choices=('self-test', 'all', 'worker', 'compare'), default='all')
    p.add_argument('--run-name', choices=NAMES, default='packed')
    p.add_argument('--max-as-mib', type=int, default=4096)
    args = p.parse_args()
    for name in ('out', 'base_root', 'model', 'library', 'common_root'):
        setattr(args, name, getattr(args, name).resolve())
    self_test()
    if args.action == 'self-test':
        print('PACKED_GATE_SELF_TEST_PASS')
        return
    common = setup(args)
    assert common.ort.__version__ == '1.31.1'
    if args.action == 'worker':
        worker(args, common)
        return
    if args.action == 'compare':
        compare(args, common)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'feeds').symlink_to(args.base_root / 'feeds', target_is_directory=True)
    folder = args.out / 'runs/target'
    for name in NAMES:
        common.dump(folder / 'state.json', {'stage': name, 'status': 'running'})
        cmd = [sys.executable, str(Path(__file__).resolve()), '--out', str(args.out),
               '--base-root', str(args.base_root), '--model', str(args.model), '--library', str(args.library),
               '--common-root', str(args.common_root), '--action', 'worker', '--run-name', name,
               '--max-as-mib', str(args.max_as_mib)]
        with (folder / (name + '.log')).open('w') as log:
            completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            common.dump(folder / 'state.json', {'stage': name, 'status': 'failed', 'returncode': completed.returncode})
            raise SystemExit(completed.returncode)
    result = compare(args, common)
    common.dump(folder / 'state.json', {'stage': 'complete', 'status': 'completed',
                                      'target_compute_gate_passed': result['target_compute_gate_passed']})
    if not result['target_compute_gate_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
