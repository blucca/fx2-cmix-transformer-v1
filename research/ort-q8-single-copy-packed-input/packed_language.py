#!/usr/bin/env python3
"""Fresh old/packed NIST chains against the existing complete same-Q8 oracle.

No dense/stock/FP32 model is rerun and no 487MB reference matrix is rebuilt.
The existing dense oracle is read-only; both fresh workers independently
build their own KV chain using the unchanged paired language_probe.worker.
"""
import argparse
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
NAMES = ('old', 'packed')


def gate(nll_delta, total_delta, chain, greedy):
    return {'same_q8_nll_pass': abs(nll_delta) <= 1e-5,
            'quantization_loss_pass': total_delta <= .03,
            'chained_precision_pass': bool(chain), 'greedy_4x64_identical': bool(greedy),
            'complete_language_gate_passed': bool(abs(nll_delta) <= 1e-5 and total_delta <= .03 and chain and greedy)}


def self_test():
    assert gate(0, .02, True, True)['complete_language_gate_passed']
    assert not gate(-1.01e-5, .02, True, True)['same_q8_nll_pass']
    assert not gate(0, .03001, True, True)['quantization_loss_pass']
    assert not gate(0, .02, False, True)['complete_language_gate_passed']
    assert not gate(0, .02, True, False)['complete_language_gate_passed']


def setup(args):
    sys.path.insert(0, str(args.common_root))
    common = importlib.import_module('native_runner')
    sys.path.insert(0, str(args.probe_root))
    probe = importlib.import_module('language_probe')
    return common, probe


def worker(args, common, probe):
    resource.setrlimit(resource.RLIMIT_AS, (args.max_as_mib*2**20,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if args.run_name == 'packed':
        from packed_runner import packed_session
        probe.session = lambda a, c, m: packed_session(c, args.model, args.library)
    else:
        probe.session = lambda a, c, m: c.session(args.base_root, 'native')
    args.root = args.out / args.run_name
    args.mode = 'native'
    probe.worker(args, common)
    path = args.root / 'language/native.json'
    report = json.loads(path.read_text())
    report.update(study_mode=args.run_name, fresh_session=True,
                  read_only_same_q8_oracle=str(args.base_root / 'language'),
                  model=str(args.model if args.run_name == 'packed' else args.base_root / 'models/native.onnx'))
    common.dump(path, report)


def compare(args, common):
    reports = {n: json.loads((args.out / n / 'language/native.json').read_text()) for n in NAMES}
    assert all(r['status'] == 'complete' and r['scored_tokens'] == 2048 and r['block_size'] == 1024 for r in reports.values())
    old, packed = reports['old'], reports['packed']
    fp32 = json.loads(args.fp32_report.read_text())
    assert fp32['status'] == 'complete' and fp32['scored_tokens'] == 2048
    current_ids = json.loads((args.base_root / 'language/tokens.json').read_text())['token_ids']
    fp32_ids = json.loads((args.fp32_report.parent / 'tokens.json').read_text())['token_ids']
    assert current_ids == fp32_ids and len(current_ids) == 2048
    assert [g['prompt_tokens'] for g in old['generations']] == [g['prompt_tokens'] for g in fp32['generations']]
    assert all(r['generations'][i]['prompt_tokens'] == old['generations'][i]['prompt_tokens'] for r in reports.values() for i in range(4))
    chain = all(r['same_weight_chained_precision_pass'] and r['teacher_forced_top1_equal'] == 2048
                and r['prefix_bits_preserved'] and r['all_outputs_finite_correct_shape'] for r in reports.values())
    nll_delta = packed['mean_nll_nats'] - old['mean_nll_nats']
    total_delta = packed['mean_nll_nats'] - fp32['mean_nll_nats']
    result = {'host': packed['host'], 'scored_tokens': 2048, 'block_size': 1024,
              'mean_nll_by_mode': {'fresh_old': old['mean_nll_nats'], 'packed': packed['mean_nll_nats'],
                                   'reused_original_fp32': fp32['mean_nll_nats']},
              'delta_packed_minus_fresh_old': nll_delta, 'delta_packed_minus_original_fp32': total_delta,
              'per_token_nll_identical': packed['per_token_nll_nats'] == old['per_token_nll_nats'],
              'max_abs_per_token_nll_difference': max(abs(a-b) for a,b in zip(packed['per_token_nll_nats'], old['per_token_nll_nats'])),
              'numerical_oracle': 'Both independent fresh KV chains compare every logits/new KV slice to the preserved full same-Q8 dense oracle; no oracle model rerun.',
              'max_norm_by_mode': {n: r['chained_max_norm_by_output'] for n,r in reports.items()},
              'original_fp32_report': str(args.fp32_report), 'original_fp32_host': fp32['host'],
              'reused_fp32_full_2048_target_ids_identical': True,
              'probe_scope': 'Fixed NIST government narrative, two independent 1024-target blocks, BOS condition only, no quantizer tuning.'}
    result.update(gate(nll_delta, total_delta, chain, packed['generations'] == old['generations']))
    common.dump(args.out / 'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--base-root', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--fp32-report', type=Path, required=True)
    p.add_argument('--common-root', type=Path, default=Path(__file__).resolve().parent.parent / 'ort-q8-whole-decode')
    p.add_argument('--probe-root', type=Path, default=Path(__file__).resolve().parent.parent / 'ort-q8-tied-vocabulary')
    p.add_argument('--action', choices=('self-test', 'all', 'worker', 'compare'), default='all')
    p.add_argument('--run-name', choices=NAMES, default='packed')
    p.add_argument('--max-as-mib', type=int, default=4096)
    args = p.parse_args()
    for name in ('out', 'base_root', 'model', 'library', 'fp32_report', 'common_root', 'probe_root'):
        setattr(args, name, getattr(args, name).resolve())
    self_test()
    if args.action == 'self-test':
        print('PACKED_LANGUAGE_GATE_SELF_TEST_PASS')
        return
    common, probe = setup(args)
    assert common.ort.__version__ == '1.31.1'
    if args.action == 'worker':
        worker(args, common, probe)
        return
    if args.action == 'compare':
        compare(args, common)
        return
    for name in NAMES:
        folder = args.out / name / 'language'
        folder.mkdir(parents=True, exist_ok=True)
        for file in ('tokens.json', 'reference-logits.npy', 'reference-new-kv.npy'):
            (folder / file).symlink_to(args.base_root / 'language' / file)
        common.dump(args.out / 'state.json', {'stage': name, 'status': 'running'})
        cmd = [sys.executable, str(Path(__file__).resolve()), '--out', str(args.out), '--base-root', str(args.base_root),
               '--model', str(args.model), '--library', str(args.library), '--fp32-report', str(args.fp32_report),
               '--common-root', str(args.common_root), '--probe-root', str(args.probe_root),
               '--action', 'worker', '--run-name', name, '--max-as-mib', str(args.max_as_mib)]
        with (args.out / (name + '.log')).open('w') as log:
            completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            common.dump(args.out / 'state.json', {'stage': name, 'status': 'failed', 'returncode': completed.returncode})
            raise SystemExit(completed.returncode)
    result = compare(args, common)
    common.dump(args.out / 'state.json', {'stage': 'complete', 'status': 'completed',
                                        'complete_language_gate_passed': result['complete_language_gate_passed']})
    if not result['complete_language_gate_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
