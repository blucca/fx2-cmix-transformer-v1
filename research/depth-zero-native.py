#!/usr/bin/env python3
"""One read-only zero8 native baseline, full held then independent audit."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--job', type=Path, required=True,
                    help='Original eight-block job with incoming/, training/ and native/depth8/')
parser.add_argument('--output', type=Path, required=True,
                    help='Fresh experiment scratch directory for result, logs and checkpoints')
parser.add_argument('--context', type=Path,
                    default=Path(__file__).resolve().parent / 'results/depth-zero-native-context.json',
                    help='Original zero export, four learned scopes and complete native identities')
parser.add_argument('--cpu', type=int, default=1)
args = parser.parse_args()
OUT, ORIGINAL = args.output.resolve(), args.job.resolve()
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT/'result.json').exists(), 'Use a fresh result directory'
WEIGHT = ORIGINAL / 'training/step-0000.tfwc3'
EVALUATOR = ORIGINAL / 'native/depth8/activation-probe'
START = time.monotonic()
result = dict(schema='hutter.depth8.zero-native.round446.v1', status='running',
    stage='preflight', started_at=time.time(), weight_file=str(WEIGHT),
    evaluator=str(EVALUATOR), native_calls=0, reused_native_calls=0,
    reruns=0, builds=0, optimizer_updates=0, new_data=0, new_hashes=0,
    native_outputs={}, baseline_outputs={}, raw_logs={}, child_runs=[])


def account():
    own = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(supervisor_cpu_seconds=time.process_time(),
        child_cpu_seconds=child.ru_utime+child.ru_stime,
        total_cpu_seconds=time.process_time()+child.ru_utime+child.ru_stime,
        wall_seconds=time.monotonic()-START,
        max_process_rss_kib=max(own.ru_maxrss, child.ru_maxrss),
        address_space_limit_bytes=4 << 30, cpu_affinity=list(os.sched_getaffinity(0)),
        threads_per_library=1, cpu_total_cap_seconds=180, wall_total_cap_seconds=240,
        resource_scope='wait4 native children plus this supervisor; largest process RSS')


def save(stage, **updates):
    result.update(stage=stage, execution=account(), updated_at=time.time(), **updates)
    data = json.dumps(result,separators=(',',':'),allow_nan=False)+'\n'
    tmp = OUT / 'result.tmp'
    tmp.write_text(data)
    tmp.replace(OUT/'result.json')
    (OUT/('checkpoint-'+stage+'.json')).write_text(data)
    print(stage, json.dumps(result['execution'],separators=(',',':')),flush=True)


def deadline(signum, frame):
    raise TimeoutError('wall_deadline' if signum == signal.SIGALRM else 'cpu_deadline')


def native(text):
    rows = [json.loads(line.split(' ',1)[1]) for line in text.splitlines()
            if line.startswith('ACTIVATION_RESULT ')]
    assert len(rows) == 1, 'complete single native output required'
    return rows[0]


IDENTITY_FIELDS = ['mode','positive_u8','input_bytes','prior_bytes','vocab','attention',
    'rope_offset','half_split','pieces','separator_ends','length_cuts','piece_bounds',
    'explicit_piece_bounds']


def check_output(x, bounds, targets, identity):
    assert all(x[k] == identity[k] for k in IDENTITY_FIELDS), 'original native scope mismatch'
    assert x['piece_bounds'] == bounds and x['explicit_piece_bounds'] is True
    assert x['total']['count'] == targets == x['input_bytes']-x['pieces']
    assert x['prior_bytes'] == x['input_bytes']*410
    assert len(x['piece_losses']) == x['pieces'] == len(bounds)-1
    assert [p['loss']['count'] for p in x['piece_losses']] == [b-a-1 for a,b in zip(bounds,bounds[1:])]
    assert x['first_half']['count']+x['second_half']['count'] == targets
    for column in ('logits','probs'):
        total = x['total']
        piece_sum = math.fsum(p['loss'][column+'_sum'] for p in x['piece_losses'])
        assert math.isclose(piece_sum,total[column+'_sum'],rel_tol=1e-11,abs_tol=1e-7)
        assert math.isclose(piece_sum/targets,total[column+'_mean'],rel_tol=1e-11,abs_tol=1e-12)
        assert math.isfinite(total[column+'_mean'])


def evaluate(name, score, identity):
    targets = score['targets']
    bounds_data = Path(score['boundaries']).read_bytes()
    assert len(bounds_data) >= 8 and len(bounds_data)%4 == 0
    bounds = list(struct.unpack('<'+'i'*(len(bounds_data)//4),bounds_data))
    assert bounds == identity['piece_bounds']
    for key in ('tokens','priors','boundaries'):
        p = Path(score[key]); assert p.is_file() and os.access(p,os.R_OK), 'missing score asset: '+str(p)
    assert Path(score['tokens']).stat().st_size == identity['input_bytes']
    assert Path(score['priors']).stat().st_size == identity['prior_bytes']
    baseline = result['baseline_outputs'][name]
    check_output(baseline,bounds,targets,identity)
    log = OUT / ('step-0000-'+name+'.log')
    candidates = [ORIGINAL/'training'/log.name, ORIGINAL/'check'/log.name]
    prior = next((p for p in candidates if p.is_file()),None)
    save('before-'+name, current_scope=name, score_manifest=score,
         existing_zero_log_candidates=[str(p) for p in candidates], reused_from=str(prior) if prior else None)
    if prior:
        text = prior.read_text()
        with log.open('x') as f: f.write(text)
        result['reused_native_calls'] += 1
    else:
        remaining = math.floor(180-account()['total_cpu_seconds'])-2
        assert remaining >= 3, 'cpu budget exhausted'
        def limits():
            resource.setrlimit(resource.RLIMIT_CPU,(max(1,remaining-1),remaining))
            resource.setrlimit(resource.RLIMIT_AS,(4 << 30,4 << 30))
        command = [str(EVALUATOR),'eval-pieces',score['tokens'],score['priors'],str(WEIGHT),score['boundaries']]
        result['native_calls'] += 1
        started = time.monotonic()
        with log.open('x') as stream:
            child = subprocess.Popen(command,cwd=OUT,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,preexec_fn=limits)
            stop = None
            try:
                _,status,usage = os.wait4(child.pid,0)
            except TimeoutError as exc:
                stop = str(exc)
                child.kill()
                _,status,usage = os.wait4(child.pid,0)
            code = os.waitstatus_to_exitcode(status); child.returncode = code
        result['child_runs'].append(dict(scope=name,command=command,exit_code=code,
            cpu_seconds=usage.ru_utime+usage.ru_stime,max_rss_kib=usage.ru_maxrss,
            wall_seconds=time.monotonic()-started,cpu_limit_seconds=remaining,stop_reason=stop))
        assert code == 0 and stop is None, 'native stopped: '+str(stop or code)
        text = log.read_text()
    x = native(text)
    check_output(x,bounds,targets,identity)
    result['native_outputs'][name] = x
    result['raw_logs'][name] = text
    save('after-'+name, current_scope=name)


def main():
    os.sched_setaffinity(0,{args.cpu})
    resource.setrlimit(resource.RLIMIT_AS,(4 << 30,4 << 30))
    resource.setrlimit(resource.RLIMIT_CPU,(179,180))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    signal.signal(signal.SIGALRM,deadline); signal.signal(signal.SIGXCPU,deadline)
    signal.alarm(225)
    save('preflight')
    context = json.loads(args.context.resolve().read_text())
    context['zero_export']['weight_file'] = str(WEIGHT)
    manifest_path = ORIGINAL/'incoming/data-manifest.json'
    baseline_path = ORIGINAL/'incoming/baseline.json'
    for p in (WEIGHT,EVALUATOR,manifest_path,baseline_path):
        assert p.is_file() and os.access(p,os.R_OK), 'missing original asset: '+str(p)
    before = WEIGHT.stat()
    assert before.st_size == context['zero_export']['weight_bytes'] == 1955187
    manifest = json.loads(manifest_path.read_text()); baseline = json.loads(baseline_path.read_text())
    assert manifest['method'] == 'causal-whole-piece-v1' and manifest['source_bytes'] == 586459020
    assert baseline['status'] == 'passed'
    samples = manifest['samples']
    # Relative score paths are resolved within the selected job; absolute paths
    # retain the original manifest's behavior.
    for sample in samples:
        for key in ('tokens','priors','boundaries'):
            path = Path(sample['score'][key])
            if not path.is_absolute():
                sample['score'][key] = str(ORIGINAL/path)
    assert [s['name'] for s in samples] == ['causal','long-audit']
    assert [s['score']['targets'] for s in samples] == [1082505,131071]
    result['baseline_outputs'] = {s['name']:baseline['steps'][s['name']]['result'] for s in samples}
    save('assets-ready',weight_bytes=before.st_size,weight_inode=before.st_ino,
        weight_mtime_ns=before.st_mtime_ns,zero_export=context['zero_export'],
        reference_native=context['native'],reference_identity=context['identity'],
        manifest_path=str(manifest_path),baseline_path=str(baseline_path),
        score_scopes={s['name']:s['score'] for s in samples},
        mask='Every explicit piece scores begin through end-2; piece-last prior/EOF rows excluded, piece-first token has no preceding target prediction.')
    for sample in samples:
        evaluate(sample['name'],sample['score'],context['identity'][sample['name']])
    held = result['native_outputs']['causal']; audit = result['native_outputs']['long-audit']
    indices = samples[0]['score']['long_piece_indices']; assert len(indices) == len(set(indices))
    long_count = sum(held['piece_losses'][i]['loss']['count'] for i in indices)
    assert long_count == 107647
    scope_indices = dict(held=list(range(473)),held_long=indices,
        held_short=[i for i in range(473) if i not in set(indices)],maximum_length_audit=[0])
    scopes = {}
    for scope,ix in scope_indices.items():
        name = 'long-audit' if scope == 'maximum_length_audit' else 'causal'
        zero = result['native_outputs'][name]; old = result['baseline_outputs'][name]
        count = sum(zero['piece_losses'][i]['loss']['count'] for i in ix)
        means = {label:{column:math.fsum(x['piece_losses'][i]['loss'][column+'_sum'] for i in ix)/count
                for column in ('logits','probs')} for label,x in (('zero8',zero),('w0',old))}
        ref = context['native'][scope]
        assert count == ref['targets']
        assert math.isclose(means['w0']['logits'],ref['original_nll'],abs_tol=1e-11)
        scopes[scope] = dict(targets=count,zero8_raw_nll=means['zero8']['logits'],
            w0_raw_nll=means['w0']['logits'],zero8_minus_w0=means['zero8']['logits']-means['w0']['logits'],
            zero8_probs_nll=means['zero8']['probs'],w0_probs_nll=means['w0']['probs'],
            zero8_probs_minus_w0=means['zero8']['probs']-means['w0']['probs'],
            learned_nll={f'e{e}':ref[f'e{e}_nll'] for e in (4,8,12,16)},
            learned_minus_zero8={f'e{e}':ref[f'e{e}_nll']-means['zero8']['logits'] for e in (4,8,12,16)})
    after = WEIGHT.stat()
    assert (before.st_ino,before.st_size,before.st_mtime_ns) == (after.st_ino,after.st_size,after.st_mtime_ns)
    execution = account()
    assert execution['total_cpu_seconds'] < 180 and execution['wall_seconds'] < 240
    assert execution['max_process_rss_kib'] < (4 << 20)
    signal.alarm(0)
    save('complete',status='passed',scopes=scopes,checks=dict(counts=True,
        complete_piece_bounds=True,target_weighted_piece_means=True,same_scope_differences=True,
        unchanged_original_weight_metadata=True,cpu_wall_rss_budget=True),completed_at=time.time())


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        signal.alarm(0)
        save('failed',status='failed',error_kind=type(error).__name__,error=str(error),
             failed_stage=result['stage'],completed_at=time.time())
        traceback.print_exc()
        raise SystemExit(124 if isinstance(error,TimeoutError) else 1)
    print('ZERO_NATIVE_SUMMARY '+json.dumps({k:result[k] for k in
        ('status','weight_bytes','native_calls','reused_native_calls','scopes','execution')},separators=(',',':')),flush=True)
