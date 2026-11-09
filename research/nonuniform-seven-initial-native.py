#!/usr/bin/env python3
"""One bounded physical W0 assay; original jobs, inputs and baseline are read-only."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path('/data/workspace/temp/hutter')
J = ROOT/'round484/q7-teacher'
OUT = ROOT/'round505/nonuniform-native'
CPU_LIMIT, WALL_LIMIT = 600, 900
INSPECTION_CPU = 0.096606895
PRIOR_CPU = 6.710751 + INSPECTION_CPU + 0.254744183 + 0.026981125
DEADLINE = datetime.fromisoformat('2026-11-09T06:15:48+00:00').timestamp()
WALL_LIMIT = max(1, int(DEADLINE-time.time()))
os.sched_setaffinity(0, {1})
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
resource.setrlimit(resource.RLIMIT_CPU, (int(CPU_LIMIT-PRIOR_CPU-1), int(CPU_LIMIT-PRIOR_CPU-1)))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
assert not (OUT/'state.json').exists(), 'inspect the existing assay before relaunch'
started = time.monotonic()
state = dict(schema='hutter.nonuniform7.initial-native.v1', status='running', stage='initialize',
             source_job=str(J), output=str(OUT), pid=os.getpid(),
             started_utc=datetime.now(timezone.utc).isoformat(),
             cpu_affinity=[1], cpu_limit_seconds=CPU_LIMIT, wall_limit_seconds=WALL_LIMIT,
             address_space_limit_bytes=2 << 30, memory_max_bytes=512 << 20,
             inspection_cpu_seconds=INSPECTION_CPU, native_calls=0, runs=[], scopes={},
             prior_cpu_including_failed_oom_seconds=PRIOR_CPU,
             absolute_gate_started_utc='2026-11-09T06:00:48+00:00',
             absolute_gate_deadline_utc='2026-11-09T06:15:48+00:00',
             train_steps=0, new_native_builds=0, original_units_and_watch_modified=0)


def cpu():
    u = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time()+u.ru_utime+u.ru_stime


def save():
    state.update(assay_cpu_seconds=cpu(), total_cpu_including_inspections=cpu()+PRIOR_CPU,
                 wall_seconds=time.monotonic()-started,
                 self_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    p=OUT/'state.tmp'
    p.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
    p.replace(OUT/'state.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(stage, command, logname):
    state.update(stage=stage, command=command)
    save()
    remaining = int(CPU_LIMIT-cpu()-PRIOR_CPU-1)
    assert remaining >= 10
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    wall=time.monotonic()
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (remaining, remaining))
    code=None
    try:
        with (OUT/logname).open('w') as stream:
            result=subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, timeout=max(1, WALL_LIMIT-(time.monotonic()-started)),
                preexec_fn=limits)
            code=result.returncode
            assert code == 0, f'{stage} exit {code}; read {logname}'
    finally:
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        state['runs'].append(dict(stage=stage, command=command, exit_code=code,
            cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
            wall_seconds=time.monotonic()-wall, child_peak_rss_kib=after.ru_maxrss, log=logname))
        save()


save()
try:
    manifest=json.loads((J/'incoming/data-manifest.json').read_text())
    baseline=json.loads((J/'incoming/baseline.json').read_text())
    assert manifest['method']=='causal-whole-piece-v1'
    assert manifest['source_bytes']==586459020 and baseline['status']=='passed'
    assert [s['name'] for s in manifest['samples']]==['causal','long-audit']
    old=J/'assets/original-static15.tfwc3'
    uniform=J/'training/step-0000.tfwc3'
    assert old.stat().st_size==2904325 and uniform.stat().st_size==2044814
    old_hash, uniform_hash=sha(old), sha(uniform)
    assert old_hash=='659c0767bce4146d2c457b9aaa13b3e707561ed5d37173a2a6ea1e5965e86b18'
    probe=J/'bin/activation-probe'
    state.update(original_W0_bytes=old.stat().st_size, original_W0_sha256=old_hash,
        original_uniform7_e0_bytes=uniform.stat().st_size, original_uniform7_e0_sha256=uniform_hash,
        evaluator=str(probe.resolve()), evaluator_sha256=sha(probe),
        manifest_sha256=sha(J/'incoming/data-manifest.json'), baseline_sha256=sha(J/'incoming/baseline.json'))
    fit=json.loads((OUT/'fit-manifest.json').read_text())
    state['fit_sha256']=fit['fit_sha256']
    spec=importlib.util.spec_from_file_location('nonuniform_initializer',OUT/'initializer.py')
    initializer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(initializer)
    # Three exact BF16/RNE fixtures exercise signs and both tie parities.
    assert initializer.scale_bits(0x3f80, 3, 4)==0x3f40
    assert initializer.scale_bits(0xbf80, 3, 4)==0xbf40
    assert initializer.scale_bits(0x3f80, 257, 512)==0x3f00
    assert initializer.scale_bits(0x3f80, 259, 512)==0x3f02
    prep_started=cpu()
    prep=initializer.initialize(OUT,fit)
    prep['cpu_seconds']=cpu()-prep_started
    prep['bf16_rne_fixture_passed']=True
    (OUT/'initialization.json').write_text(json.dumps(prep,indent=2)+'\n')
    state['initialization']={k:v for k,v in prep.items() if k!='per_tensor'}
    save()
    weight=OUT/'nonuniform-initial.tfwc3'
    assert not weight.exists()
    run('recode',[str(J/'assets/recode'),str(initializer.C2),str(weight),'--replace-payloads',prep['payload_file']],'recode.log')
    recode=json.loads((OUT/'recode.log').read_text())
    assert recode['status']=='passed' and recode['mode']=='replace-payloads'
    assert (recode['tensor_count'],recode['replacement_tensors'],recode['int4_tensors'],recode['int4_values'])==(434,429,111,5868864)
    assert recode['expected_tensor_bytes_equal'] and recode['other_tensor_bytes_equal']
    assert recode['output_bytes']==weight.stat().st_size
    weight_hash=sha(weight)
    state.update(weight_file=str(weight), weight_bytes=weight.stat().st_size, weight_sha256=weight_hash,
        recode=recode, complete_native_tensor_roundtrip_passed=True,
        single_W_extra_vs_uniform7_bytes=weight.stat().st_size-2044814,
        double_W_extra_vs_uniform7_bytes=2*(weight.stat().st_size-2044814),
        double_W_saved_vs_W0_bytes=2*(2904325-weight.stat().st_size))
    save()
    for sample,pieces,targets in zip(manifest['samples'],(473,1),(1082505,131071)):
        name,score=sample['name'],sample['score']
        expected=baseline['steps'][name]['result']
        command=[str(probe),'eval-pieces',score['tokens'],score['priors'],str(weight),score['boundaries']]
        state['native_calls']+=1
        save()
        logname=f'nonuniform-{name}.log'
        run(name,command,logname)
        rows=[json.loads(line.split(' ',1)[1]) for line in (OUT/logname).read_text().splitlines() if line.startswith('ACTIVATION_RESULT ')]
        assert len(rows)==1
        result=rows[0]
        assert result['mode']=='eval' and result['positive_u8']==0
        assert result['vocab']==205 and result['attention']=='KVI8' and result['rope_offset']==0
        assert result['explicit_piece_bounds'] and result['piece_bounds']==expected['piece_bounds']
        assert result['pieces']==pieces and result['total']['count']==targets==score['targets']
        assert result['input_bytes']==targets+pieces and result['prior_bytes']==result['input_bytes']*410
        losses=[row['loss'] for row in result['piece_losses']]
        assert len(losses)==pieces and sum(row['count'] for row in losses)==targets
        assert all(row['count']==hi-lo-1 for row,lo,hi in zip(losses,result['piece_bounds'],result['piece_bounds'][1:]))
        total=math.fsum(row['logits_sum'] for row in losses)
        nll=total/targets
        assert math.isclose(total,result['total']['logits_sum'],rel_tol=1e-11,abs_tol=1e-7)
        assert math.isfinite(nll) and abs(nll-result['total']['logits_mean'])<1e-12
        assert result['max_rss_kib']<=512*1024
        state['scopes'][name]=dict(targets=targets,raw_nll=nll,W0_raw_nll=expected['total']['logits_mean'],
            initial_quantization_delta=nll-expected['total']['logits_mean'],native=result,log=logname)
        save()
    indices=manifest['samples'][0]['score']['long_piece_indices']
    assert indices==[127,157,181]
    full=state['scopes']['causal']['native']['piece_losses']
    old_rows=baseline['steps']['causal']['result']['piece_losses']
    count=sum(full[i]['loss']['count'] for i in indices)
    assert count==107647 and 1082505-count==974858
    nll=math.fsum(full[i]['loss']['logits_sum'] for i in indices)/count
    old_nll=math.fsum(old_rows[i]['loss']['logits_sum'] for i in indices)/count
    state['held_long']=dict(targets=count,contained_in='causal',piece_indices=indices,
        raw_nll=nll,W0_raw_nll=old_nll,initial_quantization_delta=nll-old_nll)
    assert sum(row['targets'] for row in state['scopes'].values())==1213576
    assert sha(weight)==weight_hash and sha(old)==old_hash and sha(uniform)==uniform_hash
    assert cpu()+PRIOR_CPU<CPU_LIMIT and time.time()<DEADLINE
    assert max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)<=512*1024
    state.update(status='complete',stage='complete',independent_scored_targets=1213576,
        original_W0_and_uniform7_unchanged=True,new_weight_unchanged_during_native=True)
except Exception as error:
    state.update(status='failed',error=f'{type(error).__name__}: {error}')
    raise
finally:
    save()
print(json.dumps({k:v for k,v in state.items() if k not in ('scopes','initialization')},indent=2))
