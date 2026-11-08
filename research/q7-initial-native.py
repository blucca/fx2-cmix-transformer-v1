#!/usr/bin/env python3
"""One complete e0 control, using the admitted full12 native evaluator.

This independent assay leaves the training trajectory and selection intact.
It reuses the original held inputs and W0 native results, with no export.
"""
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path('/data/workspace/temp/hutter')
J = ROOT / 'round484/q7-teacher'
OUT = ROOT / 'round499/q7-initial-native'
CPU_LIMIT, WALL_LIMIT = 600, 900
os.sched_setaffinity(0, {1})
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
OUT.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
state = dict(schema='hutter.q7.initial-native.v1', status='running', epoch=0,
             source_job=str(J), output=str(OUT), cpu_affinity=[1],
             cpu_limit_seconds=CPU_LIMIT, wall_limit_seconds=WALL_LIMIT,
             address_space_limit_bytes=2 << 30, native_calls=0, scopes={},
             provenance='original q7 zero export and original full12 native family')

def cpu():
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + child.ru_utime + child.ru_stime

def save():
    state.update(total_cpu_seconds=cpu(), wall_seconds=time.monotonic()-started,
                 child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (OUT / 'state.json').write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')

save()
try:
    manifest = json.loads((J/'incoming/data-manifest.json').read_text())
    baseline = json.loads((J/'incoming/baseline.json').read_text())
    assert manifest['method'] == 'causal-whole-piece-v1'
    assert manifest['source_bytes'] == 586459020 and baseline['status'] == 'passed'
    assert [s['name'] for s in manifest['samples']] == ['causal', 'long-audit']
    weight, reference = J/'training/step-0000.tfwc3', J/'training/quant7-reference.tfwc3'
    assert weight.stat().st_size == 2044814 and weight.read_bytes() == reference.read_bytes()
    probe = J/'bin/activation-probe'
    state.update(weight_file=str(weight), weight_bytes=weight.stat().st_size,
                 initial_reference_bytes_equal=True, evaluator=str(probe.resolve()))
    save()
    for sample, pieces, targets in zip(manifest['samples'], (473, 1), (1082505, 131071)):
        name, score = sample['name'], sample['score']
        expected = baseline['steps'][name]['result']
        command = [str(probe), 'eval-pieces', score['tokens'], score['priors'], str(weight), score['boundaries']]
        state['stage'] = name
        state['command'] = command
        save()
        remaining = int(CPU_LIMIT-cpu())
        assert remaining >= 10
        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (remaining, remaining))
        log = OUT/f'e0-{name}.log'
        with log.open('w') as stream:
            subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=max(1, WALL_LIMIT-(time.monotonic()-started)),
                           preexec_fn=limits, check=True)
        rows = [json.loads(line.split(' ', 1)[1]) for line in log.read_text().splitlines()
                if line.startswith('ACTIVATION_RESULT ')]
        assert len(rows) == 1
        result = rows[0]
        assert result['mode'] == 'eval' and result['positive_u8'] == 0
        assert result['vocab'] == 205 and result['attention'] == 'KVI8' and result['rope_offset'] == 0
        assert result['explicit_piece_bounds'] and result['piece_bounds'] == expected['piece_bounds']
        assert result['pieces'] == pieces and result['total']['count'] == targets == score['targets']
        assert result['input_bytes'] == targets+pieces and result['prior_bytes'] == result['input_bytes']*410
        losses = [row['loss'] for row in result['piece_losses']]
        assert len(losses) == pieces and sum(row['count'] for row in losses) == targets
        assert all(row['count'] == hi-lo-1 for row, lo, hi in
                   zip(losses, result['piece_bounds'], result['piece_bounds'][1:]))
        nll = math.fsum(row['logits_sum'] for row in losses)/targets
        assert math.isfinite(nll) and abs(nll-result['total']['logits_mean']) < 1e-12
        state['scopes'][name] = dict(targets=targets, raw_nll=nll,
            W0_raw_nll=expected['total']['logits_mean'],
            initial_quantization_delta=nll-expected['total']['logits_mean'],
            native=result, log=str(log))
        state['native_calls'] += 1
        save()
    indices = manifest['samples'][0]['score']['long_piece_indices']
    assert indices == [127, 157, 181]
    full = state['scopes']['causal']['native']['piece_losses']
    old = baseline['steps']['causal']['result']['piece_losses']
    count = sum(full[i]['loss']['count'] for i in indices)
    assert count == 107647 and 1082505-count == 974858
    nll = math.fsum(full[i]['loss']['logits_sum'] for i in indices)/count
    old_nll = math.fsum(old[i]['loss']['logits_sum'] for i in indices)/count
    state['held_long'] = dict(targets=count, contained_in='causal', piece_indices=indices,
        raw_nll=nll, W0_raw_nll=old_nll, initial_quantization_delta=nll-old_nll)
    assert sum(row['targets'] for row in state['scopes'].values()) == 1213576
    assert cpu() <= CPU_LIMIT
    state.update(status='complete', stage='complete', independent_scored_targets=1213576)
except Exception as error:
    state.update(status='failed', error=f'{type(error).__name__}: {error}')
    raise
finally:
    save()
print(json.dumps({k: v for k, v in state.items() if k != 'scopes'}, indent=2))
