#!/usr/bin/env python3
"""Stdlib-only replay of complete zero8 native scopes and real resource receipt."""
import json
import math
import sys
from pathlib import Path

OUT=Path(__file__).resolve().parent
path=Path(sys.argv[1]) if len(sys.argv)>1 else OUT/'results/depth-zero-native.json'
r=json.loads(path.read_text())
assert r['status']=='passed'
assert r['weight_bytes']==r['zero_export']['weight_bytes']==1955187
assert r['zero_export']['blocks']==8 and r['zero_export']['changed_weight_symbols']==0
assert r['new_hashes']==r['reruns']==r['builds']==r['optimizer_updates']==r['new_data']==0
for name,targets,pieces in [('causal',1082505,473),('long-audit',131071,1)]:
    x=r['native_outputs'][name];w=r['baseline_outputs'][name];identity=r['reference_identity'][name]
    assert x['total']['count']==w['total']['count']==targets
    assert x['pieces']==w['pieces']==pieces
    for key in identity: assert x[key]==w[key]==identity[key]
    bounds=x['piece_bounds']
    assert bounds[0]==0 and bounds[-1]==x['input_bytes']
    assert [b-a-1 for a,b in zip(bounds,bounds[1:])]==[p['loss']['count'] for p in x['piece_losses']]
    assert sum(p['loss']['count'] for p in x['piece_losses'])==targets==x['input_bytes']-pieces
    assert x['prior_bytes']==x['input_bytes']*410
    assert x['probability_bytes']==targets*205*4
    rows=[json.loads(line.split(' ',1)[1]) for line in r['raw_logs'][name].splitlines() if line.startswith('ACTIVATION_RESULT ')]
    assert rows==[x]
    for z in (x,w):
        assert z['first_half']['count']+z['second_half']['count']==targets
        for col in ('logits','probs'):
            summed=math.fsum(p['loss'][col+'_sum'] for p in z['piece_losses'])
            assert math.isclose(summed,z['total'][col+'_sum'],rel_tol=1e-11,abs_tol=1e-7)
            assert math.isclose(summed/targets,z['total'][col+'_mean'],rel_tol=1e-11,abs_tol=1e-12)
long_ix=r['score_scopes']['causal']['long_piece_indices']
ixs=dict(held=list(range(473)),held_long=long_ix,
    held_short=[i for i in range(473) if i not in set(long_ix)],maximum_length_audit=[0])
for scope,ix in ixs.items():
    name='long-audit' if scope=='maximum_length_audit' else 'causal'
    x=r['native_outputs'][name];w=r['baseline_outputs'][name];s=r['scopes'][scope]
    count=sum(x['piece_losses'][i]['loss']['count'] for i in ix)
    assert count==s['targets']==r['reference_native'][scope]['targets']
    value=lambda z,c:math.fsum(z['piece_losses'][i]['loss'][c+'_sum'] for i in ix)/count
    zero,old=value(x,'logits'),value(w,'logits')
    assert zero==s['zero8_raw_nll'] and old==s['w0_raw_nll']
    assert s['zero8_minus_w0']==zero-old
    for epoch in (4,8,12,16):
        learned=r['reference_native'][scope][f'e{epoch}_nll']
        assert s['learned_minus_zero8'][f'e{epoch}']==learned-zero
        assert math.isclose((zero-old)+(learned-zero),learned-old,abs_tol=1e-15)
e=r['execution'];runs=r['child_runs']
assert e['cpu_affinity']==[1] and e['threads_per_library']==1
assert e['address_space_limit_bytes']==4 << 30
assert 0<=e['total_cpu_seconds']<180 and 0<=e['wall_seconds']<240
assert e['max_process_rss_kib']<4 << 20
assert math.isclose(sum(x['cpu_seconds'] for x in runs),e['child_cpu_seconds'],abs_tol=1e-5)
assert all(x['exit_code']==0 and x['stop_reason'] is None for x in runs)
for child in runs:
    x=r['native_outputs'][child['scope']]
    assert x['eval_cpu_seconds']<=x['cpu_seconds']<=child['cpu_seconds']+.02
    assert x['max_rss_kib']<=child['max_rss_kib']
receipt=dict(status='passed',stdlib_only=True,complete_native_logs=2,held_pieces=473,
    held_targets=1082505,independent_audit_targets=131071,new_model_calls=0,
    checked=['counts','complete bounds and last-row mask','target-weighted piece means',
        'W0 structural delta','e4/e8/e12/e16 same-scope learning deltas','real CPU/wall/RSS budgets'],
    execution=e)
print(json.dumps(receipt,separators=(',',':')))
