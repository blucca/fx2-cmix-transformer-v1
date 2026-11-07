#!/usr/bin/env python3
"""One stdlib-only check of the actual compact result; model calls zero."""
import json
import math
import sys
from pathlib import Path

OUT=Path(__file__).resolve().parent
path=Path(sys.argv[1]) if len(sys.argv)>1 else OUT/'results/depth-objective-gradients.json'
r=json.loads(path.read_text())
assert r['status']=='passed',r.get('stop_reason',r.get('error'))
assert r['optimizer_updates']==r['optimizer_instances']==r['new_teacher_calls']==r['new_cache_files']==0
assert r['native_calls']==r['capture_calls']==r['codec_calls']==0
assert r['hard_mode'] and r['checkpoint']['cursor']['epoch']>=8
n=r['piece']['targets']
assert 1025<=n<=2048 and r['piece']['hi']-r['piece']['lo']-1==n
assert r['piece']['bounds']==[[0,n]] and r['piece']['priors_dtype']=='torch.float16'
assert r['teacher_cache']['logits_shape']==[1,n,205]
assert r['teacher_cache']['hidden_shape']==[1,n,192] and r['teacher_cache']['dtype']=='float32'
assert (r['counts']['z_tensors'],r['counts']['raw_tensors'],r['counts']['trainable_tensors'])==(75,59,134)
assert r['manifest']['training_pieces']==1313 and r['manifest']['training_targets']==2941307
assert all(math.isfinite(v) for v in r['losses'].values())
loss=r['losses']
assert math.isclose(loss['old_total'],loss['ce']+loss['kl']+loss['hidden_weighted']+loss['rate'],rel_tol=1e-12)
assert math.isclose(loss['proposed_total'],2*loss['kl']+loss['hidden_weighted']+loss['rate'],rel_tol=1e-12)
assert math.isclose(loss['rate'],2*loss['soft_entropy_nats']/586459020,rel_tol=2e-7)
assert r['reconstruction']['old_relative_l2_error']<1e-4
assert r['reconstruction']['proposed_relative_l2_error']<1e-4
assert all(r['checks'].values())
assert all(v==134 for v in r['finite_gradient_tensors'].values())
rows=r['parameters']
assert len(rows)==134 and len({p['name'] for p in rows})==134
assert sum(p['name'].startswith('z/') for p in rows)==75
assert sum(p['name'].startswith('raw/') for p in rows)==59
for p in rows:
    assert math.prod(p['shape'])==p['elements']
    assert all(math.isfinite(v) and v>=0 for v in p['norms'])
    for field in ('component_pair_dots','recipe_to_data_dots'):
        assert all(math.isfinite(v) for v in p[field])
    for field in ('component_pair_cosines','recipe_to_data_cosines'):
        assert all(v is None or math.isfinite(v) and abs(v)<=1+1e-12 for v in p[field])
for k,g in r['groups'].items():
    selected=[p for p in rows if k=='all' or p['name'].startswith(k+'/')]
    for j,norm in enumerate(g['norms']):
        assert math.isclose(norm,math.sqrt(math.fsum(p['norms'][j]**2 for p in selected)),rel_tol=1e-12,abs_tol=1e-15)
    for field in ('component_pair_dots','recipe_to_data_dots'):
        for j,dot in enumerate(g[field]):
            assert math.isclose(dot,math.fsum(p[field][j] for p in selected),rel_tol=1e-10,abs_tol=1e-13)
e=r['execution']
assert e['child_exit_code']==0 and e['cpu_affinity']==[1]
assert e['total_cpu_seconds']<=180 and e['wall_seconds']<=240
assert e['address_space_limit_bytes']<=4 << 30 and e['reruns']==0
assert r['torch_threads']==r['torch_interop_threads']==1
d=r['lr_scaled_ordinary_direction']
for j,key in enumerate(('old_to_kl','old_to_ce','proposed_to_kl','proposed_to_ce')):
    expected=.01*r['groups']['z']['recipe_to_data_dots'][j]+.0001*r['groups']['raw']['recipe_to_data_dots'][j]
    assert math.isclose(expected,d['gradient_dots'][key],rel_tol=1e-12)
    assert d['directional_derivatives_under_negative_lr_scaled_gradient'][key]==-d['gradient_dots'][key]
assert d['gradient_dots']['old_to_kl']<0<d['gradient_dots']['proposed_to_kl']
scalar=d['largest_old_to_kl_negative_parameter']
selected=next(p for p in rows if p['name']==scalar['name'])
assert math.isclose(selected['recipe_to_data_dots'][0],scalar['ordinary_old_to_kl_dot'],rel_tol=1e-12)
assert math.isclose(r['groups']['all']['recipe_to_data_dots'][0]-scalar['ordinary_old_to_kl_dot'],scalar['rest_ordinary_old_to_kl_dot'],rel_tol=1e-12)
assert scalar['ordinary_old_to_kl_dot']<0<scalar['rest_ordinary_old_to_kl_dot']
print(json.dumps(dict(status='passed',targets=n,trainable_tensors=134,
                     ce_kl_cosine=r['groups']['all']['component_pair_cosines'][0],
                     lr_scaled_old_to_kl=d['gradient_dots']['old_to_kl'],
                     lr_scaled_proposed_to_kl=d['gradient_dots']['proposed_to_kl'],
                     top_level_student_forwards=r['top_level_student_forwards'],
                     optimizer_updates=r['optimizer_updates'])))
