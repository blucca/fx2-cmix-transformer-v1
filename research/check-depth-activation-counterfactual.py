#!/usr/bin/env python3
"""Minimal stdlib JSON replay of the existing two-forward activation readout."""
import argparse
import json
import math
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--result',type=Path,default=Path(__file__).resolve().parent/'results/depth-activation-counterfactual.json')
a=p.parse_args()
r=json.loads(a.result.read_text())
assert r['status']=='passed' and r['B_status']=='passed'
assert len(r['activation_quantizers'])==80
assert r['top_level_student_forwards']==2 and r['checkpoint_reads']==1
assert r['counts']['z_tensors']==75 and r['counts']['raw_tensors']==59
assert r['counts']['activation_quantizers']==80 and r['counts']['weight_quantizers']==75
assert r['counts']['mutable_tensors']==134 and r['counts']['model_state_tensors']==289
assert r['piece']['targets']==1199 and r['piece']['vocabulary_columns']==205 and r['piece']['bounds']==[[0,1199]]
assert r['native_scaled_gemm'] and r['hard_mode'] and r['dropout']==0 and all(r['checks'].values())
for key in ('optimizer_updates','optimizer_instances','gradient_calls','new_teacher_calls','new_cache_files',
            'capture_calls','native_calls','codec_calls','model_exports','scale_searches','scale_calibrations','source_writes'):
    assert r[key]==0,key
fields=('calls','elements','below_qmin','above_qmax','rounded_below_qmin','rounded_above_qmax',
        'round_to_zero','nonzero_round_to_zero','input_zero','output_zero',
        'quantization_error_squared_sum','input_squared_sum','out_of_range_error_squared_sum')
for row in r['activation_quantizers']:
    layer=row['student_layer']
    assert row['teacher_layer']==([0,1,2,3,8,9,10,11][layer] if layer is not None else None)
    for label in ('A','B'):
        m=row[label]
        for key in fields[2:10]:
            assert 0<=m[key]<=m['elements']
        assert m['below_qmin']+m['above_qmax']<=m['elements']
        for key in fields[10:]:
            assert math.isfinite(m[key]) and m[key]>=0
for label in ('A','B'):
    total=r['activation_aggregates'][label]
    for key in fields:
        assert math.isclose(total[key],sum(row[label][key] for row in r['activation_quantizers']),rel_tol=1e-12,abs_tol=1e-12)
    losses=r['losses'][label]
    assert all(math.isfinite(v) for v in losses.values())
    assert math.isclose(losses['old_data_objective'],losses['ce']+losses['kl']+losses['hidden_weighted'])
    assert math.isclose(losses['proposed_2KL_data_objective'],2*losses['kl']+losses['hidden_weighted'])
    for key,value in r['loss_reconstruction'][label]['float64_reconstruction'].items():
        assert math.isclose(losses[key],value,rel_tol=2e-5,abs_tol=2e-6)
for key in r['losses']['A']:
    delta=r['losses']['A'][key]-r['losses']['B'][key]
    assert math.isclose(r['loss_delta_A_minus_B'][key],delta)
    assert math.isclose(r['loss_delta_B_minus_A'][key],-delta)
assert r['activation_aggregates']['B']['quantization_error_squared_sum']==0
account=r['execution']
assert 0<account['total_cpu_seconds']<=180 and 0<account['wall_seconds']<=240
assert account['address_space_limit_bytes']==3<<30 and account['cpu_affinity']==[1]
assert math.isclose(account['child_cpu_seconds']+account['supervisor_cpu_seconds'],account['total_cpu_seconds'])
print(json.dumps(dict(status='passed',quantizer_rows=80,forward_pairs=2,neural_network_reruns=0,
    checks=['80 rows/aggregates','A/B objectives/deltas/loss reconstruction','resource and unchanged-state contracts']),separators=(',',':')))
