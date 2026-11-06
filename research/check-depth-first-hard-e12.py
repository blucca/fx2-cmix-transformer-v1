"""Replay the measured continuous-to-hard deployment transition, without a model."""
import json, math
from pathlib import Path
r=json.loads((Path(__file__).with_name('results')/'depth-first-hard-e12.json').read_text())
n=r['native']; ps=r['paired_pieces']; N=sum(p[0] for p in ps)
assert len(ps)==473 and N==1082505
for c,k in [(1,'original_nll'),(2,'e8_nll'),(3,'e12_nll')]:
    assert math.isclose(sum(p[c] for p in ps)/N,n['held'][k],abs_tol=1e-12)
for x in n.values():
    assert math.isclose(x['e12_nll']-x['original_nll'],x['delta_w0'],abs_tol=1e-14)
    assert math.isclose(x['e12_nll']-x['e8_nll'],x['delta_e8'],abs_tol=1e-14)
for k in ['original_nll','e8_nll','e12_nll']:
    assert math.isclose(sum(n[s]['targets']*n[s][k] for s in ['held_short','held_long'])/N,n['held'][k],abs_tol=1e-14)
assert [n[s]['targets'] for s in ['held_short','held_long','maximum_length_audit']]==[974858,107647,131071]
for k,c in [('w0',1),('e8',2)]:
    s=r['held_piece_pairing'][k]
    assert sum(s[f'{x}_pieces'] for x in ['better','worse','equal'])==473
    assert s['better_targets']+s['worse_targets']==N
    assert s['better_pieces']==sum(p[3]<p[c] for p in ps)
    assert s['better_targets']==sum(p[0] for p in ps if p[3]<p[c])
    assert math.isclose(sum(p[3]-p[c] for p in ps)/N,s['weighted_delta_nats'],abs_tol=1e-12)
ph=r['phase']; es=r['online_training_epochs']
assert [e['epoch'] for e in es]==list(range(1,13))
assert all((e['targets'],e['pieces'])==(2941307,1313) for e in es)
assert (ph['continuous_epochs_completed'],ph['hard_epochs_completed'],ph['goal_epochs'])==(8,4,32)
assert all(e['stage']==('continuous' if e['epoch']<=8 else 'hard') for e in es)
assert sum(e['optimizer_steps'] for e in es)==ph['complete_optimizer_steps']==949
assert ph['recorded_optimizer_steps']==949+ph['partial_optimizer_steps']==956
assert ph['recorded_seen_targets']==12*2941307+ph['partial_targets']==35554719
b=r['actual_package']; space=2*(b['original_weight_bytes']-b['e12_weight_bytes']+b['original_binary_bytes']-b['student_binary_bytes'])
assert space==b['double_weight_plus_binary_saved_bytes']==1895710
s=r['named_size_sensitivity']; penalty=s['full_native_targets']*n['held']['delta_w0']/(8*math.log(2))
assert math.isclose(penalty,s['kappa1_payload_increase_bytes'],abs_tol=1e-6)
assert math.isclose(s['historical_record_kappa']*penalty-space,-s['historical_record_gap_bytes'],abs_tol=1e-6)
g=r['e12_coordinate_geometry']; assert g['max_scale_delta']==0
assert g['changed_z_coordinates']-g['changed_weight_symbols']==g['displaced_z_still_rounding_to_initial_symbol']==2424944
reg=r['held_surprisal_regimes']; assert sum(x['targets'] for x in reg)==N
assert math.isclose(sum(x['hard_stage_delta_sum'] for x in reg)/N,n['held']['delta_e8'],abs_tol=1e-12)
cpu=r['actual_experiment_cpu']; assert math.isclose(cpu['latest_reported_total_cpu_seconds']+cpu['method_cap_remaining_seconds'],172800,abs_tol=1e-9)
assert r['static15_export_contract']['all_tensor_bytes_equal']
print(json.dumps(dict(status='passed',hard_epochs=4,paired_pieces=473,held_hard_stage_delta=n['held']['delta_e8'],double_W_B_saved_bytes=space,historical_scenario_kappa=s['historical_record_kappa']),indent=2))
