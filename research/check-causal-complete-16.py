"""Check the complete causal trajectory and same-W native-to-range accounting."""
import hashlib, json, math
from pathlib import Path
r=json.loads((Path(__file__).resolve().parent/'results/causal-complete-16-mixed.json').read_text())
assert r['status']=='passed'
t=r['training'];s=r['frozen_selection'];a=r['complete_trajectory']
assert (t['epoch'],t['step'],t['seen_targets'])==(16,1267,47060912)
assert t['seen_targets']==16*t['source_targets_per_epoch']
assert [x['epoch'] for x in a]==list(range(17))
assert max(a,key=lambda x:x['proxy_net_saved_bytes'])['epoch']==s['epoch']==2
assert s['step']==159 and s['weight_bytes']==2904371
w=r['same_W'];delta=2*(w['baseline_weight_bytes']-w['candidate_weight_bytes'])
assert delta==-92==s['double_weight_saved_bytes']
weight=Path(__file__).resolve().parent/'assets/causal-e02.tfwc3'
assert weight.stat().st_size==s['weight_bytes']
assert hashlib.sha256(weight.read_bytes()).hexdigest()==w['selected_weight_identity']
p={x['step']:x for x in r['production']}
for name,size in [('prefix8',8388608),('q2',1048576)]:
    b=p[name+'-baseline'];c=p[name+'-candidate']
    assert b['status']==c['status']=='passed'
    assert c['decode']['output_bytes']==size
    assert c['decode']['restored_slice_sha256']==b['decode']['restored_slice_sha256']
    saved=b['encode']['output_bytes']-c['encode']['output_bytes']
    assert saved==r['layers'][name]['actual_full_payload_saved_bytes']
    assert r['full_files_and_flush'][name]['baseline']['end_flush_bytes']==r['full_files_and_flush'][name]['candidate']['end_flush_bytes']==1
for v in r['layers'].values():
    g=v['guarded_transfer'];bits=v['u16_saved_bits']+g['fallback_final_saved_bits']
    assert math.isclose(v['actual_interval_saved_bytes'],bits/8+g['range_residual_saved_bytes'],abs_tol=1e-8)
    assert math.isclose(v['u16_saved_bits']/v['G_saved_bits'],g['kappa_final_nn_targets'],rel_tol=1e-12)
    assert math.isclose(8*v['actual_interval_saved_bytes']/v['G_saved_bits'],g['kappa_interval_bytes'],rel_tol=1e-12)
    assert v['overrides']['baseline_bits']==v['overrides']['candidate_bits']
c=r['two_sample_component'];assert c['sample_payload_saved_bytes']==697+221==918
assert c['paired_sample_component_saved_bytes']==918+delta==826
q=r['layers']['q2-non-direct-update-input'];assert q['all_input_bytes']==q['NNtargets']+q['fallback']==1037391
assert q['baseline_interval_bytes']-q['candidate_interval_bytes']==212
x=r['exposure'];assert x['q2_current_train_input_bytes']==11185 and x['q2_current_train_targets']==11184
assert r['layers']['q2']['NNtargets']-q['NNtargets']==11184
h=r['historical_margin_scenario'];projection=h['N_bytes']*98/h['mature_interval_bytes']+delta
assert math.isclose(projection,h['projected_mature_net_saved_bytes'],abs_tol=1e-8)
assert math.ceil((h['Rgap_bytes']-delta)*h['mature_interval_bytes']/h['N_bytes'])==h['required_integer_mature_saved_bytes']==1692
assert h['scenario_margin_gate']=='failed' and h['decision_for_this_named_scenario']=='stop_new_fullS'
assert r['actual_formal_package']['actual_S0_package_stat_bytes'] is None
assert r['actual_formal_package']['actual_S1_package_stat_bytes'] is None
assert r['attribution_readout']['model_forward_backward_training_codec_reruns']==0
print('PASS: full16, frozen e2, four credit layers, exposure, flush and named gate')
