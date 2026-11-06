"""Read the existing tensor census; create no model or new weight file."""
from pathlib import Path
import json,math,re
repo=Path(__file__).resolve().parents[1]
census=json.loads((repo/'research/results/weights-census.json').read_text())
groups={i:{'int4_elements':0,'static4096_estimate_bytes':0.,'matrices':0} for i in range(12)}
shared=0.
for t in census['per_tensor']:
 m=re.match(r'blocks\.(\d+)\.',t['name'])
 if not m:shared+=t['static4096_with_28B_table_bits']/8;continue
 g=groups[int(m[1])];g['int4_elements']+=t['elements'];g['matrices']+=1
 g['static4096_estimate_bytes']+=t['static4096_with_28B_table_bits']/8
assert sum(g['int4_elements'] for g in groups.values())+118080==5868864
assert sum(g['matrices'] for g in groups.values())+3==111
n=586459020;gap=952225
cases=[]
for removed in [[4,5],[4,5,6],[4,5,6,7]]:
 saved=sum(groups[i]['static4096_estimate_bytes'] for i in removed)
 budget=2*saved-gap
 cases.append({'removed_original_layers':removed,'student_depth':12-len(removed),
  'single_weight_int4_estimate_saved_bytes':saved,'two_copy_int4_estimate_saved_bytes':2*saved,
  'historical_payload_regression_allowance_bytes':budget,
  'historical_payload_allowance_bits_per_prepared_byte':budget*8/n,
  'historical_payload_allowance_nats_per_prepared_byte':budget*8*math.log(2)/n})
assert cases[0]['historical_payload_regression_allowance_bytes']<0
assert 850000<cases[-1]['historical_payload_regression_allowance_bytes']<880000
out={'schema':'hutter.depth-static-census-budget.v1','scope':'Existing static4096 INT4 census estimate. Adaptive names/scale/F32 metadata, final range rounding, retrained symbol distribution and new binary size enter actual W/B.','prepared_bytes':n,'historical_record_gap_bytes':gap,'actual_original_weight_bytes':2904325,'actual_common_two_copy_charge_bytes':6573938,'actual_new_weight_bytes':None,'new_nn_calls':0,'layers':groups,'shared_int4_estimate_bytes':shared,'cases':cases,
 'recommended_student':{'retained_original_layers':[0,1,2,3,8,9,10,11],'removed_original_layers':[4,5,6,7],'d_model':192,'d_mlp':768,'layer_types':['KDA','KDA','KDA','attention']*2,'skip_weight_source_indices':[2,3,4,5],'int4_matrices':75},
 'actual_net_saved_formula':'(P0-P1)+2*(B0-B1)+2*(D0-D1)+(O0-O1)+2*(W0-W1)',
 'decision':'Defer new complete distillation until the current causal whole-stream measurements; retain one coherent eight-block candidate.'}

print(json.dumps(out,indent=2))
