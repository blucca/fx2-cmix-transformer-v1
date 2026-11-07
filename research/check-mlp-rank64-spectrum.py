"""One stdlib check of the published 24-matrix spectral result."""
import argparse, json, math
from pathlib import Path
base=Path(__file__).resolve().parent/'results'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--result',type=Path,default=base/'mlp-rank64-spectrum.json')
parser.add_argument('--matrices',type=Path,default=base/'mlp-rank64-original-matrices.json')
args=parser.parse_args()
r=json.loads(args.result.read_text())
a=r['per_matrix']
meta=json.loads(args.matrices.read_text())
census={x['name']:x for x in json.loads((base/'weights-census.json').read_text())['per_tensor']}
assert len(meta)==24
for item in meta:
    assert item['shape']==census[item['name']]['shape']
    assert item['histogram']==census[item['name']]['histogram']
assert {x['name'] for x in meta}=={x['name'] for x in a}
assert r['status']=='passed' and len(a)==24
assert r['model_forward_backward_training_codec_calls']==0
assert r['rank64_factor_optimization_or_calibration_calls']==0
assert r['quantized_histograms_match_original_census']
assert {x['name'] for x in a}=={f'blocks.{l}.mlp.{k}.weight.q' for l in range(12) for k in ['up','down']}
total=tail=0.0
for x in a:
    s=x['singular_values']
    assert len(s)==192 and all(math.isfinite(v) and v>=0 for v in s)
    assert all(u>=v for u,v in zip(s,s[1:]))
    norm=sum(v*v for v in s); t=sum(v*v for v in s[64:])
    assert math.isclose(norm,x['squared_frobenius'],rel_tol=1e-10)
    assert math.isclose(1-t/norm,x['retained_energy_fraction_rank64'],rel_tol=1e-10)
    assert math.isclose(math.sqrt(t/norm),x['best_rank64_relative_frobenius_error'],rel_tol=1e-10)
    total+=norm;tail+=t
assert math.isclose(1-tail/total,r['joint_retained_energy_fraction'],rel_tol=1e-10)
assert math.isclose(math.sqrt(tail/total),r['joint_best_rank64_relative_frobenius_error'],rel_tol=1e-10)
assert math.isclose(r['joint_retained_energy_fraction'],0.5622765778519693,rel_tol=1e-10)
print('PASS: all 24 original spectra, fixed rank64 tails and aggregate identity')
