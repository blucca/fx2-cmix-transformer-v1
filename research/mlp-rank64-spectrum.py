"""Best rank64 Frobenius error of original effective matrices; zero model forwards."""
import argparse, json, math, time, resource
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--work', required=True, help='scratch directory containing original-matrices.{json,f32}')
p=Path(parser.parse_args().work)
meta=json.loads((p/'original-matrices.json').read_text())
census=json.loads((Path(__file__).resolve().parent/'results/weights-census.json').read_text())
reference={x['name']:x for x in census['per_tensor']}
assert len(meta)==24
raw=np.fromfile(p/'original-matrices.f32',dtype='<f4')
assert raw.size==3538944 and np.isfinite(raw).all()
start=time.monotonic(); before=time.process_time(); offset=0; results=[]
for item in meta:
    ref=reference[item['name']]
    assert item['histogram']==ref['histogram'] and item['shape']==ref['shape']
    count=math.prod(item['shape']); w=raw[offset:offset+count].reshape(item['shape']).astype(np.float64)
    offset+=count
    s=np.linalg.svd(w,full_matrices=False,compute_uv=False)
    total=float(np.sum(w*w)); all_s=float(np.sum(s*s)); tail=float(np.sum(s[64:]**2))
    assert s.size==192 and total>0 and abs(total-all_s)<=1e-10*total
    results.append({'name':item['name'],'shape':item['shape'],'squared_frobenius':total,
        'retained_energy_fraction_rank64':1-tail/total,
        'best_rank64_relative_frobenius_error':math.sqrt(tail/total),
        'singular_values':s.tolist()})
    (p/'spectrum-progress.json').write_text(json.dumps({'completed':len(results),'total':24})+'\n')
assert offset==raw.size
weighted_tail=sum(x['squared_frobenius']*x['best_rank64_relative_frobenius_error']**2 for x in results)
total=sum(x['squared_frobenius'] for x in results)
result={'status':'passed','scope':'all 24 original dequantized MLP matrices; one fixed rank64 spectral diagnostic',
    'matrix_file':'models/6m-q4-fp32.tfwc2','quantized_histograms_match_original_census':True,
    'model_forward_backward_training_codec_calls':0,'rank64_factor_optimization_or_calibration_calls':0,
    'joint_retained_energy_fraction':1-weighted_tail/total,
    'joint_best_rank64_relative_frobenius_error':math.sqrt(weighted_tail/total),
    'svd_cpu_seconds':time.process_time()-before,'svd_wall_seconds':time.monotonic()-start,
    'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'per_matrix':results}
(p/'spectrum-result.json').write_text(json.dumps(result,indent=2)+'\n')
for group in ['up','down']:
    rows=[x for x in results if f'.mlp.{group}.' in x['name']]
    print(group,'rank64 energy min/mean/max',min(x['retained_energy_fraction_rank64'] for x in rows),
        sum(x['retained_energy_fraction_rank64'] for x in rows)/len(rows),max(x['retained_energy_fraction_rank64'] for x in rows))
print({k:v for k,v in result.items() if k!='per_matrix'})
