"""One fixed rank64 fit on native train inputs; held inputs only score it.

The ridge is fixed at 1e-3 times mean input second moment.  It supplies a
weight-space prior for weakly observed directions; no centering or grid.
The two-factor INT4/INT8 path below follows native qmatvec's integer dot,
FP32 fold, and nearest-even activation quantization.  Each graph receives
the original model's inputs: chained replacement is a later full method.
"""
import argparse, json, math, time, resource
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--work', type=Path, required=True)
a = p.parse_args(); root = a.work
start = time.monotonic(); cpu = time.process_time()
meta = json.loads((root/'original-matrices.json').read_text())
census = json.loads((Path(__file__).resolve().parent/'results/weights-census.json').read_text())
reference = {x['name']: x for x in census['per_tensor']}
raw = np.fromfile(root/'original-matrices.f32', dtype='<f4')
assert len(meta) == 24 and raw.size == 3538944 and np.isfinite(raw).all()

def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)

assert np.array_equal(bf16(np.array([1.0, 1.0+2**-8, 1.0+3*2**-8], np.float32)),
                      np.array([1.0, 1.0, 1.0+2**-6], np.float32))

def weight_quant(w):
    f = np.asarray(w, dtype=np.float32)
    scale = bf16(np.max(np.abs(f), axis=1)/np.float32(7))
    assert np.isfinite(scale).all() and np.all(scale > 0)
    q = np.clip(np.rint(f/scale[:,None]), -7, 7).astype(np.int8)
    return q, scale

def linear(qx, qw, act, rows):
    # These dimensions keep the integer dot below 2^24; FP64 BLAS is exact.
    dot = np.asarray(qx, dtype=np.float64) @ np.asarray(qw.T, dtype=np.float64)
    assert np.max(np.abs(dot)) < 2**24 and np.equal(dot, np.rint(dot)).all()
    fold = np.float32(act)*np.asarray(rows, dtype=np.float32)
    return np.asarray(dot, dtype=np.float32)*fold[None,:]

def read(path, rows, cols):
    x = np.fromfile(path, dtype='<f4')
    assert x.size == rows*cols and np.isfinite(x).all(), (path, x.size)
    return x.reshape(rows, cols)

results = []; factors = {}; offset = 0
for item in meta:
    name = item['name']; shape = item['shape']; count = math.prod(shape)
    ref = reference[name]
    assert shape == ref['shape'] and item['histogram'] == ref['histogram']
    w = raw[offset:offset+count].reshape(shape).astype(np.float64); offset += count
    layer = int(name.split('.')[1]); kind = name.split('.')[3]
    act = np.float32(item['activation_scale_f32'])
    rowscale = np.asarray(item['row_scales_f32'], dtype=np.float32)
    assert act > 0 and np.isfinite(act) and rowscale.size == shape[0]
    qw = np.rint(w/rowscale[:,None]).astype(np.int8)
    assert np.equal(w, np.asarray(qw, np.float32)*rowscale[:,None]).all()
    assert np.bincount(qw.ravel().astype(np.int32)+7, minlength=15).tolist() == item['histogram']
    roles = {}
    for role, n in [('train',1199), ('held',1606)]:
        folder = root/role
        qx = read(folder/f'{layer:02d}_mlp_{kind}_input_int8.f32', n, shape[1])
        assert np.equal(qx, np.rint(qx)).all() and np.all((qx >= -128) & (qx <= 127))
        suffix = 'mlp_up_out' if kind == 'up' else 'mlp_out'
        y = read(folder/f'{layer:02d}_{suffix}.f32', n, shape[0])
        predicted = linear(qx, qw, act, rowscale)
        assert np.array_equal(predicted.view(np.uint32), y.view(np.uint32)), (name, role, float(np.max(np.abs(predicted-y))))
        roles[role] = (qx, qx.astype(np.float64)*float(act), y)
    x = roles['train'][1]
    c = x.T @ x / x.shape[0]
    ridge = 1e-3*float(np.trace(c))/shape[1]
    assert ridge > 0 and np.isfinite(ridge)
    creg = c + ridge*np.eye(shape[1])
    e, v = np.linalg.eigh(creg)
    assert np.all(e > 0)
    half = (v*np.sqrt(e)) @ v.T
    invhalf = (v/np.sqrt(e)) @ v.T
    u, s, vh = np.linalg.svd(w @ half, full_matrices=False)
    left = u[:,:64]*np.sqrt(s[:64])
    right = (np.sqrt(s[:64])[:,None]*vh[:64]) @ invhalf
    fitted = left @ right; delta = w-fitted
    objective = float(np.sum((delta @ creg)*delta))
    tail = float(np.sum(s[64:]**2))
    assert abs(objective-tail) <= 1e-8*max(1,tail)
    qr, sr = weight_quant(right); ql, sl = weight_quant(left)
    z = linear(roles['train'][0], qr, act, sr)
    midscale = bf16(np.float32(2/math.sqrt(127))*np.mean(np.abs(z), dtype=np.float32)).item()
    assert midscale > 0 and np.isfinite(midscale)
    record = dict(name=name, shape=shape, ridge_lambda=ridge,
        weighted_regularized_tail_fraction=tail/float(np.sum(s*s)),
        actual_original_outputs_bit_exact=True, intermediate_activation_scale_f32=midscale,
        factors=[])
    for label, q, scale, f in [('right',qr,sr,right), ('left',ql,sl,left)]:
        prefix = name.removesuffix('.weight.q')+'.'+label
        factors[prefix+'.q'] = q; factors[prefix+'.row_scale'] = scale
        factors[prefix+'.float'] = np.asarray(f, np.float32)
        record['factors'].append(dict(side=label, shape=list(q.shape),
            histogram=np.bincount(q.ravel().astype(np.int32)+7,minlength=15).tolist()))
    factors[name+'.first_activation_scale'] = np.asarray(act, np.float32)
    factors[name+'.intermediate_activation_scale'] = np.asarray(midscale, np.float32)
    for role, (qx, xf, y) in roles.items():
        floating = xf @ right.T @ left.T
        first = linear(qx, qr, act, sr)
        middle = np.clip(np.rint(first/np.float32(midscale)), -128,127).astype(np.int8)
        quantized = linear(middle, ql, midscale, sl)
        energy = float(np.sum(y.astype(np.float64)**2))
        assert energy > 0 and np.isfinite(floating).all() and np.isfinite(quantized).all()
        float_error = float(np.sum((floating-y)**2))
        quant_error = float(np.sum((quantized.astype(np.float64)-y)**2))
        record[role] = dict(rows=qx.shape[0], original_output_squared_energy=energy,
            fp64_rank64_output_squared_error=float_error,
            fp64_rank64_relative_squared_error=float_error/energy,
            int4_int8_output_squared_error=quant_error,
            int4_int8_relative_squared_error=quant_error/energy,
            intermediate_clipped_fraction=float(np.mean((first/np.float32(midscale) < -128)|(first/np.float32(midscale) > 127))))
    results.append(record)
    (root/'analysis-progress.json').write_text(json.dumps({'completed_matrices':len(results),'total':24})+'\n')
assert offset == raw.size
np.savez_compressed(root/'rank64-local-factors.npz', **factors)
summary = {}
for kind in ['up','down']:
    selected = [r for r in results if f'.mlp.{kind}.' in r['name']]
    summary[kind] = {}
    for role in ['train','held']:
        denominator = sum(r[role]['original_output_squared_energy'] for r in selected)
        summary[kind][role] = {
            'fp64_relative_squared_error':sum(r[role]['fp64_rank64_output_squared_error'] for r in selected)/denominator,
            'int4_int8_relative_squared_error':sum(r[role]['int4_int8_output_squared_error'] for r in selected)/denominator}
result = dict(status='passed', scope='24 local linear maps on original native inputs; one train piece and one held piece',
    rank=64, ridge_rule='1e-3 * trace(uncentered_second_moment) / input_dimension',
    train_rows=1199, held_rows=1606, held_rows_in_fit_or_calibration=0,
    original_weight_histograms_match_census=True, original_outputs_bit_exact_all_48_cases=True,
    full_replacement_model_forwards=0, optimizer_updates=0, native_factor_family_calls=0, codec_calls=0,
    factor_coordinates=sum(v.size for k,v in factors.items() if k.endswith('.q')),
    summary=summary, per_matrix=results, analysis_cpu_seconds=time.process_time()-cpu,
    analysis_wall_seconds=time.monotonic()-start,
    peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
assert result['factor_coordinates'] == 1474560
(root/'analysis-result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['per_matrix']},allow_nan=False), flush=True)
