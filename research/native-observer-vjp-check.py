"""Scalar loss and VJP contract from frozen native observer indices."""
import json
import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'native-byte-path'
N = 100001

def centre(row):
    o = row['override']
    return float(o) if o in (0, 1) else (row['lookup_index'] + .5) / N

def kl(t, s):
    if t == s:
        return 0.0
    assert 0 < t < 1 and 0 < s < 1
    return t * math.log(t / s) + (1-t) * math.log1p((s-t) / (1-s))

def normal_vjp(t, s):
    return (s-t)/(s*(1-s))

pairs = {}
for line in (ROOT / 'observations-192.jsonl').read_text().splitlines():
    r = json.loads(line)
    key = (r['regime'], r['piece'], r['row'], r['depth_1_based'])
    pairs.setdefault(key, {})[r['side']] = r
zero = 0
changes = []
for key, pair in pairs.items():
    t, s = centre(pair['teacher']), centre(pair['oracle'])
    value = kl(t, s)
    if t == s:
        zero += 1
        if 0 < s < 1:
            assert normal_vjp(t, s) == 0
    else:
        grad = normal_vjp(t, s)
        eps = 1e-7
        fd = (kl(t, s+eps)-kl(t, s-eps))/(2*eps)
        assert abs(fd-grad) < 2e-9
        changes.append({'key': list(key), 'teacher_r':t, 'oracle_r':s,
                        'bernoulli_kl':value, 'identity_shadow_vjp':grad,
                        'finite_difference_in_relaxed_coordinate':fd})
# Controlled clip case. An exact clamp's chain derivative is 0 here; an
# identity shadow gives the nonzero KL feedback needed to leave the plateau.
t, s, shadow_q = 20.5/N, 10.5/N, 1e-5
clip_case = {'source':'controlled scalar case, not native measurement',
             'teacher_index':20, 'student_index':10, 'shadow_q':shadow_q,
             'loss':kl(t,s), 'ordinary_clip_chain_vjp':0.0,
             'identity_shadow_vjp':normal_vjp(t,s)}
assert len(pairs) == 96 and zero == 95 and len(changes) == 1
assert changes[0]['bernoulli_kl'] > 0 and changes[0]['identity_shadow_vjp'] > 0
assert clip_case['loss'] > 0 and clip_case['identity_shadow_vjp'] < 0
# This FP32 control checks the STE expression's forward value.
f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
r, q = f32(10.5/N), f32(.99)
reconstructed = f32(q + f32(r-q))
zero_added = f32(r + f32(q-q))
assert reconstructed != r and zero_added == r
ste_control = {'source':'controlled FP32 arithmetic', 'actual_center':r,
               'shadow':q, 'subtract_then_add':reconstructed,
               'zero_shadow_add':zero_added}
result = {'ste_forward_value_control':ste_control, 'scope':'scalar loss/VJP over saved native observer indices',
          'saved_native_pairs':len(pairs), 'zero_loss_pairs':zero,
          'changed_pairs':changes, 'clip_case':clip_case,
          'new_native_runs':0, 'model_forwards':0, 'capture_runs':0,
          'learner_implementation':0, 'check':'passed'}
print(json.dumps(result, indent=2))
