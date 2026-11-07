#!/usr/bin/env python3
"""Stdlib replay of measured objective drift and a matched-curvature ablation."""
import json
import math
from pathlib import Path


r = json.loads((Path(__file__).resolve().parent /
                'results/teacher-objective-audit.json').read_text())
assert r['schema'] == 'hutter.teacher-objective-audit.v1'
assert r['proposed_method_status'] == 'unrun' and r['new_model_or_codec_runs'] == 0


def softmax(z):
    values = [math.exp(x - max(z)) for x in z]
    total = math.fsum(values)
    return [x / total for x in values]


# Synthetic logits isolate output curvature; model/STE/Adam updates have their
# own coordinates and are measured by a complete future learning trajectory.
teacher = [math.sin(i*.17) + (i-102)*.013 for i in range(205)]
q = softmax(teacher)
y = [float(i == 17) for i in range(205)]
x = [a + math.cos(i*.11)*.2 for i, a in enumerate(teacher)]
direction = [math.sin(i*.23) for i in range(205)]


def gradient(z, pure):
    p = softmax(z)
    return [2*p[i] - (2*q[i] if pure else y[i]+q[i]) for i in range(205)]


p = softmax(x)
mean_direction = math.fsum(a*b for a, b in zip(p, direction))
hv = [2*p[i]*(direction[i]-mean_direction) for i in range(205)]
eps = 1e-5
plus = [a+eps*b for a, b in zip(x, direction)]
minus = [a-eps*b for a, b in zip(x, direction)]
for pure in (False, True):
    gp, gm = gradient(plus, pure), gradient(minus, pure)
    assert max(abs((a-b)/(2*eps)-c) for a, b, c in zip(gp, gm, hv)) < 1e-9
assert max(abs(a) for a in gradient(teacher, True)) < 1e-14
assert max(abs(a-(b-c)) for a, b, c in zip(gradient(teacher, False), q, y)) < 1e-14

epochs = {row['epoch']: row for row in r['epochs']}
for row in epochs.values():
    assert row['targets'] == r['training_targets']
    rate = 2*row['soft_entropy_target_sum_nats']/row['targets']/r['prepared_bytes']
    objective = row['ce_mean']+row['teacher_kl_mean']+.1*row['final_hidden_mse_mean']+rate
    assert abs(rate-row['rate_mean']) < 1e-12
    assert abs(objective-row['objective_mean']) < 1e-12
    assert abs(objective-row['objective_target_sum']/row['targets']) < 1e-12
a, b = epochs[12], epochs[16]
d = r['e16_minus_e12']
for key, field, scale in (('ce', 'ce_mean', 1), ('teacher_kl', 'teacher_kl_mean', 1),
                          ('weighted_hidden', 'final_hidden_mse_mean', .1),
                          ('rate', 'rate_mean', 1), ('objective', 'objective_mean', 1)):
    assert abs(d[key]-scale*(b[field]-a[field])) < 1e-12
assert d['ce'] < 0 and d['objective'] < 0
assert all(d[k] > 0 for k in ('teacher_kl', 'weighted_hidden', 'rate', 'held_native'))
assert r['existing_teacher_cache_f32_data_bytes'] == r['training_targets']*(205+192)*4
assert abs(r['estimated_cache_hit_32epoch_cpu_seconds']-
           32*r['cache_hit_seconds_per_epoch_including_existing_native_cadence']) < 1e-8
print(json.dumps(dict(status='passed', logit_curvature='2 Fisher',
                     observed_online_objective_delta=d['objective'],
                     observed_held_native_delta=d['held_native'],
                     proposed_method_status='unrun')))
