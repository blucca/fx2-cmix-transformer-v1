#!/usr/bin/env python3
"""Standard-library arithmetic check for the published teacher-only e24 stage."""
import json
from math import isclose
from pathlib import Path

root = Path(__file__).parent / 'results'
current = json.loads((root / 'teacher-only-native-e24.json').read_text())
prior = json.loads((root / 'teacher-only-native-e20.json').read_text())

def close(a, b):
    assert isclose(a, b, rel_tol=0, abs_tol=1e-12), (a, b)

assert current['epoch'] == current['shared_contract']['same_epoch'] == 24
assert current['recipes'] == prior['recipes']
contract = dict(current['shared_contract'], same_epoch=20)
assert contract == prior['shared_contract']
domains = current['domains']
assert [(d['targets'], d['pieces']) for d in domains] == [(1082505, 473), (107647, 3), (131071, 1)]
assert [d['scope'] for d in domains] == ['full_held', 'held_long', 'giant']
for d, p in zip(domains, prior['domains']):
    for k in ('baseline_W0_raw_nll', 'old_CE_KL_e20_raw_nll', 'fresh_teacher_only_e20_raw_nll'):
        close(d[k], p[k])
    o, f, w = d['old_CE_KL_e24_raw_nll'], d['fresh_teacher_only_e24_raw_nll'], d['baseline_W0_raw_nll']
    for k, expected in (
        ('fresh_minus_old_raw_nll', f-o), ('fresh_minus_W0_raw_nll', f-w),
        ('old_minus_W0_raw_nll', o-w),
        ('fresh_e24_minus_fresh_e20_raw_nll', f-d['fresh_teacher_only_e20_raw_nll']),
        ('old_e24_minus_old_e20_raw_nll', o-d['old_CE_KL_e20_raw_nll']),
    ):
        close(d[k], expected)
    close(d['fresh_minus_old_target_nll_sum'], d['targets']*(f-o))
full, long, giant = domains
r = current['derived_fullheld_remainder']
assert full['targets'] + giant['targets'] == current['shared_contract']['independent_scored_targets'] == 1213576
assert r['targets'] == full['targets'] - long['targets'] == 974858
assert r['pieces'] == full['pieces'] - long['pieces'] == 470
for k in ('baseline_W0_raw_nll', 'old_CE_KL_e24_raw_nll', 'fresh_teacher_only_e24_raw_nll',
          'old_CE_KL_e20_raw_nll', 'fresh_teacher_only_e20_raw_nll'):
    close(r[k], (full[k]*full['targets']-long[k]*long['targets']) / r['targets'])
for k, expected in (
    ('fresh_minus_old_raw_nll', r['fresh_teacher_only_e24_raw_nll']-r['old_CE_KL_e24_raw_nll']),
    ('fresh_minus_W0_raw_nll', r['fresh_teacher_only_e24_raw_nll']-r['baseline_W0_raw_nll']),
    ('old_minus_W0_raw_nll', r['old_CE_KL_e24_raw_nll']-r['baseline_W0_raw_nll']),
    ('fresh_e24_minus_fresh_e20_raw_nll', r['fresh_teacher_only_e24_raw_nll']-r['fresh_teacher_only_e20_raw_nll']),
    ('old_e24_minus_old_e20_raw_nll', r['old_CE_KL_e24_raw_nll']-r['old_CE_KL_e20_raw_nll']),
):
    close(r[k], expected)
assert abs(r['fresh_minus_old_raw_nll']*r['targets'] + long['fresh_minus_old_target_nll_sum']-full['fresh_minus_old_target_nll_sum']) < 1e-8
w0, old, fresh = current['baseline_weight_bytes'], current['original_e24_weight_bytes'], current['fresh_e24_weight_bytes']
assert (w0, old, fresh) == (2904325, 1963088, 1961211)
assert current['old_double_weight_saved_bytes'] == 2*(w0-old) == 1882474
assert current['fresh_double_weight_saved_bytes'] == 2*(w0-fresh) == 1886228
assert current['fresh_double_weight_charge_improvement_over_old_bytes'] == 2*(old-fresh) == 3754
assert current['fresh_e20_weight_bytes'] == prior['fresh_e20_weight_bytes']
assert current['original_e20_weight_bytes'] == prior['original_e20_weight_bytes']
assert current['fresh_weight_growth_from_e20_bytes'] == fresh-prior['fresh_e20_weight_bytes'] == 432
assert current['provenance']['old_e24_step'] == 1894
assert current['provenance']['old_stage_kind'] == 'hard'
assert current['provenance']['old_source'].endswith('evaluations[epoch=24]')
assert all(v == 0 for v in current['reporting_operations'].values())
print(json.dumps({'status': 'passed', 'native_targets': 1213576, 'derived_held_pieces': 470,
                  'derived_held_targets': 974858, 'double_weight_charge_improvement_bytes': 3754,
                  'model_runs': 0, 'native_runs': 0, 'remote_state_reads': 0}))
