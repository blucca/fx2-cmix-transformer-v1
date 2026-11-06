#!/usr/bin/env python3
"""Recompute frozen native-loss domains from the supplied per-piece accumulators."""
import json
import math
from pathlib import Path

results = Path(__file__).with_name('results')
summary = json.loads((results / 'native-loss-regimes.json').read_text())
data = json.loads((results / 'native-loss-regimes-pieces.json').read_text())
rows = data['rows']
assert len(rows) == 473 and sum(r[0] for r in rows) == 1082505
assert data['columns'] == ['targets', 'w0_mean_nats', 'w0_sum_nats',
    'causal_e1_sum_nats', 'causal_e2_sum_nats', 'causal_e6_sum_nats',
    'depth_e4_sum_nats', 'depth_e8_sum_nats']
assert all(r[0] > 0 and all(math.isfinite(x) for x in r) for r in rows)
assert all(math.isclose(r[1], r[2] / r[0], abs_tol=1e-12) for r in rows)
N = summary['scope']['held_targets']
base_sum = math.fsum(r[2] for r in rows)
contrasts = {'c2_minus_w0': (4, 2), 'c2_minus_c1': (4, 3),
             'c6_minus_c2': (5, 4), 'd8_minus_d4': (7, 6),
             'd8_minus_w0': (7, 2)}
for group in [summary['all'], *summary['regimes']]:
    selected = rows if group is summary['all'] else [r for r in rows
        if r[1] >= group['lower_inclusive'] and
        (group['upper_exclusive'] is None or r[1] < group['upper_exclusive'])]
    targets = sum(r[0] for r in selected)
    assert (len(selected), targets) == (group['pieces'], group['targets'])
    assert math.isclose(targets / N, group['target_share'], abs_tol=1e-14)
    b = math.fsum(r[2] for r in selected)
    assert math.isclose(b / targets, group['w0_native_nll'], abs_tol=1e-12)
    assert math.isclose(b / base_sum, group['w0_native_surprisal_share'], abs_tol=1e-12)
    for name, (a, z) in contrasts.items():
        d = math.fsum(r[a] - r[z] for r in selected)
        assert math.isclose(d / targets, group[name + '_nats_per_target'], abs_tol=1e-12)
    d = math.fsum(r[4] - r[3] for r in selected)
    assert math.isclose(d, group['c2_minus_c1_summed_nats'], abs_tol=1e-9)
    recovery = -math.fsum(r[7] - r[6] for r in selected)
    e4_gap = math.fsum(r[6] - r[2] for r in selected)
    assert math.isclose(recovery / e4_gap, group['depth_fraction_of_e4_gap_recovered'], abs_tol=1e-12)
    assert math.isclose(math.fsum(r[7] - r[2] for r in selected) / b,
                        group['d8_relative_to_w0_native_nll'], abs_tol=1e-12)
overlap = [r for r in rows if r[5] > r[4] and r[7] < r[6]]
assert (len(overlap), sum(r[0] for r in overlap)) == (356, 1012417)
assert summary['paired_causal_late_loss_and_depth_recovery'] == {'pieces': 356, 'targets': 1012417}

# This is one named sensitivity, using the same held composition and constant
# transmission inside each domain. Actual code length uses the original mixed run.
epochs = json.loads((results / 'causal-first-seven-epochs.json').read_text())['epochs']
fee = 2 * (epochs[1]['weight_bytes'] - epochs[0]['weight_bytes'])
assert fee == 84
factor = summary['scope']['full_nn_projection_targets'] / (N * 8 * math.log(2))
low, mid, high = [g['c2_minus_c1_summed_nats'] for g in summary['regimes']]
threshold = (low + mid + fee / factor) / -high
advantage = -factor * (low + mid + high) - fee
assert math.isclose(threshold, 0.37763684275259957, abs_tol=1e-12)
assert math.isclose(advantage, 9978.763038972884, abs_tol=1e-6)
print(json.dumps({'status': 'passed', 'pieces': len(rows), 'targets': N,
    'shared_late_loss_and_depth_recovery_targets': sum(r[0] for r in overlap),
    'named_e2_over_e1_all_kappa1_net_bytes': advantage,
    'named_high_kappa_break_even_with_low_mid_kappa1': threshold}, indent=2))
