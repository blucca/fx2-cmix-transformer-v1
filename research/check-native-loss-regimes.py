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
# Precision is conditional on these observed distinct articles. Preserve the
# sampling support and selected-checkpoint scope stated in the compact result.
precision = json.loads((results / 'native-sampling-precision.json').read_text())
assert (precision['piece_article_count'], precision['targets']) == (len(rows), N)
paired_columns = [(4, 2), (4, 3), (5, 4), (7, 6), (7, 2)]
assert len(precision['comparisons']) == len(paired_columns)
for item, (a, b) in zip(precision['comparisons'], paired_columns):
    ds = [r[a] - r[b] for r in rows]
    total = math.fsum(ds)
    theta = total / N
    se = math.sqrt(len(rows) / (len(rows) - 1) *
        math.fsum((d - theta*r[0])**2 for r, d in zip(rows, ds))) / N
    deleted = [(total-d)/(N-r[0]) for r, d in zip(rows, ds)]
    assert math.isclose(theta, item['delta_nats_per_target'], abs_tol=1e-12)
    assert math.isclose(se, item['observed_article_ratio_standard_error'], abs_tol=1e-12)
    for actual, saved in zip([theta-1.96*se, theta+1.96*se], item['observed_support_normal_interval']):
        assert math.isclose(actual, saved, abs_tol=1e-12)
    for actual, saved in zip([min(deleted), max(deleted)], item['delete_one_article_delta_range']):
        assert math.isclose(actual, saved, abs_tol=1e-12)
    assert math.isclose(abs(theta)/se, item['absolute_studentized_ratio'], abs_tol=1e-10)
# Complete the same frozen partition over the original full 32-epoch depth
# trajectory. The independent giant remains outside the 473-article sample.
complete = json.loads((results / 'depth-complete-native-regimes.json').read_text())
trajectory = json.loads((results / complete['source']).read_text())
matrix, columns = trajectory['paired_pieces'], trajectory['paired_piece_columns']
assert len(matrix) == len(rows)
assert all((r[0], r[1]) == (old[0], old[2]) for r, old in zip(matrix, rows))
assert sum(g['targets'] for g in complete['regimes']) == N
for group in [complete['all'], *complete['regimes']]:
    selected = matrix if group is complete['all'] else [r for r in matrix
        if r[1]/r[0] >= group['lower_inclusive'] and
        (group['upper_exclusive'] is None or r[1]/r[0] < group['upper_exclusive'])]
    n = sum(r[0] for r in selected)
    assert (len(selected), n) == (group['pieces'], group['targets'])
    scores = {str(e['epoch']): math.fsum(r[columns.index(f"e{e['epoch']}_logits_sum")]
        for r in selected)/n for e in trajectory['exports']}
    for epoch, score in scores.items():
        assert math.isclose(score, group['native_nll_by_epoch'][epoch], abs_tol=1e-12)
    assert int(min((e for e in scores if int(e)>0), key=lambda e: scores[e])) == group['best_positive_epoch']
    ds = [r[columns.index('e32_logits_sum')]-r[columns.index('e12_logits_sum')] for r in selected]
    contrast = group['e32_minus_e12']
    assert math.isclose(math.fsum(ds), contrast['summed_nats'], abs_tol=1e-8)
    assert math.isclose(math.fsum(ds)/n, contrast['nats_per_target'], abs_tol=1e-12)
    for label, predicate in [('improved', lambda d: d<0), ('worsened', lambda d: d>0)]:
        assert sum(predicate(d) for d in ds) == contrast[label+'_pieces']
        assert sum(r[0] for r,d in zip(selected,ds) if predicate(d)) == contrast[label+'_targets']
    if group is not complete['all']:
        assert math.isclose(contrast['summed_nats']/complete['all']['e32_minus_e12']['summed_nats'],
            group['share_of_e32_e12_summed_nats'], abs_tol=1e-12)
assert math.isclose(math.fsum(g['e32_minus_e12']['summed_nats'] for g in complete['regimes']),
    complete['all']['e32_minus_e12']['summed_nats'], abs_tol=1e-8)
giant = complete['independent_giant']
assert giant['targets'] == trajectory['native_scope_contract']['giant']['targets'] == 131071
for e in trajectory['exports']:
    assert math.isclose(e['native']['giant']['raw_nll'], giant['native_nll_by_epoch'][str(e['epoch'])], abs_tol=1e-12)
assert math.isclose(giant['native_nll_by_epoch']['32']-giant['native_nll_by_epoch']['12'],
    giant['e32_minus_e12_nats_per_target'], abs_tol=1e-12)
print(json.dumps({'status': 'passed', 'pieces': len(rows), 'targets': N,
    'shared_late_loss_and_depth_recovery_targets': sum(r[0] for r in overlap),
    'named_e2_over_e1_all_kappa1_net_bytes': advantage,
    'named_high_kappa_break_even_with_low_mid_kappa1': threshold,
    'depth32_minus_depth12_nats_per_target': complete['all']['e32_minus_e12']['nats_per_target'],
    'complete_depth_regime_best_positive_epochs': [g['best_positive_epoch'] for g in complete['regimes']]}, indent=2))
