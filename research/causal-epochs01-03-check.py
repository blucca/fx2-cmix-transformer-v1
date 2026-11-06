#!/usr/bin/env python3
"""Replay the observed fixed W0/e1/e2/e3 pairs; model/evaluator calls are zero."""
import json
import math
from pathlib import Path

r = Path(__file__).resolve().parent / 'results'
d = json.loads((r / 'causal-third-native-epoch.json').read_text())
pairs = json.loads((r / 'causal-epochs01-03-paired-pieces.json').read_text())['pairs']
assert len(pairs) == len({p['article'] for p in pairs}) == 473
N = sum(p['targets'] for p in pairs)
assert N == d['sampling']['held_targets'] == 1082505
for p in pairs:
    assert p['targets'] + 1 == p['piece_bytes'] == p['source_end'] - p['source_begin']
for e in d['epochs']:
    epoch = e['epoch']
    mean = sum(p['targets'] * p['nll_by_epoch'][str(epoch)] for p in pairs) / N
    assert math.isclose(mean, e['all_nll'], abs_tol=1e-12)
    fee = 2 * (d['epochs'][0]['weight_bytes'] - e['weight_bytes'])
    assert fee == e['double_weight_saved_bytes']
    for scope, field in (('nn_targets', 'nn_scope_net_proxy_bytes'),
                         ('selector_targets', 'selector_net_proxy_bytes')):
        net = (d['epochs'][0]['all_nll'] - mean) * d['sampling'][scope] / (8 * math.log(2)) + fee
        assert math.isclose(net, e[field], abs_tol=1e-6)
for before, after in ((0, 2), (0, 3), (2, 3)):
    expected = d['paired_breadth'][f'e{before}_to_e{after}']
    delta = lambda p: p['nll_by_epoch'][str(after)] - p['nll_by_epoch'][str(before)]
    assert math.isclose(sum(p['targets'] * delta(p) for p in pairs) / N,
                        expected['weighted_delta'], abs_tol=1e-12)
    for name, sign in (('improved', -1), ('worse', 1), ('equal', 0)):
        chosen = [p for p in pairs if (delta(p) < 0 if sign < 0 else delta(p) > 0 if sign > 0 else delta(p) == 0)]
        assert len(chosen) == expected[name]['pieces']
        assert sum(p['targets'] for p in chosen) == expected[name]['targets']
    for b in expected['length_bins']:
        group = [p for p in pairs if b['lower_exclusive'] < p['piece_bytes'] <= b['upper_inclusive']]
        targets = sum(p['targets'] for p in group)
        assert len(group) == b['pieces'] and targets == b['targets']
        if targets:
            assert math.isclose(sum(p['targets'] * delta(p) for p in group) / targets,
                                b['weighted_delta'], abs_tol=1e-12)
assert all(max(d['epochs'][1:], key=lambda e: e[field])['epoch'] == 2
           for field in ('nn_scope_net_proxy_bytes', 'selector_net_proxy_bytes'))
assert d['candidate_selection']['best_completed_epoch'] == 2
print('PASS: observed epoch-0–3 native pairs, weight fees, breadth and current e2 selection')
