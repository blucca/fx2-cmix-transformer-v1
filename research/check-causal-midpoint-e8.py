#!/usr/bin/env python3
"""Replay only the newly delivered native midpoint; original training stays fixed."""
import json
import math
from pathlib import Path

root = Path(__file__).with_name('results')
r = json.loads((root / 'causal-native-midpoint-e8.json').read_text())
prior = json.loads((root / r['previous_curve']).read_text())
e, b = r['epoch8'], prior['baseline']
assert e['epoch'] == r['plan']['complete_native_checkpoints'] == 8
assert e['double_weight_saved_bytes'] == 2 * (b['weight_bytes'] - e['weight_bytes'])
assert math.isclose(e['all_nll'] - b['all_nll'], e['delta_w0'], abs_tol=1e-14)
weighted = (e['short_nll'] * e['short_targets'] + e['long_nll'] * e['long_targets']) / e['all_targets']
assert math.isclose(weighted, e['all_nll'], abs_tol=1e-14)
for metric in ('all_nll', 'short_nll', 'long_nll', 'audit_nll'):
    assert math.isclose(e[metric] - prior['epochs'][-1][metric], e['change_from_e7'][metric], abs_tol=1e-14)
    assert e['change_from_e7'][metric] > 0
raw = -e['delta_w0'] * r['full_native_targets'] / (8 * math.log(2))
assert math.isclose(raw + e['double_weight_saved_bytes'], e['nn_scope_net_proxy_bytes'], abs_tol=1e-7)
selected = max(prior['epochs'] + [e], key=lambda x: (x['selector_net_proxy_bytes'], -x['epoch']))
assert selected['epoch'] == r['selection']['epoch'] == 2
print(json.dumps({'status': 'passed', 'complete_native_epochs': 8, 'selected_epoch': 2}))
