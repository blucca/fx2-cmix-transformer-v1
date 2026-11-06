#!/usr/bin/env python3
"""Replay the original seven-epoch causal curve and its terminal selector."""
import json
import math
from pathlib import Path


def main():
    r = json.loads((Path(__file__).with_name('results') / 'causal-first-seven-epochs.json').read_text())
    b, epochs = r['baseline'], r['epochs']
    assert [e['epoch'] for e in epochs] == list(range(1, 8))
    assert (b['all_targets'], b['long_targets'], b['short_targets']) == (1082505, 107647, 974858)
    for e in epochs:
        assert math.isclose(e['all_nll'] - b['all_nll'], e['delta_w0'], abs_tol=1e-14)
        weighted = (e['short_targets'] * e['short_nll'] + e['long_targets'] * e['long_nll']) / e['all_targets']
        assert math.isclose(weighted, e['all_nll'], abs_tol=1e-14)
        dw = 2 * (b['weight_bytes'] - e['weight_bytes'])
        raw = -e['delta_w0'] * r['full_native_targets'] / (8 * math.log(2))
        assert dw == e['double_weight_saved_bytes']
        assert math.isclose(raw + dw, e['nn_scope_net_proxy_bytes'], abs_tol=1e-7)
        assert e['max_scale_delta'] == 0
    selected = max(epochs, key=lambda e: (e['selector_net_proxy_bytes'], -e['epoch']))
    assert selected['epoch'] == r['selection']['epoch'] == 2
    for metric in ('all_nll', 'short_nll', 'long_nll', 'audit_nll'):
        assert all(a[metric] < c[metric] for a, c in zip(epochs[1:], epochs[2:]))
    train = r['online_training_epochs']
    assert all((e['targets'], e['pieces']) == (2941307, 1313) for e in train)
    assert all(a['online_ce_mean'] > c['online_ce_mean'] and a['online_teacher_kl_mean'] < c['online_teacher_kl_mean'] for a, c in zip(train, train[1:]))
    snapshot = r['latest_training_snapshot']
    assert snapshot['seen_training_targets'] == 7 * 2941307 + snapshot['partial_epoch_targets']
    assert sum(e['optimizer_steps'] for e in train) + snapshot['partial_epoch_optimizer_steps'] == snapshot['step'] == 566
    pair = r['e6_held_piece_pairing']
    assert pair['worse_pieces'] + pair['better_pieces'] == pair['pieces'] == 473
    assert pair['worse_targets'] + pair['better_targets'] == pair['targets'] == 1082505
    assert math.isclose(pair['weighted_delta_nats'], epochs[5]['delta_w0'], abs_tol=1e-12)
    cpu = r['actual_cpu']
    assert math.isclose(cpu['unique_trainer_cpu_seconds'] + cpu['baseline_credit_seconds'], cpu['latest_reported_total_cpu_seconds'], abs_tol=1e-9)
    assert math.isclose(cpu['method_cap_remaining_seconds'] + cpu['latest_reported_total_cpu_seconds'], cpu['method_cpu_cap_seconds'], abs_tol=1e-9)
    print(json.dumps(dict(status='passed', complete_epochs=7, selected_epoch=2,
                          e7_held_delta_nats=epochs[-1]['delta_w0'],
                          e6_worse_held_pieces=pair['worse_pieces']), indent=2))


if __name__ == '__main__':
    main()
