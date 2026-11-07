#!/usr/bin/env python3
"""Stdlib replay of the original eleven-epoch causal curve and sole selector.

Run alongside causal-first-eleven-epochs.json, or from research/ with the result
in research/results/. --result accepts an explicit result path.
"""
import argparse
import json
import math
from pathlib import Path
import statistics


def close(actual, expected, tolerance=1e-12):
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance), (actual, expected)


def pairing(pieces, old_column, new_column, summary):
    assert len(pieces) == summary['pieces'] == 473
    assert sum(p[0] for p in pieces) == summary['targets'] == 1082505
    for name, comparator in (('worse', lambda a, b: a > b),
                             ('better', lambda a, b: a < b),
                             ('equal', lambda a, b: a == b)):
        picked = [p for p in pieces if comparator(p[new_column], p[old_column])]
        assert len(picked) == summary[name + '_pieces']
        assert sum(p[0] for p in picked) == summary[name + '_targets']
    close(math.fsum(p[0] * (p[new_column] - p[old_column]) for p in pieces)
          / summary['targets'], summary['weighted_delta_nats'])
    close(statistics.median(p[new_column] - p[old_column] for p in pieces),
          summary['median_piece_delta_nats'])


def main():
    here = Path(__file__).resolve().parent
    name = 'causal-first-eleven-epochs.json'
    default = here / 'results' / name
    if not default.exists():
        default = here / name
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, default=default)
    r = json.loads(parser.parse_args().result.read_text())
    assert r['schema'] == 'hutter.causal-first-eleven-epochs.v1'
    b, epochs = r['baseline'], r['epochs']
    assert [e['epoch'] for e in epochs] == list(range(1, 12))
    assert (b['all_targets'], b['long_targets'], b['short_targets'], b['audit_targets']) == (1082505, 107647, 974858, 131071)
    assert (r['full_native_targets'], r['original_selector_target_denominator']) == (586215331, 586459020)
    close((b['short_targets'] * b['short_nll'] + b['long_targets'] * b['long_nll'])
          / b['all_targets'], b['all_nll'])
    for e in epochs:
        assert (e['all_targets'], e['short_targets'], e['long_targets'], e['audit_targets']) == (1082505, 974858, 107647, 131071)
        for metric, delta in (('all_nll', 'delta_w0'), ('short_nll', 'short_delta_w0'),
                              ('long_nll', 'long_delta_w0'), ('audit_nll', 'audit_delta_w0')):
            close(e[metric] - b[metric], e[delta])
        close((e['short_targets'] * e['short_nll'] + e['long_targets'] * e['long_nll'])
              / e['all_targets'], e['all_nll'])
        dw = 2 * (b['weight_bytes'] - e['weight_bytes'])
        raw = -e['delta_w0'] * r['full_native_targets'] / (8 * math.log(2))
        raw_selector = -e['delta_w0'] * r['original_selector_target_denominator'] / (8 * math.log(2))
        assert dw == e['double_weight_saved_bytes']
        close(raw, e['nn_scope_raw_proxy_bytes'], 1e-7)
        close(raw + dw, e['nn_scope_net_proxy_bytes'], 1e-7)
        close(raw_selector + dw, e['selector_net_proxy_bytes'], 1e-7)
        assert e['max_scale_delta'] == 0
        close(e['changed_weight_symbols'] / 5868864, e['changed_symbol_fraction'])
    chosen = max(epochs, key=lambda e: (e['selector_net_proxy_bytes'], -e['epoch']))
    native_choice = max(epochs, key=lambda e: (e['nn_scope_net_proxy_bytes'], -e['epoch']))
    assert chosen['epoch'] == native_choice['epoch'] == r['selection']['epoch'] == 2
    assert r['selection']['kappa'] == 1
    close(chosen['nn_scope_net_proxy_bytes'], r['selection']['same_scope_full_native_net_proxy_bytes'], 1e-7)
    close(chosen['selector_net_proxy_bytes'], r['selection']['original_selector_net_proxy_bytes'], 1e-7)
    for metric in ('all_nll', 'short_nll', 'long_nll', 'audit_nll'):
        assert all(a[metric] < c[metric] for a, c in zip(epochs[1:], epochs[2:]))
    assert all(a['changed_weight_symbols'] < c['changed_weight_symbols']
               for a, c in zip(epochs, epochs[1:]))

    train = r['online_training_epochs']
    assert [e['epoch'] for e in train] == list(range(1, 12))
    assert all((e['targets'], e['pieces']) == (2941307, 1313) for e in train)
    assert all(a['online_ce_mean'] > c['online_ce_mean']
               and a['online_teacher_kl_mean'] < c['online_teacher_kl_mean']
               for a, c in zip(train, train[1:]))
    snapshot = r['latest_training_snapshot']
    assert (snapshot['completed_epochs'], snapshot['goal_epochs'], snapshot['partial_epoch']) == (11, 16, 12)
    assert snapshot['seen_training_targets'] == 11 * 2941307 + snapshot['partial_epoch_targets']
    assert sum(e['optimizer_steps'] for e in train) + snapshot['partial_epoch_optimizer_steps'] == snapshot['step']
    assert snapshot['finite_gradient_tensors_per_step'] == snapshot['nonzero_gradient_tensors_per_step'] == 111

    pieces = r['paired_pieces']
    assert r['paired_piece_columns'] == ['targets', 'w0_native_nll', 'e10_native_nll', 'e11_native_nll']
    assert all(p[0] > 0 and len(p) == 4 and all(math.isfinite(x) for x in p) for p in pieces)
    long_indices = r['held_long_piece_indices']
    assert long_indices == [127, 157, 181]
    long_pieces = [pieces[i] for i in long_indices]
    short_pieces = [p for i, p in enumerate(pieces) if i not in long_indices]
    assert sum(p[0] for p in long_pieces) == 107647
    assert sum(p[0] for p in short_pieces) == 974858
    for column, reference in ((1, b), (2, epochs[-2]), (3, epochs[-1])):
        for subset, targets, metric in ((pieces, 1082505, 'all_nll'),
                                        (long_pieces, 107647, 'long_nll'),
                                        (short_pieces, 974858, 'short_nll')):
            close(math.fsum(p[0] * p[column] for p in subset) / targets,
                  reference[metric])
    pairing(pieces, 1, 3, r['e11_held_piece_pairing'])
    pairing(pieces, 2, 3, r['e10_to_e11_held_piece_pairing'])
    close(r['e11_held_piece_pairing']['weighted_delta_nats'], epochs[-1]['delta_w0'])
    close(r['e10_to_e11_held_piece_pairing']['weighted_delta_nats'],
          epochs[-1]['all_nll'] - epochs[-2]['all_nll'])
    change = r['e10_to_e11']
    assert (change['from_epoch'], change['to_epoch']) == (10, 11)
    for key in change:
        if key.endswith('_delta'):
            field = key[:-6]
            values = train if field.startswith('online_') else epochs
            close(change[key], values[-1][field] - values[-2][field], 1e-7)
    assert epochs[-1]['weight_bytes'] == 2907839
    assert epochs[-1]['double_weight_saved_bytes'] == -7028

    cpu = r['actual_cpu']
    close(cpu['unique_trainer_cpu_seconds'] + cpu['baseline_credit_seconds'],
          cpu['latest_reported_total_cpu_seconds'], 1e-9)
    close(cpu['method_cap_remaining_seconds'] + cpu['latest_reported_total_cpu_seconds'],
          cpu['method_cpu_cap_seconds'], 1e-9)
    assert cpu['method_cpu_cap_seconds'] == 48 * 3600
    assert cpu['remaining_training_targets'] == 16 * 2941307 - snapshot['seen_training_targets']
    close(cpu['epochs_ten_to_eleven_mean_evaluated_cpu_seconds'],
          (epochs[-1]['reported_evaluated_cpu_seconds'] - epochs[-3]['reported_evaluated_cpu_seconds']) / 2, 1e-9)
    close(cpu['e11_evaluation_cycle_cpu_seconds'],
          epochs[-1]['reported_evaluated_cpu_seconds'] - epochs[-2]['reported_evaluated_cpu_seconds'], 1e-9)
    close(cpu['remaining_cpu_at_observed_e9_e11_rate_seconds'],
          cpu['remaining_training_targets'] / 2941307 * cpu['epochs_ten_to_eleven_mean_evaluated_cpu_seconds'], 1e-9)
    assert cpu['remaining_cpu_at_observed_e9_e11_rate_seconds'] < cpu['method_cap_remaining_seconds']
    collection = r['provenance']['collection']
    assert collection['status'] == 'passed' and collection['ordinal'] == 2
    assert len(collection['requests']) == 3
    assert all(q['method'] == 'GET' and q['http_status'] == 200 and q['status'] == 'passed'
               for q in collection['requests'])
    assert collection['retries'] == collection['post_requests'] == collection['model_calls'] == 0
    for file, request in zip(r['provenance']['files'], collection['requests']):
        assert file['bytes'] == request['response_bytes']
        assert len(file['sha256']) == 64
    print(json.dumps(dict(
        status='passed', complete_epochs=11, selected_epoch=2,
        consecutive_four_domain_regressions=9,
        e11_held_delta_nats=epochs[-1]['delta_w0'],
        e10_to_e11_held_delta_nats=change['all_nll_delta'],
        e11_worse_held_pieces=r['e11_held_piece_pairing']['worse_pieces'],
        e11_double_weight_saved_bytes=epochs[-1]['double_weight_saved_bytes']), indent=2))


if __name__ == '__main__':
    main()
