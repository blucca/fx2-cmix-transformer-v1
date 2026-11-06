"""Replay the original e16 hard-native pairing and byte/CPU contracts with stdlib."""
import json, math
from pathlib import Path

r = json.loads((Path(__file__).with_name('results') / 'depth-hard-e16.json').read_text())
n = r['native']; ps = r['paired_pieces']; N = sum(p[0] for p in ps)
assert len(ps) == 473 and N == 1082505
assert r['paired_piece_columns'] == ['targets', 'w0_logits_sum', 'e8_logits_sum', 'e12_logits_sum', 'e16_logits_sum']
assert all(len(p) == 5 and isinstance(p[0], int) and p[0] > 0 for p in ps)
for c, k in [(1, 'original_nll'), (2, 'e8_nll'), (3, 'e12_nll'), (4, 'e16_nll')]:
    assert math.isclose(sum(p[c] for p in ps) / N, n['held'][k], abs_tol=1e-12)
for x in n.values():
    for k, previous in [('delta_w0', 'original_nll'), ('delta_e8', 'e8_nll'), ('delta_e12', 'e12_nll')]:
        assert math.isclose(x['e16_nll'] - x[previous], x[k], abs_tol=1e-14)
for k in ['original_nll', 'e8_nll', 'e12_nll', 'e16_nll']:
    assert math.isclose(sum(n[s]['targets'] * n[s][k] for s in ['held_short', 'held_long']) / N, n['held'][k], abs_tol=1e-14)
    # The original three held-long indices are unchanged through this checkpoint.
    c = {'original_nll': 1, 'e8_nll': 2, 'e12_nll': 3, 'e16_nll': 4}[k]
    assert sum(ps[i][0] for i in [127, 157, 181]) == n['held_long']['targets']
    assert math.isclose(sum(ps[i][c] for i in [127, 157, 181]) / n['held_long']['targets'], n['held_long'][k], abs_tol=1e-12)
assert [n[s]['targets'] for s in ['held_short', 'held_long', 'maximum_length_audit']] == [974858, 107647, 131071]
assert n['held']['e16_nll'] == 1.1492552658631419
assert n['held_long']['e16_nll'] == 1.4727867300938235
assert n['maximum_length_audit']['e16_nll'] == 0.2758085249375663
for k, c in [('w0', 1), ('e12', 3)]:
    s = r['held_piece_pairing'][k]
    assert sum(s[f'{x}_pieces'] for x in ['better', 'worse', 'equal']) == 473
    assert s['better_targets'] + s['worse_targets'] == N
    for name, compare in [('better', lambda a, b: a < b), ('worse', lambda a, b: a > b), ('equal', lambda a, b: a == b)]:
        assert s[f'{name}_pieces'] == sum(compare(p[4], p[c]) for p in ps)
        if name != 'equal':
            assert s[f'{name}_targets'] == sum(p[0] for p in ps if compare(p[4], p[c]))
    assert math.isclose(sum(p[4] - p[c] for p in ps) / N, s['weighted_delta_nats'], abs_tol=1e-12)
assert (r['held_piece_pairing']['e12']['worse_pieces'], r['held_piece_pairing']['e12']['better_pieces']) == (267, 206)
assert (r['held_piece_pairing']['e12']['worse_targets'], r['held_piece_pairing']['e12']['better_targets']) == (800662, 281843)

ph = r['phase']; es = r['online_training_epochs']
assert [e['epoch'] for e in es] == list(range(1, 17))
assert all((e['targets'], e['pieces']) == (2941307, 1313) for e in es)
assert (ph['continuous_epochs_completed'], ph['hard_epochs_completed'], ph['hard_epochs_goal'], ph['goal_epochs']) == (8, 8, 24, 32)
assert all(e['stage'] == ('continuous' if e['epoch'] <= 8 else 'hard') for e in es)
assert sum(e['optimizer_steps'] for e in es) == ph['complete_optimizer_steps'] == 1261
assert sum(e['targets'] for e in es) == ph['completed_full_epoch_targets'] == 47060912
assert ph['partial_epoch'] == 17 and ph['partial_optimizer_steps'] == 53
assert ph['recorded_optimizer_steps'] == 1261 + ph['partial_optimizer_steps'] == 1314
assert ph['recorded_seen_targets'] == 16 * 2941307 + ph['partial_targets'] == 49019732
assert math.isclose(es[-1]['ce_mean'], .9918124112938157, abs_tol=1e-14)
assert math.isclose(es[-1]['teacher_kl_mean'], .1443646940310497, abs_tol=1e-14)
assert all(a['ce_mean'] > b['ce_mean'] and a['teacher_kl_mean'] < b['teacher_kl_mean']
           for a, b in zip(es[12:15], es[13:16]))

b = r['actual_package']
assert (b['original_weight_bytes'], b['e12_weight_bytes'], b['e16_weight_bytes']) == (2904325, 1960982, 1961715)
assert (b['original_binary_bytes'], b['student_binary_bytes']) == (181956, 177444)
assert 2 * (b['original_weight_bytes'] - b['e16_weight_bytes']) == b['double_weight_saved_bytes'] == 1885220
assert 2 * (b['original_binary_bytes'] - b['student_binary_bytes']) == b['double_binary_saved_bytes'] == 9024
space = b['double_weight_saved_bytes'] + b['double_binary_saved_bytes']
assert space == b['double_weight_plus_binary_saved_bytes'] == 1894244
s = r['named_size_sensitivity']
assert s['full_native_targets'] == 586215331 and s['historical_record_gap_bytes'] == 952225
penalty = s['full_native_targets'] * n['held']['delta_w0'] / (8 * math.log(2))
assert math.isclose(penalty, s['kappa1_payload_increase_bytes'], abs_tol=1e-6)
assert math.isclose(penalty - space, s['kappa1_net_size_increase_bytes'], abs_tol=1e-6)
assert s['historical_payload_allowance_bytes'] == space - s['historical_record_gap_bytes'] == 942019
assert math.isclose(s['break_even_kappa'] * penalty, space, abs_tol=1e-6)
assert math.isclose(s['historical_record_kappa'] * penalty - space, -s['historical_record_gap_bytes'], abs_tol=1e-6)
g = r['e16_coordinate_geometry']
assert g['max_scale_delta'] == 0 and g['int4_values'] == 3951936
assert g['changed_z_coordinates'] - g['changed_weight_symbols'] == g['displaced_z_still_rounding_to_initial_symbol'] == 2278005
assert math.isclose(g['changed_symbol_fraction'], g['changed_weight_symbols'] / g['int4_values'], abs_tol=1e-14)
reg = r['held_surprisal_regimes']
assert sum(x['targets'] for x in reg) == N and sum(x['pieces'] for x in reg) == 473
for x, (name, lo, hi) in zip(reg, [('low', 0, .25), ('middle', .25, 1), ('high', 1, math.inf)]):
    part = [p for p in ps if lo <= p[1] / p[0] < hi]
    assert x['name'] == name and x['pieces'] == len(part) and x['targets'] == sum(p[0] for p in part)
    for key, c1, c0 in [('original_surprisal_sum', 1, None), ('e8_delta_sum', 2, 1),
                        ('e12_delta_sum', 3, 1), ('e16_delta_sum', 4, 1), ('e16_minus_e12_sum', 4, 3)]:
        assert math.isclose(sum(p[c1] - (p[c0] if c0 else 0) for p in part), x[key], abs_tol=1e-8)
assert math.isclose(sum(x['e16_minus_e12_sum'] for x in reg) / N, n['held']['delta_e12'], abs_tol=1e-12)

cpu = r['actual_experiment_cpu']
assert cpu['method_cpu_cap_seconds'] == 172800
assert math.isclose(cpu['frozen_preparation_credit_seconds'], 121.716696643, abs_tol=1e-9)
assert cpu['e16_trained_reported_total_seconds'] < cpu['e16_evaluated_reported_total_seconds'] <= cpu['latest_reported_total_cpu_seconds'] < cpu['method_cpu_cap_seconds']
assert math.isclose(cpu['e16_evaluated_reported_total_seconds'] - cpu['e16_trained_reported_total_seconds'], cpu['e16_export_and_native_seconds'], abs_tol=1e-9)
assert math.isclose(cpu['latest_unique_training_and_evaluation_cpu_seconds'] + cpu['frozen_preparation_credit_seconds'], cpu['latest_reported_total_cpu_seconds'], abs_tol=1e-9)
assert math.isclose(cpu['latest_reported_total_cpu_seconds'] + cpu['method_cap_remaining_seconds'], 172800, abs_tol=1e-9)
for scope in ['causal', 'audit']:
    assert 0 < cpu[scope+'_eval_cpu_seconds'] <= cpu[scope+'_program_cpu_seconds']
    assert cpu[scope+'_max_rss_kib'] > 0
assert cpu['causal_program_cpu_seconds'] + cpu['audit_program_cpu_seconds'] <= cpu['e16_export_and_native_seconds']
export = r['static15_export_contract']
assert export['status'] == 'passed' and export['all_tensor_bytes_equal'] and export['expected_tensor_bytes_equal']
assert (export['tensor_count'], export['int4_tensors'], export['int4_values']) == (294, 75, 3951936)
assert r['decision']['next_native_epoch'] == 20 and r['decision']['current_native_minimum_epoch'] == 12
print(json.dumps(dict(status='passed', hard_epochs=8, paired_pieces=473,
                      held_delta_since_e12=n['held']['delta_e12'],
                      worse_pieces_since_e12=r['held_piece_pairing']['e12']['worse_pieces'],
                      worse_targets_since_e12=r['held_piece_pairing']['e12']['worse_targets'],
                      double_W_B_saved_bytes=space, historical_scenario_kappa=s['historical_record_kappa'],
                      method_cpu_remaining_seconds=cpu['method_cap_remaining_seconds']), indent=2))
