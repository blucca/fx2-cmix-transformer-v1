#!/usr/bin/env python3
"""Replay the unchanged eight-layer continuous-to-hard stage and named size gate."""
import json
import math
from pathlib import Path


def main():
    r = json.loads((Path(__file__).with_name('results') / 'depth-continuous-terminal-e8.json').read_text())
    h, l, s = (r['native'][name] for name in ('held', 'held_long', 'held_short'))
    assert (h['targets'], l['targets'], s['targets']) == (1082505, 107647, 974858)
    for row in r['native'].values():
        assert math.isclose(row['student_nll'] - row['original_nll'], row['delta_nll'], abs_tol=1e-14)
        assert math.isclose(row['student_nll'] - row['e4_nll'], row['delta_from_e4'], abs_tol=1e-14)
    for field in ('student_nll', 'original_nll', 'delta_nll'):
        weighted = (l['targets'] * l[field] + s['targets'] * s[field]) / h['targets']
        assert math.isclose(weighted, h[field], abs_tol=1e-14)
    phase, epochs = r['phase'], r['continuous_epoch_training']
    assert (phase['continuous_epochs_completed'], phase['hard_qat_epochs_completed'], phase['goal_epochs']) == (8, 0, 32)
    assert [row['epoch'] for row in epochs] == list(range(1, 9))
    assert all((row['targets'], row['pieces']) == (2941307, 1313) for row in epochs)
    assert phase['recorded_seen_targets'] == 8 * 2941307 + phase['partial_hard_targets']
    assert sum(row['optimizer_steps'] for row in epochs) + phase['partial_hard_optimizer_steps'] == phase['recorded_step'] == 656
    assert phase['first_hard_optimizer_step'] == sum(row['optimizer_steps'] for row in epochs) + 1 == 633
    assert all(a['ce_mean'] > b['ce_mean'] and a['teacher_kl_mean'] > b['teacher_kl_mean'] for a, b in zip(epochs, epochs[1:]))
    package, scenario = r['actual_package'], r['named_size_sensitivity']
    dw = 2 * (package['original_weight_bytes'] - package['e8_weight_bytes'])
    db = 2 * (package['original_binary_bytes'] - package['student_binary_bytes'])
    space = dw + db
    assert (dw, db, space) == (1888330, 9024, 1897354)
    assert space == package['double_weight_plus_binary_saved_bytes']
    penalty = scenario['full_native_targets'] * h['delta_nll'] / (8 * math.log(2))
    assert math.isclose(penalty, scenario['kappa1_payload_increase_bytes'], abs_tol=1e-6)
    assert math.isclose(scenario['break_even_kappa'] * penalty - space, 0, abs_tol=1e-6)
    assert math.isclose(scenario['historical_record_kappa'] * penalty - space, -scenario['historical_record_gap_bytes'], abs_tol=1e-6)
    g = r['e8_coordinate_geometry']
    assert g['changed_z_coordinates'] - g['changed_weight_symbols'] == g['displaced_z_still_rounding_to_initial_symbol'] == 2604816
    for compare, field in (('original', 'delta_nll'), ('e4', 'delta_from_e4')):
        p = r['held_piece_pairing'][compare]
        assert p['worse_pieces'] + p['better_pieces'] + p['equal_pieces'] == 473
        assert p['worse_targets'] + p['better_targets'] == 1082505
        assert math.isclose(p['weighted_delta_nats'], h[field], abs_tol=1e-12)
    cpu = r['actual_experiment_cpu']
    assert math.isclose(cpu['latest_reported_total_cpu_seconds'] + cpu['method_cap_remaining_seconds'], cpu['method_cpu_cap_seconds'], abs_tol=1e-9)
    assert math.isclose(cpu['causal_program_cpu_seconds'] + cpu['audit_program_cpu_seconds'], cpu['both_native_program_cpu_seconds'], abs_tol=1e-12)
    print(json.dumps(dict(status='passed', continuous_epochs=8, first_hard_step=633,
                          partial_hard_targets=phase['partial_hard_targets'],
                          held_delta_nats=h['delta_nll'], double_weight_plus_binary_saved_bytes=space,
                          historical_record_kappa=scenario['historical_record_kappa']), indent=2))


if __name__ == '__main__':
    main()
