#!/usr/bin/env python3
"""Replay the compact terminal curve and its exact q2 role/accounting boundaries."""
import json
import math
from pathlib import Path

r = json.loads((Path(__file__).with_name('results') / 'integer-coordinate-terminal-64.json').read_text())
curve, w, manifest = r['curve'], r['weight'], r['input_manifest_summary']
assert [e['epoch'] for e in curve] == list(range(1, 65))
assert sum(e['training_steps'] for e in curve) == curve[-1]['step'] == 1904
assert all(e['training_targets'] == manifest['training_targets'] == 1044469 for e in curve)
assert 64 * manifest['training_targets'] == r['metrics']['seen_targets'] == 66846016
assert r['metrics']['finite_nonzero_gradient_tensors'] == 111
assert w['all_64_fixed_scale_and_tiny_bits_preserved']
assert min(curve, key=lambda e: e['cold_target_weighted_delta_raw_nll'])['epoch'] == r['best_cold_epoch'] == 4
for e in curve:
    assert e['double_weight_growth_bytes'] == 2 * e['single_weight_growth_bytes'] == 2 * (e['weight_bytes'] - w['baseline_bytes'])
    n = sum(s['targets'] for s in e['cold_scores'].values())
    delta = sum(s['targets'] * s['delta_raw_nll'] for s in e['cold_scores'].values()) / n
    assert math.isclose(delta, e['cold_target_weighted_delta_raw_nll'], abs_tol=1e-14)
    assert math.isclose(2 * e['hard_entropy_delta_nats'] / (8 * math.log(2)),
                        e['ideal_double_weight_entropy_growth_bytes'], abs_tol=1e-8)
assert w['double_growth_bytes'] == curve[-1]['double_weight_growth_bytes'] == 20058
for scope in r['native_scopes'].values():
    assert scope['baseline_targets_and_all_bounds_equal']
    assert math.isclose(scope['candidate_raw_nll'] - scope['baseline_raw_nll'], scope['delta_raw_nll'], abs_tol=1e-14)
    assert math.isclose(-scope['delta_raw_nll'] * scope['targets'] / (8 * math.log(2)),
                        scope['ideal_raw_nn_saved_bytes_same_scope'], abs_tol=1e-8)
p = r['mature_prior']
assert math.isclose(p['baseline_benefit_nats'] - p['final_benefit_nats'], p['delta_interaction_nats'], abs_tol=1e-14)
roles, geometry = r['q2_manifest_roles'], r['q2_piece_geometry']
assert roles['train']['targets'] + roles['held']['targets'] + roles['remaining_full_targets'] == roles['full']['targets'] == 1048367
assert sum(b - a - 1 for a, b in geometry['unmatched_pieces']) == roles['remaining_full_targets'] == 10644
assert roles['held']['begin'] == 531039 and roles['held']['end'] == 1047533
assert [v - roles['held']['begin'] for v in geometry['full_bounds']
        if roles['held']['begin'] <= v <= roles['held']['end']] == geometry['held_bounds']
rest = r['q2_training_containing_complement']
for side in ('baseline', 'candidate'):
    full, held = r['q2_native_total_accumulators'][side], r['q2_held_accumulators'][side]
    assert full['count'] - held['count'] == rest[side]['targets'] == 531976
    assert math.isclose(full['logits_sum'] - held['logits_sum'], rest[side]['logits_sum'], abs_tol=1e-8)
    assert math.isclose(rest[side]['logits_sum'] / rest[side]['targets'], rest[side]['raw_nll'], abs_tol=1e-14)
assert math.isclose(rest['candidate']['raw_nll'] - rest['baseline']['raw_nll'], rest['delta_raw_nll'], abs_tol=1e-14)
assert r['decision']['measured_export_epoch'] == r['original_mixed_snapshot']['same_export_epoch'] == 64
assert r['decision']['actual_coded_result'] is None and r['decision']['fixed_recipe_new_engineering'] == 'STOP'
print(json.dumps(dict(status='passed', complete_epochs=64, same_export_epoch=64,
                     q2_train_targets=roles['train']['targets'], q2_held_targets=roles['held']['targets'],
                     crossing_eof_targets=10644, double_weight_growth_bytes=20058)))
