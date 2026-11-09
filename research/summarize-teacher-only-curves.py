#!/usr/bin/env python3
"""Rebuild complete, paired training/native curves from retained raw receipts.

One fixed 32-epoch experiment pair; no fitting, model loading, or codec calls.
"""
import argparse
import json
import math
from pathlib import Path

P = argparse.ArgumentParser()
P.add_argument('old', type=Path)
P.add_argument('fresh', type=Path)
args = P.parse_args()
N = 2941307
FIELDS = ('ce_mean', 'teacher_kl_mean', 'final_hidden_mse_mean')


def read(root, verify_native_logs):
    state = json.loads((root / 'training-state.json').read_text())
    records = [json.loads(s) for s in (root / 'metrics.jsonl').read_text().splitlines()]
    epochs = [r for r in records if r['kind'] == 'epoch_trained']
    steps = [r for r in records if r['kind'] == 'optimizer_step']
    assert state['status'] == 'complete' and state['epoch'] == 32
    assert state['seen_targets'] == N * 32 and len(steps) == state['step'] == 2524
    assert [r['trained_epoch'] for r in epochs] == list(range(1, 33))
    curve = []
    for epoch in epochs:
        e = epoch['trained_epoch']
        block = [r for r in steps if r['epoch'] == e - 1]
        assert sum(r['targets'] for r in block) == N
        for key in FIELDS:
            mean = math.fsum(r[key] * r['targets'] for r in block) / N
            assert math.isclose(mean, epoch[key], abs_tol=1e-12, rel_tol=0)
        curve.append({'epoch': e, 'stage': 'continuous' if e <= 8 else 'hard',
                      'targets': N, 'optimizer_steps': len(block),
                      **{key: epoch[key] for key in FIELDS}})
    native = []
    for evaluation in state['evaluations']:
        e, s = evaluation['epoch'], evaluation['scores']
        for suffix, expected in [('causal', 1082505), ('long-audit', 131071)]:
            assert s[suffix]['targets'] == expected
            # Old native receipts reside in its complete state; fresh logs are all collected.
            if verify_native_logs:
                lines = (root / f'epoch-{e:02d}-{suffix}.log').read_text().splitlines()
                rows = [json.loads(l[len('ACTIVATION_RESULT '):]) for l in lines
                        if l.startswith('ACTIVATION_RESULT ')]
                assert len(rows) == 1
                row = rows[0]
                assert row['total']['count'] == expected
                assert row['total']['logits_mean'] == s[suffix]['raw_nll']
                assert sum(p['loss']['count'] for p in row['piece_losses']) == expected
        native.append({'epoch': e, 'weight_bytes': evaluation['weight_bytes'],
                       'fullheld_nll': s['causal']['raw_nll'],
                       'fullheld_delta_W0': s['causal']['delta_raw_nll'],
                       'held_long_nll': s['causal']['long_native']['raw_nll'],
                       'giant_nll': s['long-audit']['raw_nll']})
    assert [r['epoch'] for r in native] == list(range(4, 33, 4))
    return state, curve, native


old, old_curve, old_native = read(args.old, False)
fresh, fresh_curve, fresh_native = read(args.fresh, True)
ignore = {'method', 'label_ce', 'teacher_kl'}
old_recipe, fresh_recipe = old['identity']['recipe'], fresh['identity']['recipe']
assert {k: v for k, v in old_recipe.items() if k not in ignore} == {
    k: v for k, v in fresh_recipe.items() if k not in ignore}
assert old_recipe['teacher_kl'] == 1 and fresh_recipe['teacher_kl'] == 2
assert fresh_recipe['label_ce'] == 0
assert old['zero_step']['weight_bytes'] == fresh['zero_step']['weight_bytes']
paired = []
for a, b in zip(old_native, fresh_native):
    assert a['epoch'] == b['epoch']
    paired.append({'epoch': a['epoch'], 'original_CE_KL': a, 'teacher_only': b,
                   'fresh_minus_old_nll': {key: b[key] - a[key] for key in
                                         ('fullheld_nll', 'held_long_nll', 'giant_nll')},
                   'fresh_double_weight_charge_improvement_bytes':
                   2 * (a['weight_bytes'] - b['weight_bytes'])})
late = {}
for name, curve, native in [('original_CE_KL', old_curve, old_native),
                             ('teacher_only', fresh_curve, fresh_native)]:
    a, b = curve[11], curve[31]
    x, y = native[2], native[7]
    late[name] = {'epochs': [12, 32],
                  'online_training_changes': {key: b[key] - a[key] for key in FIELDS},
                  'hard_exported_native_changes': {key: y[key] - x[key] for key in
                                                  ('fullheld_nll', 'held_long_nll', 'giant_nll')},
                  'double_weight_charge_growth_bytes': 2 * (y['weight_bytes'] - x['weight_bytes'])}
result = {'schema': 'hutter.teacher-only.complete-curves.v1', 'validation_status': 'passed',
          'training': {'epochs_each': 32, 'targets_each_epoch': N, 'targets_each_trajectory': N * 32,
                       'optimizer_steps_each': 2524, 'continuous_then_hard_epochs': [8, 24],
                       'online_means': 'pre-update, target-weighted, same 1313 training pieces',
                       'original_CE_KL': old_curve, 'teacher_only': fresh_curve},
          'native_scope': {'fullheld_primary_targets': 1082505, 'fullheld_pieces': 473,
                           'contained_held_long_targets': 107647,
                           'independent_report_only_giant_targets': 131071,
                           'independent_targets_each_evaluation': 1213576,
                           'measured_coordinate': 'native FP32 logits NLL before half/guard/mixer/u16/range'},
          'matched_native_checkpoints': paired, 'e12_to_e32': late,
          'source_checks': {'old_native_receipts': 'eight entries in original complete training-state',
                            'fresh_native_log_files_verified': 16,
                            'target_weighted_optimizer_steps_verified': 5048},
          'readout_operations': {'new_training_runs': 0, 'new_native_runs': 0, 'new_codec_runs': 0,
                                 'additional_scientific_downloads': 0}}
print(json.dumps(result, indent=2))
