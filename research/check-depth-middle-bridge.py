#!/usr/bin/env python3
"""Stdlib JSON replay of the retained finite middle-bridge diagnostic.

Checks saved identities, counts, role/oracle contracts and resource sums.
The existing finite-array checker supplies the raw-array validation receipt.
"""
import argparse
import json
import math
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--result', type=Path,
    default=Path(__file__).resolve().parent / 'results/depth-middle-bridge.json')
a = p.parse_args()
r = json.loads(a.result.read_text())
assert r['schema'] == 'hutter.depth8.middle-bridge.round450.v1'
assert (r['status'], r['stage']) == ('passed', 'complete')
assert r['top_level_teacher_forwards'] == 2
assert r['top_level_student_forwards'] == 4
assert set(r['losses']['train']) == {'teacher', 'zero'}
assert set(r['losses']['held']) == {'teacher', 'zero', 'fitted', 'oracle'}
for key in ('optimizer_updates', 'optimizer_instances', 'gradient_calls',
    'trained_checkpoint_reads', 'new_cache_files', 'ppmd_capture_calls',
    'native_evaluator_calls', 'codec_calls', 'model_exports', 'scale_searches',
    'scale_calibrations', 'native_int4_bridge_runs', 'full_training_runs',
    'real_coded_stream_runs', 'original_source_writes',
    'original_checkpoint_writes', 'original_cache_writes', 'original_native_writes'):
    assert r[key] == 0, key
assert r['teacher_checkpoint_reads'] == r['student_initialization_checkpoint_reads'] == r['ridge_solves'] == 1
assert r['checkpoint']['reads'] == {'teacher': 1, 'student_initialization': 1, 'trained_latest': 0}
assert r['counts'] == dict(teacher_state_tensors=429, student_state_tensors=289,
    student_hard_coordinates=75, student_raw_tensors=59)
assert r['hard_mode'] and r['fixed_scales'] and r['native_scaled_gemm']
assert r['torch_threads'] == r['torch_interop_threads'] == 1
assert r['mapping']['kept_original_layers'] == [0, 1, 2, 3, 8, 9, 10, 11]
assert r['mapping']['skip_source_indices'] == [2, 3, 4, 5]
assert r['mapping']['teacher_skip_pairs'] == [[3, 8], [2, 9], [1, 10], [0, 11]]
assert r['mapping']['student_skip_pairs'] == [[3, 4], [2, 5], [1, 6], [0, 7]]
for role, expected in (
    ('train', ('train', 1447, [1256446, 1257646], [5321, 6521], 1199)),
    ('held', ('validation', 3363, [5055675, 5057282], [14457, 16064], 1606))):
    q = r['pieces'][role]
    assert (q['role'], q['article'], q['source_token_range'], q['local_token_range'], q['targets']) == expected
    assert q['plan_piece_index'] == 7 and q['bounds'] == [[0, q['targets']]]
    assert [q['lo'], q['hi']] == q['local_token_range']
    assert q['hi'] - q['lo'] - 1 == q['targets']
    assert q['source_token_range'][1] - q['source_token_range'][0] - 1 == q['targets']
    assert q['priors_dtype'] == 'float16' and q['vocabulary_columns'] == 205
assert r['pieces']['train']['article'] != r['pieces']['held']['article']
assert r['pieces']['train']['source_token_range'][1] <= r['pieces']['held']['source_token_range'][0]
assert r['role_contract'] == 'train rows enter ridge; validation rows enter metrics/oracle only'
assert r['manifest'] == dict(training_pieces=1313, training_targets=2941307,
    held_pieces=473, held_targets=1082505)
assert r['checks']['held_rows_in_ridge'] == 0
assert r['checks']['ridge_solves'] == r['checks']['lambda_candidates'] == 1
assert all(v for v in r['checks'].values() if isinstance(v, bool))
ridge = r['ridge']
assert ridge['train_rows'] == 1199 and ridge['input_dim'] == ridge['output_dim'] == 192
assert ridge['lambda_candidates'] == ridge['rank_candidates'] == 1
assert ridge['bias_penalized'] is False and math.isfinite(ridge['lambda_value']) and ridge['lambda_value'] > 0
assert 0 <= ridge['normal_equation_relative_l2'] < 1e-12
assert r['bridge_contract']['main'] == 'x3 + RMSnorm(x3) @ A.T + b'
assert r['bridge_contract']['bridge_arithmetic'] == 'floating FP32'
for role, phases in r['losses'].items():
    for phase, losses in phases.items():
        assert all(math.isfinite(v) and v >= 0 for v in losses.values())
        assert math.isclose(losses['hidden_weighted'], .1 * losses['hidden_mse'], rel_tol=1e-12, abs_tol=1e-15)
        recon = r['loss_reconstruction'][role + '_' + phase]
        for key, value in recon['float64_reconstruction'].items():
            assert math.isclose(losses[key], value, rel_tol=2e-5, abs_tol=2e-6)
            assert math.isclose(recon['absolute_errors'][key], abs(losses[key] - value), rel_tol=1e-12, abs_tol=1e-15)
held = r['losses']['held']
assert held['oracle'] == held['teacher']
for key, delta in r['held_loss_delta_fitted_minus_zero'].items():
    assert math.isclose(delta, held['fitted'][key] - held['zero'][key], rel_tol=1e-12, abs_tol=1e-15)
fraction = (held['zero']['ce'] - held['fitted']['ce']) / (held['zero']['ce'] - held['teacher']['ce'])
assert math.isclose(fraction, r['held_ce_gap_recovery_fraction'], rel_tol=1e-12)
for q in r['bridge_midpoint_error'].values():
    assert q['unbridged_x7_mse'] > 0 and q['fitted_x7_mse'] >= 0
    assert math.isclose(q['residual_error_ratio'], q['fitted_x7_mse'] / q['unbridged_x7_mse'], rel_tol=1e-12)
    assert math.isclose(q['residual_error_reduction_fraction'], 1 - q['residual_error_ratio'], rel_tol=1e-12)
payload = r['bridge_payload_estimate']
assert payload['matrix_rows'] == payload['matrix_columns'] == 192
assert payload['int4_matrix_bytes'] == 192 * 192 // 2
assert payload['fp32_bias_bytes'] == 192 * 4 and payload['bf16_row_scale_bytes'] == 192 * 2
assert payload['nominal_payload_bytes'] == sum(payload[k] for k in ('int4_matrix_bytes', 'fp32_bias_bytes', 'bf16_row_scale_bytes')) == 19584
assert payload['float32_diagnostic_matrix_bias_bytes'] == (192 * 192 + 192) * 4
assert r['raw_readout']['path'] == 'results/depth-middle-bridge.npz'
assert r['raw_readout']['bytes'] == 19888704
assert len(r['raw_readout']['keys']) == len(set(r['raw_readout']['keys'])) == 28
assert r['raw_readout']['keys'] == sorted(r['raw_readout']['keys'])
attempts = r['diagnostic_attempts']
failed = attempts['initial_import_order_failure']
assert failed['status'] == 'failed' and failed['error'] == "No module named 'fla'"
assert failed['teacher_forwards'] == failed['student_forwards'] == failed['ridge_solves'] == 0
assert attempts['explicitly_authorized_import_order_recovery'] == attempts['scientific_successful_measurements'] == 1
assert attempts['exec_posts'] == attempts['science_bundle_gets'] == 2
for key in ('third_execs', 'dependency_installs', 'state_queries', 'resume_requests'):
    assert attempts[key] == 0, key
account = r['execution']
assert account['child_exit_code'] == 0 and account['reruns'] == account['optimizer_updates'] == 0
for q in (failed, account):
    assert math.isclose(q['child_cpu_seconds'] + q['supervisor_cpu_seconds'], q['total_cpu_seconds'], rel_tol=1e-12)
assert math.isclose(account['prior_diagnostic_cpu_seconds'], failed['total_cpu_seconds'], rel_tol=1e-12)
cpu = failed['total_cpu_seconds'] + account['total_cpu_seconds']
wall = failed['wall_seconds'] + account['wall_seconds']
assert math.isclose(cpu, account['cumulative_total_cpu_seconds'], rel_tol=1e-12)
assert 0 < cpu <= account['cumulative_cpu_cap_seconds'] == 180
assert 0 < account['total_cpu_seconds'] <= account['cpu_total_cap_seconds'] == 178
assert 0 < wall <= account['wall_total_cap_seconds'] == 240
assert account['max_process_rss_kib'] == r['peak_rss_kib'] == 581556
assert failed['max_process_rss_kib'] == 244320
assert account['address_space_limit_bytes'] == r['address_space_limit_bytes'] == 4 << 30
assert account['cpu_affinity'] == r['cpu_affinity'] == [1]
assert r['thread_environment'] == dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
assert r['decision']['bridge_full_successor'] == 'stopped'
print(json.dumps(dict(status='passed', stdlib_only=True,
    checks=['saved scientific identities and losses', 'train/held roles and single ridge',
            'recorded x3/skip/oracle/state contracts', 'failure plus explicit recovery counts',
            'child/supervisor and cumulative resource sums'],
    teacher_forwards=2, student_forwards=4, ridge_solves=1,
    cumulative_diagnostic_cpu_seconds=cpu, cumulative_diagnostic_wall_seconds=wall,
    neural_network_reexecutions=0, optimizer_native_codec_calls=0), separators=(',', ':')))
