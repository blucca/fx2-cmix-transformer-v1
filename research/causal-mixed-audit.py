#!/usr/bin/env python3
"""One frozen causal-v1 export, original prefix8/q2 mixed pairs, <=8 CPU hours."""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

CPU_LIMIT = 28800
TRACE_LIMIT = 3 << 30
FULL_BYTES = 586459020
FULL_TARGETS = 586215331
SEPARATOR = bytes([8, 8, 37, 172, 101, 39, 5, 8, 8, 8, 8, 37, 172, 104, 39])


def native_total(result, expected):
    """Use the native target-weighted total; complete-piece rows verify its mask."""
    total = result['total']
    assert total['count'] == expected and math.isfinite(total['logits_mean'])
    if 'piece_losses' in result:
        pieces = [row['loss'] for row in result['piece_losses']]
        count = sum(row['count'] for row in pieces)
        mean = sum(row['count'] * row['logits_mean'] for row in pieces) / count
        assert count == expected and math.isclose(mean, total['logits_mean'], rel_tol=1e-12, abs_tol=1e-12)
    return {'count': total['count'], 'logits_mean': total['logits_mean']}


def choose_export(baseline, full_targets, original_bytes, exports):
    """Frozen kappa=1 proxy; independent max audit and long3 stay report-only."""
    rows = []
    for export in exports:
        held = export['held_total']
        assert held['count'] == baseline['count'] and math.isfinite(held['logits_mean'])
        saved = (baseline['logits_mean'] - held['logits_mean']) * full_targets / (8 * math.log(2))
        double_saved = 2 * (original_bytes - export['weight_bytes'])
        rows.append(dict(export, held_delta_raw_nll=held['logits_mean'] - baseline['logits_mean'],
                         projected_payload_saved_bytes=saved, double_weight_saved_bytes=double_saved,
                         proxy_net_saved_bytes=saved + double_saved))
    assert rows and len({row['epoch'] for row in rows}) == len(rows)
    different = [row for row in rows if row['epoch'] > 0 and not row['original_byte_equal']]
    pool = different or [row for row in rows if row['epoch'] == 0]
    assert pool
    best = max(pool, key=lambda row: (row['proxy_net_saved_bytes'], -row['epoch']))
    return best, sorted(rows, key=lambda row: row['epoch'])


def aggregate(paired, double_saved):
    payload = sum(row['payload_saved_bytes'] for row in paired.values())
    return {'scope': 'two fixed sample payloads plus one shared double-weight delta; common static15 code',
            'sample_count': len(paired), 'sample_payload_saved_bytes': payload,
            'double_weight_saved_bytes': double_saved, 'dual_common_code_delta_bytes': 0,
            'paired_sample_component_saved_bytes': payload + double_saved}


def selection_range(plan, boundaries):
    """Report realized length coverage alongside the frozen unstratified held mean."""
    caps = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
    counts = {role: {cap: {'pieces': 0, 'targets': 0} for cap in caps}
              for role in ('full', 'train', 'validation')}
    def add(role, length):
        cap = next(cap for cap in caps if length <= cap)
        counts[role][cap]['pieces'] += 1
        counts[role][cap]['targets'] += length - 1
    assert boundaries[0] == 0 and boundaries[-1] == FULL_BYTES
    for start, end in zip(boundaries, boundaries[1:]):
        for lo in range(start, end, 1 << 17):
            add('full', min(end - lo, 1 << 17))
    for row in plan['intervals']:
        add(row['role'], row['end'] - row['begin'])
    assert sum(row['targets'] for row in counts['full'].values()) == FULL_TARGETS
    for role in ('train', 'validation'):
        assert sum(row['targets'] for row in counts[role].values()) == plan['roles'][role]['targets']
    strata, lower = [], 0
    for cap in caps:
        strata.append({'piece_bytes': [lower + 1, cap],
            **{role: dict(counts[role][cap], target_fraction=counts[role][cap]['targets'] /
                          (FULL_TARGETS if role == 'full' else plan['roles'][role]['targets']))
               for role in counts}})
        lower = cap
    return {'seed': plan['seed'], 'article_holdout_probability': plan['article_holdout_probability'],
            'piece_sampling_probability': plan['probability'], 'train_piece_limit': plan['train_piece_limit'],
            'held_max_piece_bytes': plan['roles']['validation']['max_piece_bytes'],
            'score_weighting': 'realized target share across sampled held pieces', 'length_strata': strata}


def report_baselines(baseline, held_sample):
    reports = {name: {'targets': row['result']['total']['count'],
                     'raw_nll': row['result']['total']['logits_mean'], 'delta_raw_nll': 0}
               for name, row in baseline['steps'].items()}
    indices = held_sample.get('long_piece_indices', [])
    if indices:
        losses = [baseline['steps']['causal']['result']['piece_losses'][i]['loss'] for i in indices]
        count = sum(row['count'] for row in losses)
        reports['causal']['long_native'] = {'targets': count,
            'raw_nll': sum(row['count'] * row['logits_mean'] for row in losses) / count, 'delta_raw_nll': 0}
    return reports


def activation_result(path):
    rows = [json.loads(line.split(' ', 1)[1]) for line in path.read_text().splitlines()
            if line.startswith('ACTIVATION_RESULT ')]
    assert len(rows) == 1, str(path)
    return rows[0]


def probe_result(path):
    rows = [json.loads(line.split(' ', 1)[1]) for line in path.read_text().splitlines()
            if line.startswith('PROBE_RESULT ')]
    assert len(rows) == 1, str(path)
    return rows[0]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def target_count(tokens):
    """Exactly the reducer's separator-END / 2^17 / first-token fallback mask."""
    window = bytearray([255] * 15)
    in_piece = pieces = 0
    for token in tokens:
        if in_piece == 0:
            pieces += 1
        window.pop(0)
        window.append(token)
        in_piece += 1
        if in_piece == 1 << 17 or window == SEPARATOR:
            in_piece = 0
    return len(tokens) - pieces, pieces


def paired_record(a, b):
    x, y = a['interval'], b['interval']
    for key in ('full_input_bytes', 'score_begin', 'score_end', 'pieces'):
        assert x[key] == y[key], key
    assert x['all']['bytes'] == y['all']['bytes']
    assert x['neural_targets']['bytes'] == y['neural_targets']['bytes']
    masks = {}
    for mask in ('all', 'neural_targets', 'first_token_fallback'):
        if mask == 'first_token_fallback':
            left = {k: x['all'][k] - x['neural_targets'][k] for k in x['all']}
            right = {k: y['all'][k] - y['neural_targets'][k] for k in y['all']}
        else:
            left, right = x[mask], y[mask]
        assert left['bytes'] == right['bytes']
        masks[mask] = {'bytes': left['bytes'],
            **{k + '_saved': left[k] - right[k] for k in ('neural_bits', 'pre_sse_logit_bits', 'final_u16_bits')},
            'baseline_override_bits': left['override_bits'], 'candidate_override_bits': right['override_bits']}
    g = masks['neural_targets']['neural_bits_saved']
    transfer = {'status': 'defined' if g else 'zero_guarded_neural_gain',
                'scope': 'same NN-target mask and score interval; neural f16/guard boundary',
                'guarded_neural_saved_bits': g,
                'kappa_final_nn_targets': masks['neural_targets']['final_u16_bits_saved'] / g if g else None,
                'kappa_interval_bytes': 8 * (x['interval_coded_bytes'] - y['interval_coded_bytes']) / g if g else None,
                'fallback_final_saved_bits': masks['first_token_fallback']['final_u16_bits_saved'],
                'range_residual_saved_bytes': x['interval_range_residual_bytes'] - y['interval_range_residual_bytes']}
    return {'payload_saved_bytes': a['encode']['output_bytes'] - b['encode']['output_bytes'],
            'guarded_transfer': transfer,
            'interval_coded_saved_bytes': x['interval_coded_bytes'] - y['interval_coded_bytes'],
            'interval_range_residual_saved_bytes': x['interval_range_residual_bytes'] - y['interval_range_residual_bytes'],
            'same_native_target_mask': True, 'masks': masks}


def self_check():
    baseline = native_total({'total': {'count': 8, 'logits_mean': 2.5}, 'piece_losses': [
        {'loss': {'count': 2, 'logits_mean': 1.}}, {'loss': {'count': 6, 'logits_mean': 3.}}]}, 8)
    exports = [dict(epoch=e, weight_bytes=w, held_total={'count': 8, 'logits_mean': n},
                    original_byte_equal=(e == 0), report_only={'long-audit': {'raw_nll': audit}})
               for e, w, n, audit in [(0, 100, 2.5, 0.), (1, 104, 1.5, -1000.),
                                     (3, 100, 2., -2000.), (2, 100, 2., 1000.)]]
    best, rows = choose_export(baseline, 80 * math.log(2), 100, exports)
    assert best['epoch'] == 2 and math.isclose(best['proxy_net_saved_bytes'], 5.)
    assert math.isclose(rows[1]['proxy_net_saved_bytes'], 2.)
    best0, _ = choose_export(baseline, FULL_TARGETS, 100, exports[:1])
    assert best0['epoch'] == 0 and best0['proxy_net_saved_bytes'] == 0
    equal = dict(exports[0], epoch=1)
    worse = dict(exports[0], epoch=2, original_byte_equal=False, weight_bytes=101,
                 held_total={'count': 8, 'logits_mean': 2.6})
    selected, _ = choose_export(baseline, FULL_TARGETS, 100, [exports[0], equal, worse])
    assert selected['epoch'] == 2 and selected['proxy_net_saved_bytes'] < 0
    selected, _ = choose_export(baseline, FULL_TARGETS, 100, [exports[0], equal])
    assert selected['epoch'] == 0
    summary = aggregate({'prefix8': {'payload_saved_bytes': 10}, 'q2': {'payload_saved_bytes': -2}}, -6)
    assert summary['sample_payload_saved_bytes'] == 8 and summary['paired_sample_component_saved_bytes'] == 2
    assert target_count(bytes([2, 3, 4, 5])) == (3, 1)
    fixture = {'encode': {'output_bytes': 9}, 'interval': {'full_input_bytes': 8, 'score_begin': 2,
        'score_end': 6, 'pieces': 1, 'interval_coded_bytes': 4, 'interval_range_residual_bytes': 0,
        'all': {'bytes': 4, 'neural_bits': 32, 'pre_sse_logit_bits': 32, 'final_u16_bits': 32, 'override_bits': 0},
        'neural_targets': {'bytes': 3, 'neural_bits': 24, 'pre_sse_logit_bits': 24, 'final_u16_bits': 24, 'override_bits': 0}}}
    pair = paired_record(fixture, fixture)
    assert pair['payload_saved_bytes'] == pair['interval_coded_saved_bytes'] == 0
    assert pair['masks']['first_token_fallback']['bytes'] == 1
    assert all(row['final_u16_bits_saved'] == 0 for row in pair['masks'].values())
    assert pair['guarded_transfer']['status'] == 'zero_guarded_neural_gain'
    changed = json.loads(json.dumps(fixture))
    changed['encode']['output_bytes'] -= 1
    changed['interval']['interval_coded_bytes'] -= 1
    for mask in ['all', 'neural_targets']:
        changed['interval'][mask]['neural_bits'] -= 4
        changed['interval'][mask]['pre_sse_logit_bits'] -= 4
        changed['interval'][mask]['final_u16_bits'] -= 8
    transfer = paired_record(fixture, changed)['guarded_transfer']
    assert transfer['kappa_final_nn_targets'] == transfer['kappa_interval_bytes'] == 2
    print(json.dumps({'status': 'passed', 'checks': ['target-weighted native total', 'kappa1 net selection',
        'earliest-epoch tie', 'audit report-only', 'zero export', 'double weights counted once', 'paired trace masks', 'byte-different negative proxy', 'all-equal reuse', 'guarded kappa']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--base', type=Path, help='Original build/baseline-trace objects')
    parser.add_argument('--loader-object', type=Path, help='Original static15 loader object, copied read-only')
    parser.add_argument('--interval-source', type=Path, help='Public unchanged mixed-interval.cpp')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    job = args.job.resolve()
    root, training_dir = job.parent, job / 'training'
    out = job / 'mixed-audit'
    out.mkdir(exist_ok=True)
    state_path = out / 'state.json'
    assert not state_path.exists(), 'inspect the existing mixed receipt before starting another queue'
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (24 << 30, 24 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT, CPU_LIMIT))
    state = {'status': 'waiting_pipeline_terminal', 'started_at': time.time(), 'steps': {}, 'paired': {},
             'cpu_limit_seconds': CPU_LIMIT, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
             'address_space_limit_gib': 24, 'trace_budget_bytes': TRACE_LIMIT, 'extra_machine_restarts': 0,
             'selection_recipe': 'trained-byte-different-random-held-kappa1-minus-doubleW-earliest-epoch'}

    def cpu():
        a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime

    def save():
        state.update(cpu_seconds=cpu(), updated_at=time.time(),
                     child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(state_path)

    def run(label, command, trace=None):
        remaining = math.floor(CPU_LIMIT - cpu()) - 2
        if remaining < 30:
            raise TimeoutError('mixed stage CPU budget exhausted')
        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (remaining - 1, remaining))
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   HUTTER_TRACE_PATH=str(trace) if trace else '', HUTTER_PPMD_TRACE_PATH='', HUTTER_MIXER_TRACE_PATH='')
        state.update(status='running', phase=label, command=list(map(str, command)))
        save()
        log = out / (label + '.log')
        with log.open('w') as f:
            subprocess.run(state['command'], stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                           check=True, timeout=max(900, 3 * remaining), cwd=out, env=env, preexec_fn=limits)
        save()
        return log

    save()
    try:
        while True:
            pipeline = json.loads((job / 'pipeline-state.json').read_text())
            if pipeline['status'] in ('failed', 'stopped'):
                state.update(status='stopped', reason='pipeline ' + pipeline['status'], pipeline=pipeline)
                save()
                return
            if pipeline['status'] == 'passed':
                assert pipeline['stage'] == 'training_terminal', pipeline
                # training-job saved this receipt after p.wait(): the large Python child has exited.
                code = int((job / 'training.exit').read_text())
                trajectory = json.loads((training_dir / 'state.json').read_text())
                if trajectory['status'] == 'stopped':
                    state.update(status='stopped', reason='training stopped', pipeline=pipeline)
                    save()
                    return
                assert code in (0, 124) and trajectory['status'] in ('complete', 'cpu_budget')
                assert pipeline['training_status'] == trajectory['status']
                break
            time.sleep(60)
        state['pipeline_terminal'] = {'status': pipeline['status'], 'stage': pipeline['stage'],
            'training_status': trajectory['status'], 'training_exit': code,
            'training_epoch': trajectory['epoch'], 'training_step': trajectory['step'],
            'large_python_child_reaped': True}
        plan = json.loads((job / 'data/plan.json').read_text())
        data = json.loads((job / 'data-manifest.json').read_text())
        assert plan['source_bytes'] == FULL_BYTES and plan['full_targets'] == FULL_TARGETS
        assert (root / 'prepared/stream.bin').stat().st_size == FULL_BYTES
        held_sample = next(s for s in data['samples'] if s['name'] == 'causal')['score']
        held_targets = plan['roles']['validation']['targets']
        assert held_sample['targets'] == held_targets
        baseline = json.loads((job / 'baseline.json').read_text())
        assert baseline['status'] == 'passed'
        original = job / 'assets/original-static15.tfwc3'
        assert (training_dir / 'original-static15.tfwc3').read_bytes() == original.read_bytes()
        base_total = native_total(baseline['steps']['causal']['result'], held_targets)
        zero = trajectory['zero_step']
        assert Path(zero['weight_file']).read_bytes() == original.read_bytes()
        assert zero['weight_bytes'] == original.stat().st_size == trajectory['original_static15_bytes']
        exports = [dict(epoch=0, step=0, weight_file=zero['weight_file'], weight_bytes=zero['weight_bytes'],
                        held_total=base_total, original_byte_equal=True, report_only=report_baselines(baseline, held_sample))]
        for row in trajectory['evaluations']:
            epoch = row['epoch']
            held = native_total(activation_result(training_dir / f'epoch-{epoch:02d}-causal.log'), held_targets)
            assert held['count'] == row['scores']['causal']['targets']
            assert held['logits_mean'] == row['scores']['causal']['raw_nll']
            assert Path(row['weight_file']).stat().st_size == row['weight_bytes']
            exports.append(dict(epoch=epoch, step=row['step'], weight_file=row['weight_file'],
                                weight_bytes=row['weight_bytes'], held_total=held,
                                original_byte_equal=(Path(row['weight_file']).read_bytes() == original.read_bytes()), report_only=row['scores']))
        best, scores = choose_export(base_total, FULL_TARGETS, original.stat().st_size, exports)
        weight = Path(best['weight_file'])
        boundaries = array('i')
        boundaries.frombytes((root / 'prepared/article_boundaries.i32').read_bytes())
        state['selection'] = {'status': 'frozen', 'scope': 'one byte-different trained export ranked by randomized held proxy; zero is the keep/delivery baseline',
            'criterion': '(original held logits_mean - candidate held logits_mean) * fullN / (8 ln2) - 2 * (candidate W bytes - original W bytes)',
            'held_targets': held_targets, 'full_target_count': FULL_TARGETS, 'kappa': 1,
            'realized_sampling_range': selection_range(plan, boundaries),
            'zero_role': 'keep/delivery baseline; reuse if all trained weights byte-equal',
            'trained_nonpositive_proxy': 'measure one byte-different candidate for coupled codec gain',
            'tie_breaker': 'earliest epoch', 'independent_max_audit_role': 'report-only', 'held_long3_role': 'report-only',
            'baseline_held_total': base_total, 'completed_exports': scores, 'selected': best,
            'selected_weight_sha256': sha(weight), 'original_weight_sha256': sha(original)}
        state['double_weight_saved_bytes'] = best['double_weight_saved_bytes']
        state['full_s'] = {'status': 'awaiting_actual_full_stream_package', 'source_bytes': FULL_BYTES,
            'native_targets': FULL_TARGETS, 'known_double_weight_saved_bytes': best['double_weight_saved_bytes'],
            'same_static15_dual_code_delta_bytes': 0, 'actual_full_s_saved_bytes': None,
            'scope': 'actual full archive + compressor + decompressor + double weights + unchanged auxiliaries'}
        save()

        stream, vocab, dictionary = root / 'prepared/stream.bin', root / 'prepared/vocab.bin', root / 'upstream/dictionary/english.dic'
        packed = vocab.read_bytes()
        assert len(packed) == 32
        values = [b for b in range(256) if (packed[b // 8] >> (b % 8)) & 1]
        assert len(values) == 205
        mapping = bytearray([255] * 256)
        for token, byte in enumerate(values):
            mapping[byte] = token
        token_dir = out / 'data'
        token_dir.mkdir(exist_ok=True)
        with stream.open('rb') as f:
            prefix = f.read(8 << 20)
        sep = bytes(values[i] for i in SEPARATOR)
        ends, cursor = [], 0
        while True:
            pos = prefix.find(sep, cursor)
            if pos < 0:
                break
            cursor = pos + len(sep)
            ends.append(cursor)
        mature_begin, mature_end = next(e for e in ends if e >= 7 << 20), ends[-1]
        assert (mature_begin, mature_end) == (7344758, 8386448)
        mature_tokens = prefix[mature_begin:mature_end].translate(mapping)
        with stream.open('rb') as f:
            f.seek(293232482)
            q2 = f.read(1 << 20)
        q2_tokens = q2.translate(mapping)
        cases = [('prefix8', 0, 8 << 20, root / 'runs/long-prefix/prefix-8m/baseline',
                  mature_tokens, mature_begin, 1072279, 1041397),
                 ('q2', 293232482, 1 << 20, root / 'runs/first-ab/article-q2-1m/baseline',
                  q2_tokens, 0, 208769, 1048367)]
        identities = {'stream_sha256': sha(stream), 'vocab_sha256': sha(vocab), 'dictionary_sha256': sha(dictionary),
                      'weights_sha256': sha(root / 'candidate/models/6m-q4-fp32.tfwc2')}
        goldens = {}
        for name, offset, length, golden, tokens, begin, expected, targets in cases:
            assert 255 not in tokens and target_count(tokens)[0] == targets
            path = token_dir / (name + '.tokens.u8')
            path.write_bytes(tokens)
            old = json.loads((golden / 'result.json').read_text())
            assert old['status'] == 'passed' and old['encoded_bytes'] == expected
            assert (golden / 'encoded.bin').stat().st_size == expected
            assert old['identity']['sample']['offset'] == offset and old['identity']['sample']['length'] == length
            for key, digest in identities.items():
                assert old['identity'][key] == digest, key
            raw = prefix if name == 'prefix8' else q2
            source_digest = hashlib.sha256(raw).hexdigest()
            assert old['original_slice_sha256'] == old['restored_slice_sha256'] == source_digest
            assert old['decode']['mode'] == 'decode' and old['decode']['output_bytes'] == length
            goldens[name] = old
        state['cases'] = [{'name': name, 'offset': offset, 'length': length, 'golden': str(golden),
                           'score_begin': begin, 'score_bytes': len(tokens), 'neural_targets': targets,
                           'first_token_fallback_bytes': len(tokens) - targets,
                           'token_source': 'exact prepared stream bytes through original vocabulary'}
                          for name, offset, length, golden, tokens, begin, expected, targets in cases]
        if weight.read_bytes() == original.read_bytes():
            state['original_weight_bytes_equal'] = True
            for name, offset, length, golden, tokens, begin, expected, targets in cases:
                state['steps'][name + '-reused'] = {'status': 'passed', 'source_result': str(golden / 'result.json'),
                    'original_golden_decode_reused': True, 'encoded_bytes': expected, 'weight_bytes_equal': True}
                state['paired'][name] = {'payload_saved_bytes': 0, 'interval_coded_saved_bytes': 0,
                    'same_native_target_mask': True, 'neural_targets': targets,
                    'first_token_fallback_bytes': len(tokens) - targets,
                    'method': 'identical static15 weights and common codec; original passed round trips reused'}
            state['actual_mixed'] = aggregate(state['paired'], best['double_weight_saved_bytes'])
            state['full_s'].update(status='closed_original_byte_identity', causal_component_saved_bytes=0,
                                   method='common codec and byte-identical original static15 weights; existing original receipts retained')
            state.update(status='complete', reason='selected original-byte weights; payload delta 0 and original decode receipts reused', finished_at=time.time())
            save()
            return

        state['original_weight_bytes_equal'] = False
        trace_planned = 2 * ((8 << 20) + (1 << 20)) * 128
        assert trace_planned <= TRACE_LIMIT and shutil.disk_usage(out).free > TRACE_LIMIT
        state['trace_planned_bytes'] = trace_planned
        base = (args.base or root / 'build/baseline-trace').resolve()
        loader = (args.loader_object or job / 'assets/tf_weights_io_compressed.o').resolve()
        interval_source = (args.interval_source or job / 'assets/mixed-interval.cpp').resolve()
        build_dir = job / 'mixed-build'
        if not (build_dir / 'build.json').is_file():
            run('build-mixed', ['python3', job / 'build-mixed.py', '--base', base, '--loader-object', loader,
                              '--interval-source', interval_source, '--out', build_dir])
        build = json.loads((build_dir / 'build.json').read_text())
        assert build['status'] == 'built' and build['reference_relink_byte_equal'] and build['retained_main_abi']
        state['build'] = build
        probe, interval = build_dir / 'probe', build_dir / 'mixed-interval'
        trace_used = 0
        for name, offset, length, golden, tokens, begin, expected, targets in cases:
            for side, selected_weight in [('baseline', original), ('candidate', weight)]:
                destination = out / name / side
                destination.mkdir(parents=True, exist_ok=True)
                encoded, trace = destination / 'encoded.bin', destination / 'trace.bin'
                log = run(name + '-' + side + '-encode', [probe, 'encode', stream, encoded, vocab,
                          dictionary, selected_weight, offset, length], trace)
                encoded_result = probe_result(log)
                assert encoded_result['mode'] == 'encode' and encoded_result['input_bytes'] == length
                assert encoded_result['output_bytes'] == encoded.stat().st_size
                assert trace.stat().st_size == length * 128
                trace_used += trace.stat().st_size
                assert trace_used <= TRACE_LIMIT
                state['trace_used_bytes'] = trace_used
                if side == 'baseline':
                    assert encoded.read_bytes() == (golden / 'encoded.bin').read_bytes(), 'original golden encode byte mismatch'
                    decoded_result = {'original_golden_decode_reused': True, 'source_result': str(golden / 'result.json'),
                                      'restored_slice_sha256': goldens[name]['restored_slice_sha256']}
                else:
                    decoded_path = destination / 'decoded.bin'
                    decode_log = run(name + '-' + side + '-decode', [probe, 'decode', encoded, decoded_path,
                                     vocab, dictionary, selected_weight, offset, length])
                    decoded_result = probe_result(decode_log)
                    assert decoded_result['mode'] == 'decode' and decoded_result['output_bytes'] == length
                    raw = prefix if name == 'prefix8' else q2
                    assert decoded_path.read_bytes() == raw, 'candidate restored bytes differ'
                    decoded_result['restored_slice_sha256'] = sha(decoded_path)
                interval_log = run(name + '-' + side + '-interval', [interval, trace, token_dir / (name + '.tokens.u8'),
                                   vocab, begin, encoded.stat().st_size])
                accounting = json.loads(interval_log.read_text())
                assert accounting['status'] == 'passed' and accounting['full_input_bytes'] == length
                assert accounting['full_coded_bytes'] == encoded.stat().st_size
                assert accounting['score_begin'] == begin and accounting['score_end'] == begin + len(tokens)
                assert accounting['all']['bytes'] == len(tokens) and accounting['neural_targets']['bytes'] == targets
                record = {'status': 'passed', 'encode': encoded_result, 'decode': decoded_result, 'interval': accounting,
                          'encoded_sha256': sha(encoded), 'trace_bytes': trace.stat().st_size}
                (destination / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
                state['steps'][name + '-' + side] = record
                save()
            state['paired'][name] = paired_record(state['steps'][name + '-baseline'], state['steps'][name + '-candidate'])
            save()
        state['actual_mixed'] = aggregate(state['paired'], best['double_weight_saved_bytes'])
        state.update(status='complete', finished_at=time.time())
        save()
    except TimeoutError as exc:
        state.update(status='cpu_budget', error=repr(exc), finished_at=time.time())
        save()
        raise
    except Exception as exc:
        state.update(status='failed', error=repr(exc), finished_at=time.time())
        save()
        raise


if __name__ == '__main__':
    main()
