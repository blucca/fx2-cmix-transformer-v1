#!/usr/bin/env python3
"""One admitted v22/static15 fullstream side, in two explicit phases.

encode runs the retained batch encoder and assembles the real native package.
decode consumes that package once and verifies the official 1GB restoration.
The caller freezes the mixed-selected W and approves resources in entry.json.
No candidate search, prepared decode, native preprocessing, or polling occurs.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from batch import digest, save

PREPARED_BYTES = 586459020
ORIGINAL_BYTES = 1000000000
ORIGINAL_MD5 = 'e206c3450ac99950df65bf70ef61a12d'
W0_SHA256 = '659c0767bce4146d2c457b9aaa13b3e707561ed5d37173a2a6ea1e5965e86b18'
PPMD_MAP_BYTES = 14000 << 20
DISK_HEADROOM_BYTES = 256 << 20
CPU_RESERVE_SECONDS = 900
SIDE_CPU_MAX = 150 * 3600
CAUSAL_FAMILY = 'old-v22-untraced-static15'
DEPTH_FAMILY = 'old-v22-depth8-static15'


def cpu():
    own, child = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + child.ru_utime + child.ru_stime


def measurements_from_mixed(s):
    """Read the sole completed causal/depth W and its actual two-side fees."""
    assert s['status'] == 'complete' and s['original_weight_bytes_equal'] is False
    depth = s.get('family') == DEPTH_FAMILY
    selection, mixed = s['selection'], s['actual_mixed']
    selected = selection['selected']
    assert selection['status'] == 'frozen' and selected['epoch'] > 0 and selected['original_byte_equal'] is False
    if depth:
        assert s['schema'] == 'hutter.depth-mixed.v1'
        baseline = s['baseline_weight']
        assert baseline['bytes'] == 2904325 and baseline['sha256'] == W0_SHA256
        assert s['candidate_weight'] == {'path': selected['weight_file'], 'bytes': selected['weight_bytes'],
                                         'sha256': selection['selected_weight_sha256']}
        initial = selection['initial_student']
        assert initial['epoch'] == 0 and initial['selected'] is False and initial['original_byte_equal'] is False
        assert all(row['epoch'] > 0 for row in selection['completed_exports'])
        terminal = s['pipeline_terminal']
        assert terminal['status'] in {'complete', 'cpu_budget'} and terminal['large_python_child_reaped'] is True
        assert terminal['target_epochs'] == 32 and 0 < selected['epoch'] <= terminal['completed_epochs'] <= 32
        assert selected['epoch'] % 4 == 0
        dual_b = 2*(s['B0']['bytes']-s['B1']['bytes'])
        assert s['dual_code_saved_bytes'] == selected['dual_code_saved_bytes'] == mixed['dual_code_saved_bytes'] == dual_b
        assert s['full_s']['known_dual_code_saved_bytes'] == dual_b
    else:
        zero, = [row for row in selection['completed_exports'] if row['epoch'] == 0]
        assert zero['step'] == 0 and zero['original_byte_equal'] is True and zero['weight_bytes'] == 2904325
        baseline = {'path': zero['weight_file'], 'sha256': W0_SHA256, 'bytes': zero['weight_bytes']}
        dual_b = 0
    assert selected in selection['completed_exports'] and selected['weight_bytes'] > 0
    assert selection['original_weight_sha256'] == W0_SHA256
    assert len(selection['selected_weight_sha256']) == 64 and selection['selected_weight_sha256'] != W0_SHA256
    assert selected['held_total']['count'] == selection['baseline_held_total']['count'] == selection['held_targets'] == 1082505
    double_w = 2*(baseline['bytes'] - selected['weight_bytes'])
    fixed_saved = double_w + dual_b
    assert s['double_weight_saved_bytes'] == selected['double_weight_saved_bytes'] == mixed['double_weight_saved_bytes'] == double_w
    assert s['full_s']['known_double_weight_saved_bytes'] == double_w
    assert s['full_s']['source_bytes'] == PREPARED_BYTES and s['full_s']['native_targets'] == 586215331
    specs = {'prefix8': (0, 8 << 20, 7344758, 1041690, 1041397),
             'q2': (293232482, 1 << 20, 0, 1048576, 1048367)}
    assert len(s['cases']) == 2 and {case['name'] for case in s['cases']} == set(specs)
    scopes = {}
    for name, (offset, length, begin, count, targets) in specs.items():
        case = next(case for case in s['cases'] if case['name'] == name)
        assert tuple(case[k] for k in ['offset', 'length', 'score_begin', 'score_bytes', 'neural_targets']) == (offset, length, begin, count, targets)
        assert case['first_token_fallback_bytes'] == count-targets
        a, b = s['steps'][name+'-baseline'], s['steps'][name+'-candidate']
        pair = s['paired'][name]
        assert a['status'] == b['status'] == 'passed'
        for record in [a, b]:
            encode, interval = record['encode'], record['interval']
            assert encode['mode'] == 'encode' and encode['input_bytes'] == length
            assert math.isfinite(encode['cpu_seconds']) and encode['cpu_seconds'] > 0
            assert interval['status'] == 'passed' and interval['full_input_bytes'] == length
            assert interval['full_coded_bytes'] == encode['output_bytes']
            assert (interval['score_begin'], interval['score_end'], interval['all']['bytes'], interval['neural_targets']['bytes'], interval['pieces']) == (begin, begin+count, count, targets, count-targets)
        decode = b['decode']
        assert decode['mode'] == 'decode' and decode['output_bytes'] == length
        assert math.isfinite(decode['cpu_seconds']) and decode['cpu_seconds'] > 0
        assert a['decode']['original_golden_decode_reused'] is True
        assert a['decode']['restored_slice_sha256'] == decode['restored_slice_sha256']
        assert pair['payload_saved_bytes'] == a['encode']['output_bytes']-b['encode']['output_bytes']
        assert pair['interval_coded_saved_bytes'] == a['interval']['interval_coded_bytes']-b['interval']['interval_coded_bytes']
        assert pair['masks']['neural_targets']['bytes'] == targets
        assert pair['masks']['neural_targets']['final_u16_bits_saved'] == a['interval']['neural_targets']['final_u16_bits']-b['interval']['neural_targets']['final_u16_bits']
        scopes[name] = {'offset': offset, 'input_bytes': length, 'score_begin': begin, 'score_bytes': count,
            'neural_targets': targets, 'first_token_fallback_bytes': count-targets,
            'statistical_start': 'stream0_continuous' if name == 'prefix8' else 'fresh_cold',
            'encode': {side: {'cpu_seconds': r['encode']['cpu_seconds'], 'input_bytes': length,
                'cpu_seconds_per_prepared_byte': r['encode']['cpu_seconds']/length} for side, r in [('baseline', a), ('candidate', b)]},
            'candidate_decode': {'cpu_seconds': decode['cpu_seconds'], 'output_bytes': length,
                'cpu_seconds_per_prepared_byte': decode['cpu_seconds']/decode['output_bytes']}}
    payload = sum(s['paired'][name]['payload_saved_bytes'] for name in specs)
    assert mixed['sample_count'] == 2
    if not depth:
        assert mixed['dual_common_code_delta_bytes'] == 0
    assert mixed['sample_payload_saved_bytes'] == payload and mixed['paired_sample_component_saved_bytes'] == payload+fixed_saved
    mature, prefix, q2 = s['paired']['prefix8'], s['paired']['prefix8']['payload_saved_bytes'], s['paired']['q2']['payload_saved_bytes']
    admission = {'mature_target_gain_bits': mature['masks']['neural_targets']['final_u16_bits_saved'],
        'mature_actual_saved_bytes': mature['interval_coded_saved_bytes'], 'mature_score_bytes': specs['prefix8'][3],
        'prefix_payload_saved_bytes': prefix, 'q2_payload_saved_bytes': q2,
        'dual_code_saved_bytes': dual_b, 'actual_pair_net_saved_bytes': payload+fixed_saved}
    rates = {name: {'score_bytes': count, 'actual_saved_bytes': saved, 'saved_bytes_per_prepared_byte': saved/count,
        'projected_full_payload_saved_bytes': PREPARED_BYTES*saved/count, 'double_weight_saved_bytes': double_w,
        'dual_code_saved_bytes': dual_b, 'projected_full_net_saved_bytes': PREPARED_BYTES*saved/count+fixed_saved}
        for name, saved, count in [('mature', mature['interval_coded_saved_bytes'], specs['prefix8'][3]), ('prefix8', prefix, 8 << 20)]}
    result = {'family': DEPTH_FAMILY if depth else CAUSAL_FAMILY,
            'selected': {'epoch': selected['epoch'], 'step': selected['step'], 'path': selected['weight_file'],
                'sha256': selection['selected_weight_sha256'], 'bytes': selected['weight_bytes']},
            'baseline': baseline, 'admission': admission, 'G_samples': payload+fixed_saved,
            'double_weight_saved_bytes': double_w, 'dual_code_saved_bytes': dual_b,
            'scopes': scopes, 'rates': rates, 'native_reports': selected['report_only']}
    if depth:
        result.update(binaries={'W0': s['B0'], 'W1': s['B1']},
                      probes={'W0': s['baseline_probe'], 'W1': s['candidate_probe']},
                      fixed_aux=s['full_s']['fixed_aux'], training_coverage=terminal)
    return result


def side_records(entry, side):
    """Depth uses its actual per-side probe/B; causal retains the common family."""
    names = ['weights', 'probe', 'binary'] if entry['family'] == DEPTH_FAMILY else ['weights']
    return {**entry['common'], **{name: entry['sides'][side][name] for name in names}}


def admitted(entry):
    """Freeze the caller's measured direction, named record scenario and budget."""
    assert entry['status'] == 'admitted' and entry['family'] in {CAUSAL_FAMILY, DEPTH_FAMILY}
    depth = entry['family'] == DEPTH_FAMILY
    d, sides = entry['admission'], entry['sides']
    assert d['decision'] == 'continue'
    assert all(d[k] is True for k in ['mixed_complete', 'prepared_inverse_passed',
                                    'original_queues_terminal', 'scope_extension_recorded',
                                    'cpu_calendar_passed'])
    assert set(sides) == {'W0', 'W1'}
    assert sides['W0']['weights']['sha256'] == W0_SHA256
    assert sides['W0']['weights']['sha256'] != sides['W1']['weights']['sha256']
    common_names = {'stream', 'vocab', 'dictionary', 'compressed_dictionary', 'compressed_order'}
    assert set(entry['common']) == (common_names if depth else common_names | {'probe', 'binary'})
    double_w = 2*(sides['W0']['weights']['bytes'] - sides['W1']['weights']['bytes'])
    dual_b = 2*(side_records(entry, 'W0')['binary']['bytes']-side_records(entry, 'W1')['binary']['bytes'])
    assert d.get('dual_code_saved_bytes', 0) == dual_b
    if not depth:
        assert dual_b == 0
        assert d['mature_target_gain_bits'] > 0 and d['mature_actual_saved_bytes'] > 0
        assert d['prefix_payload_saved_bytes'] > 0
    assert d['actual_pair_net_saved_bytes'] == d['prefix_payload_saved_bytes'] + d['q2_payload_saved_bytes'] + double_w + dual_b
    assert d['actual_pair_net_saved_bytes'] > 0
    assert d['s0_reference_kind'] in {'actual', 'scenario'} and d['latest_record_bytes'] > 0
    # The fixed-size depth saving may pay for a measured payload increase.
    # Freeze one named scenario; mature is contained in prefix, not added to it.
    scenario = d['record_scenario'] if depth else 'mature'
    assert scenario in {'mature', 'prefix8'}
    score_bytes, payload_saved = ((d['mature_score_bytes'], d['mature_actual_saved_bytes'])
        if scenario == 'mature' else (8 << 20, d['prefix_payload_saved_bytes']))
    assert 0 < score_bytes <= PREPARED_BYTES
    required_saved = d['s0_reference_bytes'] - 99*d['latest_record_bytes']//100
    # Integer cross multiplication retains the exact floor(.99*L) boundary.
    assert (PREPARED_BYTES*payload_saved + (double_w+dual_b)*score_bytes
            >= required_saved*score_bytes), 'named net-S scenario misses the record margin'
    for s in sides.values():
        assert 0 < s['cpu_limit_seconds'] <= SIDE_CPU_MAX and s['prep_cpu_seconds'] >= 0
        estimate = s['forecast_remaining_cpu_seconds']
        assert math.isfinite(estimate) and estimate > 0
        assert s['prep_cpu_seconds'] + estimate + CPU_RESERVE_SECONDS <= s['cpu_limit_seconds']
    assert sum(s['cpu_limit_seconds'] for s in sides.values()) <= 2*SIDE_CPU_MAX


def paired_size(w0, w1):
    """Actual two-package S; double-W and dual-B each occur once."""
    assert w0['identity']['side'] == 'W0' and w1['identity']['side'] == 'W1'
    assert w0['identity']['entry_sha256'] == w1['identity']['entry_sha256'], 'use one frozen admission'
    family = w0['identity']['family']
    assert family == w1['identity']['family'] and family in {CAUSAL_FAMILY, DEPTH_FAMILY}
    p0, p1 = w0['package'], w1['package']
    assert p0['prepared_bytes'] == p1['prepared_bytes'] == PREPARED_BYTES
    c0, c1 = p0['components'], p1['components']
    for name in ['compressed_dictionary', 'compressed_order']:
        assert c0[name] == c1[name], 'use the same frozen D/O'
    if family == CAUSAL_FAMILY:
        assert c0['binary'] == c1['binary'], 'causal uses the same frozen B'
    for package, c in [(p0, c0), (p1, c1)]:
        assert package['full_s'] == (c['raw_payload'] + 2*c['binary']
                + 2*c['compressed_dictionary'] + c['compressed_order'] + 2*c['weights'] + 69)
    saved = p0['full_s'] - p1['full_s']
    payload_saved = c0['raw_payload']-c1['raw_payload']
    double_w, dual_b = 2*(c0['weights']-c1['weights']), 2*(c0['binary']-c1['binary'])
    assert saved == payload_saved + double_w + dual_b
    return {'s0': p0['full_s'], 's1': p1['full_s'], 'net_saved_bytes': saved,
            'payload_saved_bytes': payload_saved, 'double_weight_saved_bytes': double_w,
            'dual_code_saved_bytes': dual_b,
            'formal_decode_status': {s: x['status'] for s, x in [('W0',w0),('W1',w1)]},
            'verification': 'package sizes; each official archive decode is recorded separately'}


def run(a):
    started_cpu = cpu()
    entry = json.loads(a.entry.read_text())
    admitted(entry)
    sides = entry['sides']
    side = sides[a.side]
    root, job = Path(entry['root']).resolve(), a.out.resolve()
    records = side_records(entry, a.side)
    required = {'stream', 'vocab', 'dictionary', 'probe', 'binary',
                'compressed_dictionary', 'compressed_order', 'weights'}
    assert set(records) == required
    files = {name: Path(rec['path']).resolve() for name, rec in records.items()}
    assert files['stream'] == root/'prepared/stream.bin'
    assert files['vocab'] == root/'prepared/vocab.bin'
    assert files['dictionary'] == root/'upstream/dictionary/english.dic'
    assert records['stream']['bytes'] == PREPARED_BYTES and records['vocab']['bytes'] == 32
    for name, path in files.items():
        assert path.stat().st_size == records[name]['bytes'], f'{name} size changed'
        assert digest(path) == records[name]['sha256'], f'{name} identity changed'
    identity = {'entry_sha256': digest(a.entry), 'side': a.side,
                'family': entry['family'], 'files': records}
    job.mkdir(parents=True, exist_ok=True)
    lock = (job/'runner.lock').open('a')
    # Replayed deployment commands share the same simple exclusion as batch.py.
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = job/'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        assert state['identity'] == identity, 'inspect the existing side before changing its entry'
        assert state['status'] in {'encoded', 'passed'}, 'inspect interrupted/failed command and its CPU ledger'
        if state['status'] == 'passed' or a.phase == 'encode':
            print(json.dumps(state), flush=True)
            return state
    else:
        assert a.phase == 'encode', 'decode needs this side\'s completed encode/package receipt'
        state = {'schema': 'hutter.formal-fullstream.side.v1', 'status': 'ready',
                 'identity': identity, 'started_at': time.time(), 'phases': {},
                 'cpu_limit_seconds': side['cpu_limit_seconds'],
                 'cpu_used_seconds': side['prep_cpu_seconds'],
                 'prior_prep_cpu_seconds': side['prep_cpu_seconds']}
    prior_cpu = state['cpu_used_seconds']
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    scripts = Path(__file__).resolve().parent
    package = job/'package'

    def checkpoint(stage, **updates):
        state.update(stage=stage, updated_at=time.time(),
                     cpu_used_seconds=prior_cpu + cpu() - started_cpu, **updates)
        assert state['cpu_used_seconds'] <= state['cpu_limit_seconds'], 'side CPU budget exhausted'
        save(state_path, state)

    def command(stage, argv, cwd):
        remaining = math.floor(state['cpu_limit_seconds'] - prior_cpu - (cpu()-started_cpu)
                               - CPU_RESERVE_SECONDS)
        assert remaining > 30, 'reserve auxiliary, digest and receipt CPU'

        def limits():
            resource.setrlimit(resource.RLIMIT_AS, (24 << 30, 24 << 30))
            resource.setrlimit(resource.RLIMIT_CPU, (remaining-1, remaining))

        # One main model command per phase. The reserve covers the native
        # dictionary subprocess and verification; actual descendants enter rusage.
        # GNU timeout terminates the command group if a native pause exhausts
        # the independent seven-day wall limit. It does not renew VM tokens.
        argv = ['timeout', '-k', '30s', '604800', *map(str, argv)]
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   HUTTER_TRACE_PATH='', HUTTER_PPMD_TRACE_PATH='', HUTTER_MIXER_TRACE_PATH='')
        wall_started = time.time()
        with (job/(stage+'.log')).open('w') as log:
            process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, preexec_fn=limits, env=env)
            checkpoint(stage, status='running', phase=a.phase, command=argv,
                       command_pid=process.pid,
                       boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                       wrapper_pid=os.getpid(), child_cpu_limit_seconds=remaining)
            _, status, usage = os.wait4(process.pid, 0)
            code = os.waitstatus_to_exitcode(status)
            process.returncode = code
        # wait4 measures this waited command and its reaped descendants only;
        # the cumulative owner ledger below adds each phase once.
        state['phases'][stage] = {'exit_code': code, 'cpu_seconds': usage.ru_utime + usage.ru_stime,
                                 'max_process_rss_kib': usage.ru_maxrss,
                                 'wall_elapsed_seconds': time.time()-wall_started,
                                 'command': argv, 'log': str(job/(stage+'.log')),
                                 'resource_scope': 'wait4 command and reaped descendants; largest process RSS'}
        checkpoint(stage, command_exit_code=code)
        if code:
            raise subprocess.CalledProcessError(code, argv)

    try:
        scratch = PPMD_MAP_BYTES + PREPARED_BYTES + DISK_HEADROOM_BYTES
        if a.phase == 'decode':
            scratch += ORIGINAL_BYTES
        free = shutil.disk_usage(job).free
        checkpoint('inputs_ready', phase=a.phase, free_disk_bytes=free,
                   required_free_disk_bytes=scratch)
        assert free >= scratch, 'reclaim completed caches before this model phase'
        if a.phase == 'encode':
            q = job/'batch-root'
            (q/'build'/a.side).mkdir(parents=True, exist_ok=True)
            for path, target in [(q/'prepared', root/'prepared'), (q/'upstream', root/'upstream'),
                                 (q/'build'/a.side/'probe', files['probe'])]:
                if path.is_symlink():
                    assert path.resolve() == target
                else:
                    path.symlink_to(target)
            plan = job/'plan.json'
            plan.write_text(json.dumps([{'name': 'fullstream', 'offset': 0, 'length': PREPARED_BYTES}])+'\n')
            command('prepared_encode', [sys.executable, '-B', '-u', scripts/'batch.py',
                    '--root', q, '--plan', plan, '--out', job/'runs', '--variants', a.side,
                    '--weights', a.side+'='+str(files['weights']), '--phase', 'encode',
                    '--timeout', '604800'], job)
            trial = job/'runs/fullstream'/a.side
            encoded = json.loads((trial/'result.json').read_text())
            payload = trial/'encoded.bin'
            assert encoded['status'] == 'encoded'
            assert encoded['identity']['sample'] == {'name':'fullstream','offset':0,'length':PREPARED_BYTES}
            for name, key in [('stream', 'stream_sha256'), ('vocab', 'vocab_sha256'),
                              ('dictionary', 'dictionary_sha256'), ('weights', 'weights_sha256'),
                              ('probe', 'binary_sha256')]:
                assert encoded['identity'][key] == records[name]['sha256']
            assert encoded['encode']['input_bytes'] == PREPARED_BYTES
            assert payload.stat().st_size == encoded['encoded_bytes'] == encoded['encode']['output_bytes']
            checkpoint('assemble_native', status='running', encoded_receipt=str(trial/'result.json'),
                       raw_payload_bytes=payload.stat().st_size, raw_payload_sha256=digest(payload))
            subprocess.run([sys.executable, '-B', str(scripts/'archive-package.py'),
                '--binary', str(files['binary']), '--compressed-dictionary', str(files['compressed_dictionary']),
                '--compressed-order', str(files['compressed_order']), '--weights', str(files['weights']),
                '--raw-payload', str(payload), '--vocab', str(files['vocab']),
                '--prepared-bytes', str(PREPARED_BYTES), '--out', str(package)], check=True)
            p = json.loads((package/'package.json').read_text())
            checkpoint('encoded_package_ready', status='encoded', package=p,
                       package_files={name: {'bytes': (package/name).stat().st_size,
                                             'sha256': digest(package/name)} for name in ['cmix','archive9']},
                       next_step='compare both actual S values, then explicitly start each formal decode')
        else:
            assert state['status'] == 'encoded'
            for name, rec in state['package_files'].items():
                assert (package/name).stat().st_size == rec['bytes']
                assert digest(package/name) == rec['sha256']
            command('formal_decode', ['./archive9'], package)
            checkpoint('verify_official_original', status='running')
            final = package/'enwik9_uncompressed'
            h = hashlib.md5()
            with final.open('rb') as f:
                for block in iter(lambda: f.read(1 << 20), b''):
                    h.update(block)
            assert final.stat().st_size == ORIGINAL_BYTES
            assert h.hexdigest() == ORIGINAL_MD5
            retained = {'cmix', 'archive9', 'package.json', 'enwik9_uncompressed'}
            # Only this fresh package directory receives native intermediates.
            for path in package.iterdir():
                if path.is_file() and path.name not in retained:
                    path.unlink()
            checkpoint('formal_original_passed', status='passed', completed_at=time.time(),
                       next_step='collect paired formal results and actual resource qualification',
                       result={'restored_bytes': final.stat().st_size, 'restored_md5': h.hexdigest(),
                               'full_s': state['package']['full_s'], 'retained_original': str(final),
                               'actual_compressor_e_preprocessing': 'outside this prepared-encode receipt'})
    except Exception as error:
        state.update(status='failed', error=repr(error), completed_at=time.time(),
                     cpu_used_seconds=prior_cpu + cpu() - started_cpu)
        save(state_path, state)
        raise
    print(json.dumps(state), flush=True)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', type=Path, required=True)
    parser.add_argument('--side', choices=['W0', 'W1'], required=True)
    parser.add_argument('--phase', choices=['encode', 'decode'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
