#!/usr/bin/env python3
"""A small orchestration fixture; native/model/full-1GB executions are zero."""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--work', type=Path, required=True)
root = p.parse_args().work.resolve()
root.mkdir(parents=True, exist_ok=False)
source = Path(__file__).with_name('causal-fullstream-job.py')
spec = importlib.util.spec_from_file_location('formal_job', source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# One synthetic completed mixed receipt exercises the pure source-state reader.
zero = {'epoch': 0, 'step': 0, 'weight_file': '/synthetic/W0', 'weight_bytes': 2904325,
        'original_byte_equal': True, 'held_total': {'count': 1082505, 'logits_mean': 1.0}}
selected = {'epoch': 1, 'step': 4, 'weight_file': '/synthetic/W1', 'weight_bytes': 2904326,
            'original_byte_equal': False, 'double_weight_saved_bytes': -2,
            'held_total': {'count': 1082505, 'logits_mean': .999},
            'report_only': {'causal': {'targets': 1082505, 'delta_raw_nll': -.001,
                                      'long_native': {'targets': 100000, 'delta_raw_nll': .01}},
                            'long-audit': {'targets': 131071, 'delta_raw_nll': .02}}}
source_state = {'status': 'complete', 'original_weight_bytes_equal': False,
    'selection': {'status': 'frozen', 'selected': selected, 'completed_exports': [zero, selected],
                  'original_weight_sha256': m.W0_SHA256, 'selected_weight_sha256': '1'*64,
                  'baseline_held_total': zero['held_total'], 'held_targets': 1082505},
    'double_weight_saved_bytes': -2,
    'full_s': {'source_bytes': m.PREPARED_BYTES, 'native_targets': 586215331,
               'known_double_weight_saved_bytes': -2}, 'cases': [], 'steps': {}, 'paired': {},
    'actual_mixed': {'sample_count': 2, 'double_weight_saved_bytes': -2,
                     'dual_common_code_delta_bytes': 0, 'sample_payload_saved_bytes': 26,
                     'paired_sample_component_saved_bytes': 24}}
for name, offset, length, begin, count, targets, p0, saved, interval_saved, gain in [
        ('prefix8', 0, 8 << 20, 7344758, 1041690, 1041397, 1072279, 30, 5, 32),
        ('q2', 293232482, 1 << 20, 0, 1048576, 1048367, 208769, -4, -4, -8)]:
    source_state['cases'].append({'name': name, 'offset': offset, 'length': length,
        'score_begin': begin, 'score_bytes': count, 'neural_targets': targets,
        'first_token_fallback_bytes': count-targets})
    restored_sha = hashlib.sha256(name.encode()).hexdigest()
    for side, change, enc_cpu in [('baseline', 0, 2), ('candidate', 1, 4)]:
        coded = p0-change*saved
        source_state['steps'][name+'-'+side] = {'status': 'passed',
            'encode': {'mode': 'encode', 'input_bytes': length, 'output_bytes': coded,
                       'cpu_seconds': enc_cpu},
            'decode': ({'original_golden_decode_reused': True, 'restored_slice_sha256': restored_sha}
                if side == 'baseline' else {'mode': 'decode', 'input_bytes': coded,
                    'output_bytes': length, 'cpu_seconds': 8, 'restored_slice_sha256': restored_sha}),
            'interval': {'status': 'passed', 'full_input_bytes': length, 'full_coded_bytes': coded,
                'score_begin': begin, 'score_end': begin+count, 'pieces': count-targets,
                'all': {'bytes': count}, 'neural_targets': {'bytes': targets,
                    'final_u16_bits': 1000-change*gain},
                'interval_coded_bytes': (100000 if name == 'prefix8' else p0-1)-change*interval_saved}}
    source_state['paired'][name] = {'payload_saved_bytes': saved,
        'interval_coded_saved_bytes': interval_saved,
        'masks': {'neural_targets': {'bytes': targets, 'final_u16_bits_saved': gain}}}
source_before = deepcopy(source_state)
measured = m.measurements_from_mixed(source_state)
assert source_state == source_before
assert measured['G_samples'] == measured['admission']['actual_pair_net_saved_bytes'] == 30-4-2
assert measured['double_weight_saved_bytes'] == -2  # Mature's 5B stays outside the sample sum.
assert measured['rates']['mature']['projected_full_net_saved_bytes'] == m.PREPARED_BYTES*5/1041690-2
assert measured['rates']['prefix8']['projected_full_net_saved_bytes'] == m.PREPARED_BYTES*30/(8 << 20)-2
decode_rate = measured['scopes']['prefix8']['candidate_decode']['cpu_seconds_per_prepared_byte']
assert decode_rate == 8/(8 << 20) and decode_rate != 8/(1072279-30)
assert measured['scopes']['q2']['statistical_start'] == 'fresh_cold'
assert measured['native_reports'] == selected['report_only']

# Depth has a genuine smaller student at epoch0, separate from original W0.
# Its fixed W/B saving can pay for negative measured payload deltas.
depth_source = deepcopy(source_state)
depth_selected = deepcopy(selected)
depth_selected.update(epoch=4, weight_bytes=1955187,
                      double_weight_saved_bytes=1898276, dual_code_saved_bytes=9024)
depth_source.update(schema='hutter.depth-mixed.v1', family=m.DEPTH_FAMILY,
    baseline_weight={'path':zero['weight_file'], 'bytes':2904325, 'sha256':m.W0_SHA256},
    candidate_weight={'path':depth_selected['weight_file'], 'bytes':1955187, 'sha256':'1'*64},
    baseline_probe={'path':'/synthetic/probe12', 'bytes':10, 'sha256':'2'*64},
    candidate_probe={'path':'/synthetic/probe8', 'bytes':9, 'sha256':'3'*64},
    B0={'path':'/synthetic/B0', 'bytes':181956, 'sha256':'4'*64},
    B1={'path':'/synthetic/B1', 'bytes':177444, 'sha256':'5'*64},
    dual_code_saved_bytes=9024, double_weight_saved_bytes=1898276,
    pipeline_terminal={'status':'cpu_budget', 'large_python_child_reaped':True,
                       'target_epochs':32, 'completed_epochs':4,
                       'completed_continuous_epochs':4, 'completed_hard_epochs':0})
depth_source['selection'].update(selected=depth_selected, completed_exports=[depth_selected],
    initial_student={'epoch':0, 'selected':False, 'original_byte_equal':False,
                     'weight_file':'/synthetic/initial8', 'weight_bytes':1955187, 'sha256':'6'*64})
depth_source['full_s'].update(known_double_weight_saved_bytes=1898276,
                            known_dual_code_saved_bytes=9024, fixed_aux={})
for name, saved, interval_saved, gain in [('prefix8',-10000,-1000,-32), ('q2',-4,-4,-8)]:
    base, candidate = depth_source['steps'][name+'-baseline'], depth_source['steps'][name+'-candidate']
    candidate['encode']['output_bytes'] = base['encode']['output_bytes']-saved
    candidate['interval']['full_coded_bytes'] = candidate['encode']['output_bytes']
    candidate['interval']['interval_coded_bytes'] = base['interval']['interval_coded_bytes']-interval_saved
    candidate['interval']['neural_targets']['final_u16_bits'] = base['interval']['neural_targets']['final_u16_bits']-gain
    depth_source['paired'][name].update(payload_saved_bytes=saved, interval_coded_saved_bytes=interval_saved)
    depth_source['paired'][name]['masks']['neural_targets']['final_u16_bits_saved'] = gain
depth_source['actual_mixed'] = {'sample_count':2, 'sample_payload_saved_bytes':-10004,
    'double_weight_saved_bytes':1898276, 'dual_code_saved_bytes':9024,
    'paired_sample_component_saved_bytes':1897296}
depth_before = deepcopy(depth_source)
depth_measured = m.measurements_from_mixed(depth_source)
assert depth_source == depth_before
assert depth_measured['G_samples'] == -10000-4+1898276+9024
assert depth_measured['binaries']['W1']['bytes'] == 177444
assert depth_measured['training_coverage']['completed_hard_epochs'] == 0
assert depth_measured['rates']['mature']['projected_full_net_saved_bytes'] == m.PREPARED_BYTES*(-1000)/1041690+1907300
assert depth_measured['scopes']['prefix8']['candidate_decode']['cpu_seconds_per_prepared_byte'] == 8/(8 << 20)
depth_gate = {'status':'admitted', 'family':m.DEPTH_FAMILY,
    'common':{name:{} for name in ['stream','vocab','dictionary','compressed_dictionary','compressed_order']},
    'sides':{side:{'weights':depth_measured['baseline'] if side=='W0' else depth_measured['selected'],
                   'binary':depth_measured['binaries'][side], 'probe':depth_measured['mixed_probes'][side],
                   'cpu_limit_seconds':540000, 'prep_cpu_seconds':0,
                   'forecast_remaining_cpu_seconds':1000} for side in ['W0','W1']},
    'admission':dict(depth_measured['admission'], decision='continue', mixed_complete=True,
                    prepared_inverse_passed=True, original_queues_terminal=True,
                    scope_extension_recorded=True, cpu_calendar_passed=True,
                    s0_reference_kind='scenario', s0_reference_bytes=990000+952225,
                    latest_record_bytes=1000000)}
for scenario in ['mature','prefix8']:
    depth_gate['admission']['record_scenario'] = scenario
    m.admitted(depth_gate)

for d in ['prepared', 'upstream/dictionary', 'assets']:
    (root/d).mkdir(parents=True)
stream = root/'prepared/stream.bin'
stream.write_bytes(bytes(range(100))*100)
(root/'prepared/vocab.bin').write_bytes(bytes(range(32)))
(root/'upstream/dictionary/english.dic').write_text('fixture dictionary')
original = root/'original.bin'
original.write_bytes(b'original fixture '*1000)
w0, w1 = root/'assets/W0', root/'assets/W1'
w0.write_bytes(b'f'*19)
w1.write_bytes(b'c')
probe = root/'assets/probe'
probe.write_text('''#!/usr/bin/env python3
import sys,json
from pathlib import Path
mode,source,target,vocab,dictionary,weights,offset,length=sys.argv[1:]
assert mode=='encode'
data=Path(source).read_bytes()[int(offset):int(offset)+int(length)]
Path(target).write_bytes(bytes(b^Path(weights).read_bytes()[0] for b in data))
with Path(target).with_name('fixture-calls').open('a') as f:f.write('encode\\n')
print('PROBE_RESULT '+json.dumps({'input_bytes':len(data),'output_bytes':Path(target).stat().st_size,'seconds':.01,'cpu_seconds':.01}))
''')
probe.chmod(0o755)
binary = root/'assets/B'
binary.write_text(f'''#!/bin/sh
printf 'decode\\n' >> '{root/'fixture-decodes'}'
cp '{original}' enwik9_uncompressed
printf 'generated' > .native-intermediate
exit 0
''')
binary.chmod(0o755)
for name in ['D','O']:
    (root/'assets'/name).write_bytes(name.encode()*13)
m.PREPARED_BYTES = stream.stat().st_size
m.ORIGINAL_BYTES = original.stat().st_size
m.ORIGINAL_MD5 = hashlib.md5(original.read_bytes()).hexdigest()
m.W0_SHA256 = m.digest(w0)
m.PPMD_MAP_BYTES = m.DISK_HEADROOM_BYTES = 0

def record(path):
    return {'path':str(path), 'bytes':path.stat().st_size, 'sha256':m.digest(path)}

entry = {'status':'admitted', 'family':'old-v22-untraced-static15', 'root':str(root),
 'admission':{'decision':'continue', 'mixed_complete':True, 'prepared_inverse_passed':True,
  'original_queues_terminal':True, 'scope_extension_recorded':True, 'cpu_calendar_passed':True,
  'mature_target_gain_bits':1, 'mature_actual_saved_bytes':100, 'mature_score_bytes':1000,
  'prefix_payload_saved_bytes':2, 'q2_payload_saved_bytes':3, 'actual_pair_net_saved_bytes':41,
  's0_reference_kind':'scenario', 's0_reference_bytes':99000, 'latest_record_bytes':100000},
 'common':{name:record(path) for name,path in [
  ('stream',stream), ('vocab',root/'prepared/vocab.bin'),
  ('dictionary',root/'upstream/dictionary/english.dic'), ('probe',probe), ('binary',binary),
  ('compressed_dictionary',root/'assets/D'), ('compressed_order',root/'assets/O')]},
 'sides':{s:{'weights':record(w), 'cpu_limit_seconds':540000,
             'prep_cpu_seconds':502.463058 if s=='W0' else 0,
             'forecast_remaining_cpu_seconds':1000} for s,w in [('W0',w0),('W1',w1)]}}
m.admitted(entry)
for modify in [lambda x:x['admission'].update(mature_target_gain_bits=0),
               lambda x:x['admission'].update(actual_pair_net_saved_bytes=42),
               lambda x:x['admission'].update(s0_reference_bytes=1000000),
               lambda x:x['sides']['W1'].update(forecast_remaining_cpu_seconds=540000)]:
    bad = deepcopy(entry)
    modify(bad)
    try:
        m.admitted(bad)
        raise RuntimeError('invalid direction/money/resource entry accepted')
    except AssertionError:
        pass
entry_path = root/'entry.json'
entry_path.write_text(json.dumps(entry)+'\n')
states = {}

def run_fixture(args):
    with (root/'fixture-state-output.log').open('a') as log, redirect_stdout(log):
        return m.run(args)

for side in ['W0','W1']:
    args = SimpleNamespace(entry=entry_path, side=side, phase='encode', out=root/side)
    states[side] = run_fixture(args)
    again = run_fixture(args)
    assert again['cpu_used_seconds'] == states[side]['cpu_used_seconds']
    assert (root/side/'runs/fullstream'/side/'fixture-calls').read_text() == 'encode\n'
    assert states[side]['status'] == 'encoded'
    assert 'formal_decode' not in states[side]['phases']
pair = m.paired_size(states['W0'],states['W1'])
assert pair['net_saved_bytes'] == 36  # One actual two-copy W fee, equal payload lengths.
assert pair['formal_decode_status'] == {'W0':'encoded','W1':'encoded'}
for side in ['W0','W1']:
    args = SimpleNamespace(entry=entry_path, side=side, phase='decode', out=root/side)
    states[side] = run_fixture(args)
    again = run_fixture(args)
    assert again['status'] == states[side]['status'] == 'passed'
    assert states[side]['result']['restored_bytes'] == m.ORIGINAL_BYTES
    assert states[side]['result']['restored_md5'] == m.ORIGINAL_MD5
    assert states[side]['phases']['formal_decode']['cpu_seconds'] > 0
    assert states[side]['phases']['formal_decode']['max_process_rss_kib'] > 0
    assert not (root/side/'package/.native-intermediate').exists()
    assert states[side]['cpu_used_seconds'] < states[side]['cpu_limit_seconds']
assert (root/'fixture-decodes').read_text() == 'decode\ndecode\n'
assert m.paired_size(states['W0'],states['W1'])['formal_decode_status'] == {'W0':'passed','W1':'passed'}

# Exercise the same real assembler/phase runner with two actual B/probe files.
depth_b0, depth_probe = root/'assets/depth-B0', root/'assets/probe8'
depth_b0.write_text(binary.read_text()+'# retained twelve-layer fixture padding\n')
depth_b0.chmod(0o755)
depth_probe.write_text(probe.read_text().replace(
    'for b in data))', "for b in data)+b'extra')"))
depth_probe.chmod(0o755)
depth_entry = deepcopy(entry)
depth_entry['family'] = m.DEPTH_FAMILY
depth_entry['common'].pop('probe')
depth_entry['common'].pop('binary')
for side, b, q in [('W0',depth_b0,probe), ('W1',binary,depth_probe)]:
    depth_entry['sides'][side].update(binary=record(b), probe=record(q))
dual_b = 2*(depth_b0.stat().st_size-binary.stat().st_size)
depth_entry['admission'].update(record_scenario='mature', dual_code_saved_bytes=dual_b,
    mature_target_gain_bits=-1, mature_actual_saved_bytes=-1,
    prefix_payload_saved_bytes=-2, q2_payload_saved_bytes=-3,
    actual_pair_net_saved_bytes=-5+36+dual_b)
m.admitted(depth_entry)
bad_margin = deepcopy(depth_entry)
# One integer byte above this scenario's exact floor must stop admission.
projected_floor = (m.PREPARED_BYTES*(-1)+(36+dual_b)*1000)//1000
bad_margin['admission']['s0_reference_bytes'] = 99000+projected_floor+1
try:
    m.admitted(bad_margin)
except AssertionError:
    pass
else:
    raise AssertionError('depth net-S record boundary accepted an extra byte')
depth_path = root/'depth-entry.json'
depth_path.write_text(json.dumps(depth_entry,indent=2)+'\n')
depth_states = {}
for side in ['W0','W1']:
    args = SimpleNamespace(entry=depth_path, side=side, phase='encode', out=root/'depth'/side)
    depth_states[side] = run_fixture(args)
    assert run_fixture(args)['status'] == 'encoded'
    assert (root/'depth'/side/'runs/fullstream'/side/'fixture-calls').read_text() == 'encode\n'
depth_pair = m.paired_size(depth_states['W0'],depth_states['W1'])
assert depth_pair['payload_saved_bytes'] == -5
assert depth_pair['double_weight_saved_bytes'] == 36 and depth_pair['dual_code_saved_bytes'] == dual_b
assert depth_pair['net_saved_bytes'] == -5+36+dual_b
for side in ['W0','W1']:
    args = SimpleNamespace(entry=depth_path, side=side, phase='decode', out=root/'depth'/side)
    depth_states[side] = run_fixture(args)
    assert run_fixture(args)['status'] == 'passed'
assert (root/'fixture-decodes').read_text() == 'decode\ndecode\ndecode\ndecode\n'
assert m.paired_size(depth_states['W0'],depth_states['W1'])['formal_decode_status'] == {'W0':'passed','W1':'passed'}
receipt = {'status':'passed', 'scope':'model-free orchestration fixture',
           'mixed_reader': {'status':'passed', 'actual_pair_net_saved_bytes':measured['G_samples'],
                            'checks':['one shared double-W fee', 'mature separate from complete prefix',
                                      'decode output-byte denominator', 'fresh q2 and native report roles']},
           'prepared_bytes_per_side':m.PREPARED_BYTES, 'restored_bytes_per_side':m.ORIGINAL_BYTES,
           'unique_encode_calls':2, 'unique_decode_calls':2, 'native_model_full_1gb_calls':0,
           'actual_two_copy_weight_saved_bytes':36,
           'carried_prep_cpu_seconds':{s:x['prior_prep_cpu_seconds'] for s,x in states.items()},
           'measured_fixture_cpu_seconds':{s:x['cpu_used_seconds']-x['prior_prep_cpu_seconds']
                                           for s,x in states.items()},
           'depth': {'status':'passed', 'mixed_G_samples':depth_measured['G_samples'],
                     'reported_completed_epochs':4, 'reported_completed_hard_epochs':0,
                     'actual_payload_saved_bytes':-5, 'actual_double_weight_saved_bytes':36,
                     'actual_dual_code_saved_bytes':dual_b,
                     'actual_pair_net_saved_bytes':depth_pair['net_saved_bytes'],
                     'unique_encode_calls':2, 'unique_decode_calls':2,
                     'checks':['original W0 distinct from epoch0 student', 'two per-side probes and binaries',
                               'signed payload delta plus one double-W and one dual-B',
                               'named net-S scenario exact integer record boundary', 'phase replay reuse']}}
(root/'check.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
