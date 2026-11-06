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
print('PROBE_RESULT '+json.dumps({'input_bytes':len(data),'output_bytes':len(data),'seconds':.01,'cpu_seconds':.01}))
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
receipt = {'status':'passed', 'scope':'model-free orchestration fixture',
           'mixed_reader': {'status':'passed', 'actual_pair_net_saved_bytes':measured['G_samples'],
                            'checks':['one shared double-W fee', 'mature separate from complete prefix',
                                      'decode output-byte denominator', 'fresh q2 and native report roles']},
           'prepared_bytes_per_side':m.PREPARED_BYTES, 'restored_bytes_per_side':m.ORIGINAL_BYTES,
           'unique_encode_calls':2, 'unique_decode_calls':2, 'native_model_full_1gb_calls':0,
           'actual_two_copy_weight_saved_bytes':36,
           'carried_prep_cpu_seconds':{s:x['prior_prep_cpu_seconds'] for s,x in states.items()},
           'measured_fixture_cpu_seconds':{s:x['cpu_used_seconds']-x['prior_prep_cpu_seconds']
                                           for s,x in states.items()}}
(root/'check.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
