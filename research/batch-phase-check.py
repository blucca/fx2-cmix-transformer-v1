from pathlib import Path
import json
import subprocess
import sys
import argparse
parser=argparse.ArgumentParser(description="Check per-variant weights, encode reuse and later-phase decode with a tiny fake codec")
parser.add_argument('--work',type=Path,required=True)
root=parser.parse_args().work.resolve()
for d in ['prepared','upstream/dictionary','upstream/models','build/baseline','build/candidate']:(root/d).mkdir(parents=True,exist_ok=True)
(root/'prepared/stream.bin').write_bytes(bytes(range(256))*4)
(root/'prepared/vocab.bin').write_bytes(bytes(range(256)))
(root/'upstream/dictionary/english.dic').write_text('fixture')
(root/'upstream/models/6m-q4-fp32.tfwc2').write_text('fixture')
(root/'candidate-weights').write_text('candidate')
probe=root/'build/baseline/probe'
probe.write_text('''#!/usr/bin/env python3
from pathlib import Path
import sys,json
mode,source,target,*rest=sys.argv[1:]
data=Path(source).read_bytes()
key=Path(rest[2]).read_bytes()[0]
if mode=='encode':
 offset,length=map(int,rest[-2:]);data=data[offset:offset+length]
data=bytes(value^key for value in data)
Path(target).write_bytes(data)
print('PROBE_RESULT '+json.dumps({'seconds':.01,'coding_seconds':.01,'input_bytes':len(data),'output_bytes':len(data),'max_rss_kib':100,'fixture_weight':key}))
''');probe.chmod(0o700)
(root/'build/candidate/probe').write_bytes(probe.read_bytes())
(root/'build/candidate/probe').chmod(0o700)
plan=root/'plan.json';plan.write_text(json.dumps([{'name':'sample','offset':128,'length':384}]))
cmd=[sys.executable,str(Path(__file__).with_name('batch.py').resolve()),'--root',str(root),'--plan',str(plan),'--out',str(root/'run'),'--variants','baseline','candidate','--weights','candidate='+str(root/'candidate-weights')]
for phase in ['encode','encode','decode']:
 result=subprocess.run([*cmd,'--phase',phase],capture_output=True,text=True,check=True)
 receipts={name:json.loads((root/f'run/sample/{name}/result.json').read_text()) for name in ['baseline','candidate']}
 for name,receipt in receipts.items():
  assert receipt['status']==('passed' if phase=='decode' else 'encoded')
  assert receipt['encode']['fixture_weight']==ord('f' if name=='baseline' else 'c')
  if phase=='encode':assert not (root/f'run/sample/{name}/decoded.bin').exists()
 print(phase,result.stdout.splitlines()[-1])
for name,receipt in receipts.items():
 assert receipt['original_slice_sha256']==receipt['restored_slice_sha256']
 assert receipt['encoded_bytes']==384 and receipt['active_seconds']==.02
 assert receipt['decode']['fixture_weight']==receipt['encode']['fixture_weight']
assert receipts['baseline']['identity']['binary_sha256']==receipts['candidate']['identity']['binary_sha256']
assert receipts['baseline']['identity']['weights_sha256']!=receipts['candidate']['identity']['weights_sha256']
assert (root/'run/sample/baseline/encoded.bin').read_bytes()!=(root/'run/sample/candidate/encoded.bin').read_bytes()
print('per-weight-stage-checkpoint-reuse-roundtrip-ok')
