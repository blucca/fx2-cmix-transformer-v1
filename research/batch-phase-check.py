from pathlib import Path
import json
import subprocess
import sys
import argparse
parser=argparse.ArgumentParser(description="Check encode checkpoint reuse and later-phase decode with a tiny fake codec")
parser.add_argument('--work',type=Path,required=True)
root=parser.parse_args().work.resolve()
for d in ['prepared','upstream/dictionary','upstream/models','build/baseline']:(root/d).mkdir(parents=True,exist_ok=True)
(root/'prepared/stream.bin').write_bytes(bytes(range(256))*4)
(root/'prepared/vocab.bin').write_bytes(bytes(range(256)))
(root/'upstream/dictionary/english.dic').write_text('fixture')
(root/'upstream/models/6m-q4-fp32.tfwc2').write_text('fixture')
probe=root/'build/baseline/probe'
probe.write_text('''#!/usr/bin/env python3
from pathlib import Path
import sys,json
mode,source,target,*rest=sys.argv[1:]
data=Path(source).read_bytes()
if mode=='encode':
 offset,length=map(int,rest[-2:]);data=data[offset:offset+length]
Path(target).write_bytes(data)
print('PROBE_RESULT '+json.dumps({'seconds':.01,'coding_seconds':.01,'input_bytes':len(data),'output_bytes':len(data),'max_rss_kib':100}))
''');probe.chmod(0o700)
plan=root/'plan.json';plan.write_text(json.dumps([{'name':'sample','offset':128,'length':384}]))
cmd=[sys.executable,str(Path(__file__).with_name('batch.py').resolve()),'--root',str(root),'--plan',str(plan),'--out',str(root/'run'),'--variants','baseline']
for phase in ['encode','encode','decode']:
 result=subprocess.run([*cmd,'--phase',phase],capture_output=True,text=True,check=True)
 receipt=json.loads((root/'run/sample/baseline/result.json').read_text())
 assert receipt['status']==('passed' if phase=='decode' else 'encoded')
 if phase=='encode':assert not (root/'run/sample/baseline/decoded.bin').exists()
 print(phase,result.stdout.splitlines()[-1])
assert receipt['original_slice_sha256']==receipt['restored_slice_sha256']
assert receipt['encoded_bytes']==384 and receipt['active_seconds']==.02
print('stage-checkpoint-reuse-roundtrip-ok')
