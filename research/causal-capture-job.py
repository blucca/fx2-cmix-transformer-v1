"""One recovered causal pass with train/held and an independent native audit.
2号独立CPU/RAM；计划与f16文件均沿原piece采样。原4号队列保持。
"""
from pathlib import Path
import json,os,resource,subprocess,time
j=Path(__file__).resolve().parent;r=j.parent
os.sched_setaffinity(0,{0})
resource.setrlimit(resource.RLIMIT_AS,(18<<30,18<<30))
resource.setrlimit(resource.RLIMIT_CPU,(16200,16200))
plan=json.loads((j/'data/plan.json').read_text());out=j/'capture';out.mkdir(exist_ok=True)
state={'status':'starting','started_at':time.time(),'cpu_limit_seconds':16200,
       'prep_cpu_limit_seconds':32400,
       'prior_failed_capture_cpu_seconds':14400,
       'recovery_prepare_cpu_seconds':json.loads((j/'recovery-ready.json').read_text())['cpu_seconds'],
       'address_space_limit_gib':18,'source_bytes':plan['source_bytes'],
       'roles':{role:{k:v for k,v in info.items() if k!='pieces_local'} for role,info in plan['roles'].items()}}
def save():
 p=j/'capture-state.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
save()
try:
 assert all(not (out/f'{role}.{ext}').exists() for role in [*plan['roles'],'audit'] for ext in ['u8','f16'])
 assert (r/'prepared/stream.bin').stat().st_size==plan['source_bytes']
 # The real map plus final captured rows needs 15.3GiB; unused old swap has been
 # released on this dedicated VM. Original trainer/4号swap are untouched.
 available=os.statvfs(j).f_bavail*os.statvfs(j).f_frsize
 required=14000*(1<<20)+sum(x['bytes']*411 for x in plan['roles'].values())+131072*411
 assert available>required+(1<<30),(available,required)
 command=[str(j/'bin/causal-ppmd'),str(r/'prepared/stream.bin'),str(r/'prepared/vocab.bin'),str(j/'data/joint-plan.tsv'),str(out)]
 with (j/'capture.log').open('w') as log:
  p=subprocess.Popen(command,cwd=out,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  state.update(status='running',stage='full_causal_capture',capture_pid=p.pid,command=command);save()
  try:code=p.wait(timeout=43200) # Includes natural platform pause windows.
  except subprocess.TimeoutExpired:p.kill();p.wait();raise
 assert code==0,code
 rows=[json.loads(line.split(' ',1)[1]) for line in (j/'capture.log').read_text().splitlines() if line.startswith('CAUSAL_RESULT ')]
 assert rows and rows[-1]['status']=='passed' and rows[-1]['processed_bytes']==plan['source_bytes']
 for role,info in plan['roles'].items():
  assert (out/f'{role}.u8').stat().st_size==info['bytes']
  assert (out/f'{role}.f16').stat().st_size==410*info['bytes']
 audit=json.loads((j/'long-audit/plan.json').read_text())
 assert rows[-1]['audit_bytes']==audit['bytes']==131072
 for ext,multiple in [('u8',1),('f16',410)]:
  assert (out/f'audit.{ext}').stat().st_size==audit['bytes']*multiple
 usage=resource.getrusage(resource.RUSAGE_CHILDREN)
 state.update(status='passed',stage='ready_for_native_and_teacher',result=rows[-1],
              full_capture_cpu_seconds=usage.ru_utime+usage.ru_stime,
              capture_wrapper_cpu_seconds=time.process_time(),
              independent_long_audit=audit,completed_at=time.time());save()
 print(json.dumps(state),flush=True)
except Exception as e:
 state.update(status='failed',error=repr(e),completed_at=time.time());save()
 # A killed source PPMd leaves only a rebuildable sparse map.
 (out/'ppm.temp').unlink(missing_ok=True)
 raise
