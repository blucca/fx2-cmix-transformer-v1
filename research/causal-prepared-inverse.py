#!/usr/bin/env python3
"""One original prepared->1GB inverse after the existing causal heavy queue.
No model/range work. The compiler and inverse share a 0.5 CPUh ceiling;
original pipeline/mixed run first, on the same owned VM and preserved process.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, os, resource, shutil, subprocess, time

j=Path(__file__).resolve().parent
r=j.parent
original=r/'round414'
job=j/'prepared-inverse'
job.mkdir(exist_ok=True)
state_path=j/'prepared-inverse-state.json'
assert not state_path.exists(), 'inspect existing inverse job before starting'
state={'status':'running','stage':'waiting_original_queues','started_at':time.time(),
       'cpu_limit_seconds':1800,'address_space_limit_gib':4,'ppmd_range_neural_calls':0}
TERMINAL={'complete','passed','cpu_budget','stopped','failed'}
CUTOFF=datetime.fromisoformat('2026-11-13T03:10:00+08:00')
os.sched_setaffinity(0,{0})

def cpu():
 child=resource.getrusage(resource.RUSAGE_CHILDREN)
 return time.process_time()+child.ru_utime+child.ru_stime

def save(stage,**updates):
 state.update(stage=stage,updated_at=time.time(),total_cpu_seconds=cpu(),**updates)
 p=state_path.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2)+'\n');p.replace(state_path)

def run(stage,command,as_gib,compile_step=False):
 remaining=math.floor(1800-cpu()-15) # Reserve final MD5/log/cleanup CPU.
 assert remaining>30,'inverse CPU budget exhausted'
 if compile_step:remaining=min(180,remaining)
 def limits():
  resource.setrlimit(resource.RLIMIT_AS,(as_gib<<30,as_gib<<30))
  resource.setrlimit(resource.RLIMIT_CPU,(remaining-1,remaining))
 save(stage,command=list(map(str,command)))
 with (job/(stage+'.log')).open('w') as log:
  subprocess.run(list(map(str,command)),cwd=job,stdin=subprocess.DEVNULL,stdout=log,
                 stderr=subprocess.STDOUT,preexec_fn=limits,check=True,timeout=604800,
                 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))

try:
 save('waiting_original_queues')
 while True:
  parts=[]
  for path in [original/'pipeline-state.json',original/'mixed-audit/state.json']:
   parts.append(json.loads(path.read_text()) if path.exists() else {})
  if any(x.get('status')=='failed' for x in parts):
   save('original_queue_failed',status='stopped',reason='original queue failed; fullS stops')
   (j/'prepared-inverse.exit').write_text('0\n');raise SystemExit(0)
  if all(x.get('status') in TERMINAL for x in parts):break
  if datetime.now(timezone.utc)>=CUTOFF:
   save('original_window_stop',status='stopped',reason='reserve the original token deadline')
   (j/'prepared-inverse.exit').write_text('0\n');raise SystemExit(0)
  time.sleep(60)
 assert shutil.disk_usage(job).free>=8<<30,'inverse needs 8 GiB free after the original heavy queue'
 source=original/'source'
 cpp=source/'research/prepare.cpp'
 assert cpp.is_file() and '--restore' in cpp.read_text()
 save('inputs_ready',input_stream_bytes=(r/'prepared/stream.bin').stat().st_size,
      prepared_source=str(cpp),prepared_source_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),
      free_disk_bytes=shutil.disk_usage(job).free)
 binary=job/'prepare'
 command=['clang++-17','-O3','-std=c++17','-march=x86-64-v3','-mrecip=none',
          '-DSEED=923','-DUPDATE_LIMIT=3000','-ffunction-sections','-fdata-sections',
          cpp,source/'src/preprocess/preprocessor.cpp',source/'src/preprocess/dictionary.cpp',
          '-Wl,--gc-sections','-o',binary]
 run('compile',command,2,compile_step=True)
 compile_cpu=cpu()
 restored=job/'restored'
 assert not restored.exists()
 run('inverse',['/usr/bin/time','-v','-o',job/'inverse.time',binary,'--restore',
                r/'prepared/stream.bin',r/'upstream/dictionary/english.dic',restored],4)
 save('verify_original_enwik9')
 final=restored/'enwik9_uncompressed'
 h=hashlib.md5()
 with final.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 assert final.stat().st_size==1000000000
 assert h.hexdigest()=='e206c3450ac99950df65bf70ef61a12d',h.hexdigest()
 files=[p for p in restored.iterdir() if p.is_file()]
 disk={'logical_bytes':sum(p.stat().st_size for p in files),
       'allocated_bytes':sum(p.stat().st_blocks*512 for p in files),'files':len(files)}
 # This fresh directory contains only this inverse's generated artifacts.
 # Keep the restored original for final compressor timing; discard duplicate intermediates.
 for p in files:
  if p!=final:p.unlink()
 result={'restored_bytes':final.stat().st_size,'restored_md5':h.hexdigest(),
         'compile_cpu_seconds':compile_cpu,'cpu_seconds':cpu(),
         'max_child_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
         'retained_original':str(final),'completed_disk_peak':disk,'range_neural_calls':0}
 save('provenance_complete',status='passed',result=result,completed_at=time.time())
 (j/'prepared-inverse.exit').write_text('0\n')
 print(json.dumps(result),flush=True)
except Exception as e:
 save('failed',status='failed',error=repr(e),completed_at=time.time())
 (j/'prepared-inverse.exit').write_text('1\n')
 raise
