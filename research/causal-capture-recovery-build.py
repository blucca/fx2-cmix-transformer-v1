"""Compile only the changed collector and check its added audit route.
Original PPMd/model objects and the existing full short capture are reused.
The old failed queue is retained until this build and exact-row check pass.
"""
from pathlib import Path
from array import array
import json,os,resource,subprocess,time

j=Path(__file__).resolve().parent;r=j.parent
os.sched_setaffinity(0,{0})
resource.setrlimit(resource.RLIMIT_AS,(18<<30,18<<30))
resource.setrlimit(resource.RLIMIT_CPU,(300,300))
b=j/'recovery-bin';b.mkdir(exist_ok=True)
state={'status':'running','started_at':time.time(),'stage':'compile_changed_collector',
       'training_updates':0,'cpu_limit_seconds':300,'address_space_limit_gib':18}
def cpu():
 u=resource.getrusage(resource.RUSAGE_CHILDREN)
 prior=json.loads((j/'recovery-build-failures.json').read_text())['cpu_seconds'] if (j/'recovery-build-failures.json').exists() else 0
 return prior+time.process_time()+u.ru_utime+u.ru_stime
def save():
 p=j/'recovery-build-state.json';q=p.with_suffix('.tmp')
 q.write_text(json.dumps(state,indent=2)+'\n');q.replace(p)
def run(label,command,cwd=j):
 state.update(stage=label);save()
 with (j/(label+'.log')).open('w') as log:
  subprocess.run(list(map(str,command)),cwd=cwd,stdout=log,stderr=subprocess.STDOUT,
                 stdin=subprocess.DEVNULL,check=True,timeout=900)

save()
try:
 prior_meta=json.loads((j/'recovery-build-failures.json').read_text()) if (j/'recovery-build-failures.json').exists() else {}
 reuse_fixture=prior_meta.get('reuse_completed_row_fixture',False)
 original_bin=j/'bin';source=j/'source'
 if not reuse_fixture:
  old=r/'build/v26';original_bin=j/'bin';source=j/'source'
  manifest=json.loads((old/'build.json').read_text())
  link=next(c for c in manifest['commands'] if c[-2:]==['-o','cmix'])
  tf=next(c for c in manifest['commands'] if c[-2:]==['-o','tf_qmat_dense.o'])
  flags=tf[1:tf.index('-c')]
  pred=next(c for c in manifest['commands'] if any(str(a).endswith('/predictor.cpp') for a in c))
  first=next(i for i,a in enumerate(pred) if str(a).endswith('.cpp'))
  # VM2 originally received the two ready binaries, not VM4's private object dir.
  # Recompile the same two adapted objects with their original recorded flags.
  for name,rel in [('mixer','src/mixer/mixer.cpp'),('predictor','src/predictor.cpp')]:
   run('recovery-build-'+name,[pred[0],*pred[1:first],'-I',source/'src','-c',source/rel,'-o',b/(name+'.o')])
  run('recovery-compile',[tf[0],*flags,'-fno-fast-math','-ffp-contract=off','-DSEED=923',
        '-DUPDATE_LIMIT=3000','-I',source/'src','-c',source/'research/causal-ppmd.cpp','-o',b/'causal-ppmd.o'])
  args=[]
  for a in link[:-2]:
   if a in ('runner.o','-s'):continue
   if a.endswith('.o'):
    replacement=b/a if (b/a).exists() else original_bin/a
    if a=='tf_weights_io_compressed.o':replacement=j/'assets'/a
    a=replacement if replacement.exists() else old/a
   args.append(a)
  run('recovery-link',[*args,b/'causal-ppmd.o','-o',b/'causal-ppmd'])
 
 fixture=j/'recovery-check';fixture.mkdir(exist_ok=True)
 with (r/'prepared/stream.bin').open('rb') as f:(fixture/'stream-64k.bin').write_bytes(f.read(65536))
 triples=[('train',256,4096),('audit',8192,16384),('validation',32768,65536)]
 (fixture/'plan.tsv').write_text(''.join(f'{lo} {hi} {role}\n' for role,lo,hi in triples))
 if not reuse_fixture:
  run('recovery-audit-row-check',[b/'causal-ppmd',fixture/'stream-64k.bin',r/'prepared/vocab.bin',fixture/'plan.tsv',fixture],fixture)
 reference=r/'round412/mature-ppmd/check'
 reference_reused=(reference/'full.u8').exists() and (reference/'full.f16').exists()
 if not reference_reused:
  # VM2 received the original binary, while VM4 retained the old fixture files.
  # Make only the missing reference, preserving the completed new-role capture.
  reference=fixture/'original-reference';reference.mkdir(exist_ok=True)
  run('recovery-original-reference',[original_bin/'activation-probe','capture',fixture/'stream-64k.bin',
       reference/'full.u8',reference/'full.f16',r/'prepared/vocab.bin',0,65536,0],reference)
 for role,lo,hi in triples:
  for ext,multiple in [('u8',1),('f16',410)]:
   with (reference/f'full.{ext}').open('rb') as f:
    f.seek(lo*multiple);expected=f.read((hi-lo)*multiple)
   assert (fixture/f'{role}.{ext}').read_bytes()==expected,(role,ext)
 rows=[json.loads(x.split(' ',1)[1]) for x in (j/'recovery-audit-row-check.log').read_text().splitlines() if x.startswith('CAUSAL_RESULT ')]
 assert rows[-1]['status']=='passed' and rows[-1]['processed_bytes']==65536
 assert all(rows[-1][role+'_bytes']==hi-lo for role,lo,hi in triples)

 # Preserve the seeded estimator. Only the actual fixed audit interval is added.
 plan=json.loads((j/'data/plan.json').read_text())
 text=(j/'data/plan.tsv').read_text()
 intervals=[(int(a),int(z),role) for a,z,role in (line.split() for line in text.splitlines())]
 assert len(intervals)==1786 and plan['roles']['train']['targets']==2941307
 assert plan['roles']['validation']['targets']==1082505 and plan['source_bytes']==586459020
 bounds=array('i');bounds.frombytes((r/'prepared/article_boundaries.i32').read_bytes())
 assert bounds[-1]==plan['source_bytes']
 article=max(range(len(bounds)-1),key=lambda a:bounds[a+1]-bounds[a])
 assert not any(row['article']==article for row in plan['intervals'])
 cap=1<<17;complete=(bounds[article+1]-bounds[article])//cap
 lo=bounds[article]+(complete//2)*cap;hi=lo+cap
 assert (article,lo,hi)==(243425,570595966,570727038)
 assert all(hi<=a or lo>=z for a,z,_ in intervals)
 joint=sorted([*intervals,(lo,hi,'audit')])
 assert len(joint)==1787 and all(a[1]<=z[0] for a,z in zip(joint,joint[1:]))
 assert [x for x in joint if x[2]!='audit']==intervals
 (j/'data/joint-plan.tsv').write_text(''.join(f'{a} {z} {role}\n' for a,z,role in joint))
 audit=j/'long-audit';audit.mkdir(exist_ok=True)
 info={'article':article,'article_begin':bounds[article],'article_end':bounds[article+1],
       'source_begin':lo,'source_end':hi,'bytes':cap,'targets':cap-1,
       'training_article_overlap':0,'sampling':'separate-fixed-max-native-length-audit'}
 (audit/'plan.json').write_text(json.dumps(info,indent=2)+'\n')
 (audit/'bounds.i32').write_bytes(array('i',[0,cap]).tobytes())
 assert 14400+16200+1800==32400
 used=cpu();assert used+14.349395+1200<1800,used
 result={'status':'passed','stage':'ready_to_replace_failed_queue','cpu_seconds':used,
         'original_prepare_cpu_seconds':14.349395,'failed_capture_cpu_seconds':14400,
         'new_capture_limit_seconds':16200,'cumulative_prep_limit_seconds':32400,
         'child_peak_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
         'all_205_columns_exact':True,'original_full_short_capture_reused':reference_reused,
         'original_capture_binary_reused':True,'completed_new_role_fixture_reused':reuse_fixture,
         'fixture_source_bytes':65536,'fixture_roles':{role:hi-lo for role,lo,hi in triples},
         'source_bytes':plan['source_bytes'],'original_intervals':1786,'joint_intervals':1787,
         'original_seeded_plan_unchanged':True,'independent_long_audit':info,'training_updates':0}
 state.update(**result,completed_at=time.time());save()
 (j/'recovery-ready.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
except Exception as e:
 state.update(status='failed',error=repr(e),cpu_seconds=cpu(),completed_at=time.time());save();raise
finally:
 # This fixture's sparse map is regenerateable; the original plan/assets stay.
 for rel in ['recovery-check/ppm.temp','recovery-check/original-reference/ppm.temp']:
  (j/rel).unlink(missing_ok=True)
