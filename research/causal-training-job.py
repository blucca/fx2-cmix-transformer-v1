"""Serial causal capture, held max-length audit, and one complete QAT trajectory.
Copy beside capture.py into the dedicated experiment directory. Original VM4
queue is separate. This job uses existing prepared data, evaluator and CPU kernel.
"""
from pathlib import Path
from array import array
import json,os,resource,subprocess,time

j=Path(__file__).resolve().parent;r=j.parent
os.sched_setaffinity(0,{0})
state={'status':'running','started_at':time.time(),'stage':'waiting_for_inputs',
       'prep_cpu_limit_seconds':28800,'training_cpu_limit_seconds':172800,
       'recipe':'causal-v1','training_epochs':16}
state_path=j/'pipeline-state.json'
assert not state_path.exists(), 'inspect the existing pipeline before starting'
def save(stage,**values):
 state.update(stage=stage,updated_at=time.time(),**values)
 p=state_path.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2)+'\n');p.replace(state_path)
def command(label,argv,cwd,as_gib,cpu_seconds):
 def limits():
  resource.setrlimit(resource.RLIMIT_AS,(as_gib<<30,as_gib<<30))
  resource.setrlimit(resource.RLIMIT_CPU,(cpu_seconds,cpu_seconds))
 with (j/(label+'.log')).open('w') as log:
  p=subprocess.Popen(list(map(str,argv)),cwd=cwd,stdin=subprocess.DEVNULL,
                     stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits)
  save(label,child_pid=p.pid,command=list(map(str,argv)))
  code=p.wait()
 assert code==0,(label,code)
 lines=(j/(label+'.log')).read_text().splitlines()
 return [json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('ACTIVATION_RESULT ')]
try:
 save('waiting_for_inputs')
 gates=['capture.exit','apt-torch.exit','typing-install.exit','asset-transfer.exit']
 while not all((j/name).exists() for name in gates):
  for name in gates:
   if (j/name).exists():assert int((j/name).read_text())==0,(name,(j/name).read_text())
  if time.time()-state['started_at']>172800:raise TimeoutError('input stages exceeded two days including pauses')
  time.sleep(60)
 for name in gates:assert int((j/name).read_text())==0,name
 captured=json.loads((j/'capture-state.json').read_text());assert captured['status']=='passed'
 plan=json.loads((j/'data/plan.json').read_text())
 full_cpu=captured.get('full_capture_cpu_seconds',captured['result']['cpu_seconds'])
 capture_wrapper_cpu=captured.get('capture_wrapper_cpu_seconds',0)
 prep_limit=captured.get('prep_cpu_limit_seconds',28800)
 prior_cpu=captured.get('prior_failed_capture_cpu_seconds',0)
 recovery_prepare_cpu=captured.get('recovery_prepare_cpu_seconds',0)
 save('full_capture_ready',full_capture=captured['result'],prep_cpu_limit_seconds=prep_limit,
      prior_failed_capture_cpu_seconds=prior_cpu,recovery_prepare_cpu_seconds=recovery_prepare_cpu)

 # Current seeded random held sample has no full 2^17 piece. Keep its estimator
 # intact and reserve a separate native audit from the actual largest article.
 # This article has zero sampled training pieces, so its entire NN cluster is held.
 bounds=array('i');bounds.frombytes((r/'prepared/article_boundaries.i32').read_bytes())
 assert bounds[-1]==plan['source_bytes']
 article=max(range(len(bounds)-1),key=lambda a:bounds[a+1]-bounds[a])
 assert not any(row['article']==article and row['role']=='train' for row in plan['intervals'])
 maximum=1<<17;complete=(bounds[article+1]-bounds[article])//maximum
 assert complete>=1
 lo=bounds[article]+(complete//2)*maximum;hi=lo+maximum
 assert hi<=bounds[article+1]
 audit=j/'long-audit';audit.mkdir(exist_ok=True)
 assert not any((audit/name).exists() for name in ['tokens.u8','priors.f16'])
 (audit/'bounds.i32').write_bytes(array('i',[0,maximum]).tobytes())
 audit_info={'article':article,'article_begin':bounds[article],'article_end':bounds[article+1],
             'source_begin':lo,'source_end':hi,'bytes':maximum,'targets':maximum-1,
             'training_article_overlap':0,'sampling':'separate-fixed-max-native-length-audit'}
 (audit/'plan.json').write_text(json.dumps(audit_info,indent=2)+'\n')
 if captured.get('independent_long_audit'):
  shared=captured['independent_long_audit']
  assert all(shared[k]==audit_info[k] for k in audit_info), (shared,audit_info)
  for ext,name,multiple in [('u8','tokens.u8',1),('f16','priors.f16',410)]:
   p=j/'capture'/f'audit.{ext}';assert p.stat().st_size==maximum*multiple
   p.rename(audit/name)
  audit_cpu=0 # The one full pass already pays for these independently held rows.
  audit_info.update(capture={'mode':'shared_full_causal_capture','warm_start':0,
                            'source_begin':lo,'source_end':hi,'bytes':maximum,
                            'full_pass_processed_bytes':captured['result']['processed_bytes'],
                            'cpu_accounting':'charged_once_in_full_capture'})
 else:
  remaining=min(14400,int(prep_limit-prior_cpu-recovery_prepare_cpu-full_cpu-600-14.349395))
  assert remaining>=600, 'prep CPU exhausted before the reserved native-length audit'
  rows=command('long-causal-capture',[j/'bin/activation-probe','capture',r/'prepared/stream.bin',
               audit/'tokens.u8',audit/'priors.f16',r/'prepared/vocab.bin',lo,maximum,0],audit,18,remaining)
  assert len(rows)==1 and rows[0]['processed_bytes']==hi and rows[0]['warm_start']==0
  assert (audit/'tokens.u8').stat().st_size==maximum
  assert (audit/'priors.f16').stat().st_size==maximum*410
  audit_cpu=rows[0]['cpu_seconds'];audit_info.update(capture=rows[0])
 (audit/'plan.json').write_text(json.dumps(audit_info,indent=2)+'\n')
 save('long_audit_ready',long_audit=audit_info)

 train,held=plan['roles']['train'],plan['roles']['validation']
 score={'tokens':str(j/'capture/validation.u8'),'priors':str(j/'capture/validation.f16'),
        'boundaries':str(j/'data/validation.i32'),'targets':held['targets'],
        'long_piece_indices':[i for i,(a,b) in enumerate(held['pieces_local']) if b-a>32768]}
 samples=[{'name':'causal','tokens':str(j/'capture/train.u8'),'priors':str(j/'capture/train.f16'),
           'train_pieces':train['pieces_local'],'score':score},
          {'name':'long-audit','tokens':str(audit/'tokens.u8'),'priors':str(audit/'priors.f16'),
           'train_pieces':[], 'score':{'tokens':str(audit/'tokens.u8'),'priors':str(audit/'priors.f16'),
                                     'boundaries':str(audit/'bounds.i32'),'targets':maximum-1}}]
 data={'method':plan['method'],'source_bytes':plan['source_bytes'],'training_targets':train['targets'],
       'training_pieces':train['pieces'],'samples':samples,'independent_long_audit':audit_info}
 (j/'data-manifest.json').write_text(json.dumps(data,indent=2)+'\n')
 baseline={'status':'running','steps':{}}
 for sample in samples:
  name=sample['name'];s=sample['score']
  result=command('baseline-'+name,[j/'bin/activation-probe','eval-pieces',s['tokens'],s['priors'],
                 j/'assets/original-static15.tfwc3',s['boundaries']],j,2,600)
  assert len(result)==1 and result[0]['total']['count']==s['targets']
  baseline['steps'][name]={'status':'passed','result':result[0]}
 baseline['child_cpu_seconds']=sum(v['result']['cpu_seconds'] for v in baseline['steps'].values())
 used=resource.getrusage(resource.RUSAGE_CHILDREN)
 native_child_cpu=used.ru_utime+used.ru_stime
 prep_cpu=prior_cpu+recovery_prepare_cpu+full_cpu+capture_wrapper_cpu+native_child_cpu+time.process_time()+14.349395
 baseline['native_child_cpu_seconds']=native_child_cpu
 assert prep_cpu<prep_limit,prep_cpu
 baseline['status']='passed';(j/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
 save('native_baselines_ready',prep_cpu_seconds=prep_cpu,
      baseline={n:v['result']['total'] for n,v in baseline['steps'].items()})

 env=dict(os.environ,PYTHONPATH=str(j/'source')+':'+str(j/'deps'),
          OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
 argv=['python3','-B',str(j/'source/research/integer-coordinate-qat.py'),
       '--checkpoint',str(j/'assets/original.tch'),'--kernel',str(j/'assets/cpu-kda.so'),
       '--data',str(j/'data-manifest.json'),'--baseline',str(j/'baseline.json'),
       '--weights',str(r/'candidate/models/6m-q4-fp32.tfwc2'),'--recode',str(j/'assets/recode'),
       '--evaluator',str(j/'bin/activation-probe'),'--output',str(j/'training'),
       '--recipe','causal-v1','--cpu-seconds','172800']
 with (j/'preflight.log').open('w') as log:
  p=subprocess.Popen([*argv,'--check-only'],cwd=j,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  save('causal_preflight',child_pid=p.pid,command=argv+['--check-only']);code=p.wait()
 assert code==0,('preflight',code)
 preflight=json.loads((j/'training/preflight.json').read_text());assert preflight['status']=='passed'
 (j/'preflight.exit').write_text('0\n')
 with (j/'training.log').open('w') as log:
  p=subprocess.Popen([*argv,'--resume'],cwd=j,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  save('training',child_pid=p.pid,command=argv+['--resume'],preflight='passed');code=p.wait()
 (j/'training.exit').write_text(str(code)+'\n')
 terminal=json.loads((j/'training/state.json').read_text())
 assert code in (0,124) and terminal['status'] in ('complete','cpu_budget','stopped'),(code,terminal['status'])
 save('training_terminal',status='passed',training_status=terminal['status'],training_epoch=terminal['epoch'],
      training_step=terminal['step'],training_total_cpu_seconds=terminal['total_cpu_seconds'],completed_at=time.time())
except Exception as e:
 save('failed',status='failed',error=repr(e),completed_at=time.time());raise
