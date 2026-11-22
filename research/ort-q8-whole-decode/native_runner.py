#!/usr/bin/env python3
"""Complete fixed DistilGPT2 decode contract and same-host whole-call timing.
One session per fresh worker. This is not a general model converter/benchmark.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import onnxruntime as ort

PAST=(16,128,512)
OUTPUTS=['logits']+[f'present.{i}.{kind}' for i in range(6) for kind in ('key','value')]
INPUTS=['input_ids']+[f'past_key_values.{i}.{kind}' for i in range(6) for kind in ('key','value')]+['attention_mask']
TOL=2e-5

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def host():
    text=Path('/proc/cpuinfo').read_text()
    flags=next(x.split(':',1)[1].split() for x in text.splitlines() if x.startswith('flags'))
    return {'cpu':next(x.split(':',1)[1].strip() for x in text.splitlines() if x.startswith('model name')),
            'features':{k:k in flags for k in ('avx2','fma','avx512f','avx512_vnni','avx_vnni')},
            'python':sys.version,'numpy':np.__version__,'ort':ort.__version__}

def session(root,mode):
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level=3
    if mode=='native':opts.register_custom_ops_library(str(root/'libq8_m1.so'))
    s=ort.InferenceSession(str(root/'models'/f'{mode}.onnx'),opts,providers=['CPUExecutionProvider'])
    assert [x.name for x in s.get_inputs()]==INPUTS
    assert [x.name for x in s.get_outputs()]==OUTPUTS
    return s

def output_map(s,feed):return dict(zip(OUTPUTS,s.run(OUTPUTS,feed)))

def stats(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype==np.float32
    delta=float(np.max(np.abs(np.subtract(a,b,dtype=np.float64))))
    scale=max(1.0,float(np.max(np.abs(b))))
    return {'max_abs_error':delta,'norm_max_error':delta/scale,'passed':delta/scale<=TOL}

def validate(actual,ref,feed):
    assert list(actual)==OUTPUTS and list(ref)==OUTPUTS
    p=feed['past_key_values.0.key'].shape[2]
    rows={};passed=True
    for name in OUTPUTS:
        a,b=actual[name],ref[name]
        shape=(1,1,50257) if name=='logits' else (1,12,p+1,64)
        assert a.shape==b.shape==shape,(name,a.shape,b.shape,shape)
        assert a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all(),name
        row={'shape':list(a.shape),'whole':stats(a,b)}
        if name!='logits':
            old=feed[name.replace('present.','past_key_values.',1)]
            row['prefix_bits_preserved']=bool(np.array_equal(a[:,:,:p,:].view(np.uint32),old.view(np.uint32)))
            row['new_slice']=stats(a[:,:,p:,:],b[:,:,p:,:])
            passed &= row['prefix_bits_preserved'] and row['new_slice']['passed']
        passed &= row['whole']['passed']
        rows[name]=row
    top1=int(np.argmax(actual['logits'][0,0]))
    ref_top1=int(np.argmax(ref['logits'][0,0]))
    passed &= top1==ref_top1
    return {'all_outputs_passed':bool(passed),'top1':top1,'reference_top1':ref_top1,'outputs':rows}

def self_test():
    # An unchanged large cached prefix can mask a wrong newly computed value
    # under whole-tensor normalization. The appended slice has its own oracle.
    a=np.zeros((1,1,101,1),dtype=np.float32);b=a.copy()
    a[:,:,:100,:]=b[:,:,:100,:]=1000.0;a[:,:,100:,:]=.01
    assert stats(a,b)['passed'] and not stats(a[:,:,100:,:],b[:,:,100:,:])['passed']
    assert stats(b,b)['passed']

def prepare(root):
    out=root/'feeds';out.mkdir(exist_ok=True)
    s=session(root,'dense')
    past={f'past_key_values.{i}.{kind}':np.empty((1,12,0,64),dtype=np.float32)
          for i in range(6) for kind in ('key','value')}
    rng=np.random.default_rng(725)
    tokens=rng.integers(0,50257,size=max(PAST)+1,dtype=np.int64)
    # These fixed, explicitly synthetic token IDs produce actual dense-Q8
    # transformer caches, not invented zero caches or user adoption evidence.
    for p,token in enumerate(tokens):
        feed={'input_ids':np.array([[token]],dtype=np.int64),**past,
              'attention_mask':np.ones((1,p+1),dtype=np.int64)}
        if p in PAST:
            np.savez(out/f'past-{p}.npz',**feed)
        ys=output_map(s,feed)
        if p in PAST:
            check=validate(ys,ys,feed)
            assert check['all_outputs_passed']
            dump(out/f'past-{p}.json',{'past':p,'token_id':int(token),'complete_source_cache':True,'check':check})
            print('FEED_COMPLETE',p,'top1',check['top1'],flush=True)
        past={name.replace('present.','past_key_values.',1):ys[name] for name in OUTPUTS[1:]}
    dump(out/'manifest.json',{'host':host(),'past_lengths':list(PAST),'seed':725,
         'tokens':tokens.tolist(),'material':'synthetic fixed-token sequence; actual chained dense-Q8 model cache outputs',
         'bytes':sum(x.stat().st_size for x in out.glob('*.npz'))})
    del s;gc.collect()

def run(root,label,mode,run_name,reference):
    folder=root/'runs'/label/run_name;folder.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter();s=session(root,mode);initialization=time.perf_counter()-began
    report={'status':'running','mode':mode,'host':host(),'threads':1,'optimization':'ORT_ENABLE_ALL',
            'initialization_seconds':initialization,'warmup':30,'timed':200,'rows':[],
            'reference_run':reference,'comparison_scope':'same quantized B/S; all logits/KV and appended KV slice'}
    dump(folder/'result.json',report)
    for p in PAST:
        feed=load_npz(root/'feeds'/f'past-{p}.npz')
        actual=output_map(s,feed)
        ref=actual if reference=='self' else load_npz(root/'runs'/label/reference/f'past-{p}-outputs.npz')
        check=validate(actual,ref,feed)
        np.savez(folder/f'past-{p}-outputs.npz',**actual)
        del actual,ref;gc.collect()
        for _ in range(30):s.run(OUTPUTS,feed)
        samples=[]
        for _ in range(200):
            start=time.perf_counter();s.run(OUTPUTS,feed)
            samples.append((time.perf_counter()-start)*1000)
        # This includes allocation/return of logits plus all twelve full KV
        # tensors. No output selection, model-phase subtraction or disk I/O.
        row={'past':p,'check':check,'p50_ms':float(np.median(samples)),
             'p95_ms':float(np.percentile(samples,95)),'samples_ms':samples}
        report['rows'].append(row);dump(folder/'result.json',report)
        print('CASE_COMPLETE',label,run_name,p,'p50',row['p50_ms'],'precision',check['all_outputs_passed'],flush=True)
    report.update(status='completed',all_outputs_passed=all(x['check']['all_outputs_passed'] for x in report['rows']),
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  completed_at=datetime.now(timezone.utc).isoformat())
    dump(folder/'result.json',report)
    del s;gc.collect()

def compare(root,label):
    reports={name:json.loads((root/'runs'/label/name/'result.json').read_text())
             for name in ('dense-before','native','q8_acc0','split8','dense-after')}
    assert all(r['status']=='completed' and len(r['rows'])==3 for r in reports.values())
    rows=[]
    for i,p in enumerate(PAST):
        d0=reports['dense-before']['rows'][i]['p50_ms'];d1=reports['dense-after']['rows'][i]['p50_ms']
        native=reports['native']['rows'][i]
        baseline=min(d0,d1)
        rows.append({'past':p,'dense_before_p50_ms':d0,'dense_after_p50_ms':d1,
                     'conservative_dense_p50_ms':baseline,'native_p50_ms':native['p50_ms'],
                     'native_time_reduction':1-native['p50_ms']/baseline,
                     'native_precision_pass':native['check']['all_outputs_passed'],
                     'stock_q8_p50_ms':reports['q8_acc0']['rows'][i]['p50_ms'],
                     'split8_p50_ms':reports['split8']['rows'][i]['p50_ms']})
    target=reports['native']['host']['features']['avx2'] and not any(reports['native']['host']['features'][k]
             for k in ('avx512f','avx512_vnni','avx_vnni'))
    precision_by_mode={name:bool(r['all_outputs_passed']) for name,r in reports.items()}
    numerical=all(precision_by_mode.values())
    performance=all(x['native_time_reduction']>=.15 for x in rows)
    result={'label':label,'host':reports['native']['host'],'rows':rows,
            'numerical_pass':numerical,'precision_by_mode':precision_by_mode,
            'performance_pass':performance,'target_avx2_no_vnni':target,
            'target_gate_passed':bool(target and numerical and performance),
            'resource_peak_rss_kib':max(r['peak_rss_kib'] for r in reports.values()),
            'full_reports':{n:str((root/'runs'/label/n/'result.json').relative_to(root)) for n in reports}}
    dump(root/'runs'/label/'comparison.json',result)
    print('COMPARISON',json.dumps({k:result[k] for k in ('label','numerical_pass','performance_pass','target_avx2_no_vnni','target_gate_passed')}),flush=True)
    return result

def all_runs(args):
    root=args.root;folder=root/'runs'/args.label;folder.mkdir(parents=True,exist_ok=True)
    stages=[('prepare','dense','prepare','self')]+[('run',mode,name,'self' if name=='dense-before' else 'dense-before')
            for mode,name in [('dense','dense-before'),('native','native'),('q8_acc0','q8_acc0'),('split8','split8'),('dense','dense-after')]]
    for action,mode,name,ref in stages:
        dump(folder/'state.json',{'stage':name,'status':'running'})
        cmd=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--label',args.label,
             '--action',action,'--mode',mode,'--run-name',name,'--reference',ref,'--max-as-mib',str(args.max_as_mib)]
        with (folder/f'{name}.log').open('w') as log:
            completed=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            dump(folder/'state.json',{'stage':name,'status':'failed','returncode':completed.returncode,'log':f'{name}.log'})
            raise SystemExit(completed.returncode)
        print('STAGE_DONE',name,flush=True)
    compare(root,args.label)
    dump(folder/'state.json',{'stage':'complete','status':'completed'})
    (folder/'DONE').write_text('Complete fixed-model five-run comparison; gate is in comparison.json\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--action',choices=('self-test','prepare','run','compare','all'),default='all')
    ap.add_argument('--label',default='remote');ap.add_argument('--mode',default='dense')
    ap.add_argument('--run-name',default='dense-before');ap.add_argument('--reference',default='self')
    ap.add_argument('--max-as-mib',type=int,default=2048)
    args=ap.parse_args();args.root=args.root.resolve()
    self_test()
    if args.action=='self-test':print('SELF_TEST_PASS');raise SystemExit(0)
    assert ort.__version__=='1.31.1'
    if args.action in ('prepare','run'):
        resource.setrlimit(resource.RLIMIT_AS,(args.max_as_mib*2**20,args.max_as_mib*2**20))
        if hasattr(os,'sched_getaffinity'):
            os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    if args.action=='prepare':prepare(args.root)
    elif args.action=='run':run(args.root,args.label,args.mode,args.run_name,args.reference)
    elif args.action=='compare':compare(args.root,args.label)
    else:all_runs(args)
