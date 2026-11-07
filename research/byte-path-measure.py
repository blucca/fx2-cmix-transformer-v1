#!/usr/bin/python3
"""One initial-eight/teacher geometry assay on three existing causal held pieces."""
import argparse, json, math, os, resource, signal, sys, time, traceback
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
for name in ('source','data','checkpoint','kernel','selection','output'):
    parser.add_argument('--'+name,type=Path,required=True)
a=parser.parse_args()
E=a.output.resolve();E.mkdir(parents=True,exist_ok=False)
J=a.data.resolve().parent
S=a.source.resolve()
sys.dont_write_bytecode = True
sys.path[:0] = [str(Path(__file__).resolve().parent), str(S/'research'), str(S), str(J/'deps')]
START = time.monotonic()
result = dict(schema='hutter.initial8.byte-path-kl.round463.v1', status='running',
              scope='three complete previously scored causal held pieces; initial eight-block student',
              teacher_forwards=0, student_forwards=0, parameter_vjps=0,
              optimizer_updates=0, native_calls=0, codec_calls=0, teacher_cache_writes=0,
              selection='first short, first 1025..2048-target, first <=2048-target teacher-NLL>=1 piece',
              probability_arithmetic='FP32 model logits; FP64 softmax; 1e-6 guard and renormalization',
              projection='probability-space reverse-KL I-projection; current-byte path only',
              gradient_coordinates='logits of normalized effective row; zero model VJPs',
              cpu_limit_seconds=120, memory_budget_bytes=1<<30, pieces=[], example_rows=[])


def save():
    result.update(cpu_seconds=time.process_time(), wall_seconds=time.monotonic()-START,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    tmp=E/'result.tmp'; tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    tmp.replace(E/'result.json')


def limit(sig,frame):
    raise TimeoutError('cpu_limit' if sig==signal.SIGXCPU else 'wall_limit')


try:
    core=min(os.sched_getaffinity(0)); os.sched_setaffinity(0,{core})
    resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
    resource.setrlimit(resource.RLIMIT_CPU,(118,120))
    signal.signal(signal.SIGXCPU,limit); signal.signal(signal.SIGALRM,limit); signal.alarm(170)
    save()
    import numpy as np
    import torch
    from cpu_model import load_model
    from depth_student import load_student
    from byte_path_kl import audit, conditionals
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    chosen=json.loads(a.selection.read_text())
    vocab=chosen['vocab_bytes']; assert len(vocab)==205
    result['vocab_bytes']=vocab
    result['vocab_mask_hex']=chosen['vocab_mask_hex']
    manifest=json.loads(a.data.read_text())
    sample=manifest['samples'][0]; assert sample['name']=='causal'
    score=sample['score']; bounds=np.fromfile(score['boundaries'],dtype='<i4').tolist()
    assert len(bounds)==474 and score['targets']==1082505
    result['paths']=dict(source=str(S),checkpoint=str(a.checkpoint),
                         kernel=str(a.kernel),data=stra.data)
    teacher,_=load_model(a.checkpoint,a.kernel,native_scaled_gemm=True)
    student,_,_=load_student(a.checkpoint,a.kernel,native_scaled_gemm=True)
    teacher.requires_grad_(False).eval(); student.requires_grad_(False).eval()
    assert teacher.cfg.n_layers==12 and student.cfg.n_layers==8
    result.update(torch_version=torch.__version__,cpu_affinity=[core],native_scaled_gemm=True,
                  retained_layers=[0,1,2,3,8,9,10,11],student_initialization='original mapped checkpoint; zero updates')
    rawbytes=np.array(vocab,dtype=np.int32)

    def native_q(p,y):
        # Match ByteModel::Predict: right float sum, then left seeded by that sum.
        full=np.zeros(256,dtype=np.float32);full[rawbytes]=p
        byte=vocab[y]; out=[]
        for depth in range(8):
            shift=8-depth; lo=(byte>>shift)<<shift; hi=lo+(1<<shift); mid=(lo+hi)//2
            num=np.cumsum(full[mid:hi],dtype=np.float32)[-1]
            denom=np.cumsum(np.concatenate(([num],full[lo:mid])),dtype=np.float32)[-1]
            out.append(float(np.float32(num/denom)))
        return np.array(out)

    total_fields=('full_kl','path_kl','within_sibling_kl','teacher_ce','student_ce')
    totals={k:[] for k in total_fields}; count=0
    for selected in chosen['pieces']:
        index=selected['piece']; lo,hi=bounds[index:index+2]; n=hi-lo-1
        assert [lo,hi]==selected['local_token_range'] and n==selected['targets']
        tok=np.fromfile(score['tokens'],dtype=np.uint8,count=n+1,offset=lo)
        prior=np.fromfile(score['priors'],dtype='<f2',count=n*205,offset=lo*410).reshape(n,205)
        x=torch.from_numpy(tok[:-1].astype(np.int64))[None]
        pp=torch.from_numpy(prior)[None]; nb=torch.tensor([[0,n]],dtype=torch.int32)
        zz=[]
        with torch.no_grad():
            for model,name in ((teacher,'teacher'),(student,'student')):
                h=model.compute_last_activation(input_tokens=x,priors=pp,cumulative_sequence_lengths=nb)
                z=model.last_activation_to_logits(h,priors=pp)
                assert z.dtype==torch.float32 and tuple(z.shape)==(1,n,205)
                zz.append(z[0].double().numpy().copy()); result[name+'_forwards']+=1
                del h,z
        rows=[]; masks=[]; unreguarded_bins=reguarded_bins=0
        maxcond=maxreguard=maxgroup=0.; belowfloor=0
        for k,(tz,sz,y) in enumerate(zip(*zz,tok[1:])):
            probs=[]; sums=[]
            for z in (tz,sz):
                p=np.exp(z-z.max());p/=p.sum(dtype=np.float64)
                g=np.maximum(p,1e-6); scale=g.sum(dtype=np.float64)
                probs.append(g/scale);sums.append(scale)
            t,s=probs; y=int(y)
            row=audit(t.tolist(),s.tolist(),vocab,y)
            r=np.array(row['projected']); rows.append(row)
            tcond=np.array(conditionals(t.tolist(),vocab,y)); rcond=np.array(conditionals(r.tolist(),vocab,y))
            maxcond=max(maxcond,float(np.abs(tcond-rcond).max()))
            maxgroup=max(maxgroup,max(abs(sum(r[g])-tm) for g,tm in zip(row['groups'],row['teacher_group_mass'])))
            # External byte rows have arbitrary normalization; the original native guard is absolute.
            rabs=r*sums[0];tabs=t*sums[0]
            belowfloor+=int(np.count_nonzero(rabs<1e-6))
            reguarded=np.maximum(rabs,1e-6)
            q0,q1,q2=native_q(tabs,y),native_q(rabs,y),native_q(reguarded,y)
            maxreguard=max(maxreguard,float(np.abs(q0-q2).max()))
            def bins(q):return np.minimum(100000,(np.clip(q,1e-4,1-1e-4)*100001).astype(np.int64))
            unreguarded_bins+=int(np.count_nonzero(bins(q0)!=bins(q1)))
            reguarded_bins+=int(np.count_nonzero(bins(q0)!=bins(q2)))
            fullgrad=s-t; cg=np.array(row['gradient'])
            masks.append((np.dot(fullgrad,fullgrad),np.dot(cg,cg),np.dot(fullgrad,cg)))
            for field in ('full_kl','path_kl','within_sibling_kl'):totals[field].append(row[field])
            totals['teacher_ce'].append(-math.log(t[y]));totals['student_ce'].append(-math.log(s[y]))
            if k<4:
                result['example_rows'].append(dict(regime=selected['regime'],piece=index,row=k,target=y,
                    teacher_logits=tz.tolist(),student_logits=sz.tolist(),full_kl=row['full_kl'],path_kl=row['path_kl']))
        sums_grads=np.sum(masks,axis=0)
        mean={f:math.fsum(r[f] for r in rows)/n for f in ('full_kl','path_kl','within_sibling_kl')}
        mean.update(teacher_ce=math.fsum(totals['teacher_ce'][count:])/n,
                    student_ce=math.fsum(totals['student_ce'][count:])/n)
        result['pieces'].append(dict(regime=selected['regime'],piece=index,local_token_range=[lo,hi],targets=n,
            means=mean,path_fraction_of_full_kl=mean['path_kl']/mean['full_kl'],
            max_path_probability_error=maxcond,max_group_mass_error=maxgroup,
            projected_below_native_guard=belowfloor,native_bin_changes_from_float_sum=unreguarded_bins,
            native_bin_changes_after_reapplying_guard=reguarded_bins,total_path_bits=8*n,
            reguard_max_native_conditional_error=maxreguard,
            output_gradient_norm_ratio=math.sqrt(sums_grads[1]/sums_grads[0]),
            output_gradient_cosine=float(sums_grads[2]/math.sqrt(sums_grads[0]*sums_grads[1])),
            fisher_rank_counts={str(g-1):sum(len(r['groups'])==g for r in rows) for g in sorted({len(r['groups']) for r in rows})}))
        count+=n; result['targets']=count; save()
        del tok,prior,x,pp,nb,zz,rows,masks
    result['means']={f:math.fsum(values)/count for f,values in totals.items()}
    result['path_fraction_of_full_kl']=result['means']['path_kl']/result['means']['full_kl']
    result.update(status='passed',fisher_full_rank=204,fisher_path_rank_upper=8)
    save(); print(json.dumps({k:result[k] for k in ('status','targets','means','path_fraction_of_full_kl','cpu_seconds','peak_rss_kib')}))
except BaseException as exc:
    result.update(status='bounded_stop' if isinstance(exc,TimeoutError) else 'failed',
                  error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());save();raise
finally:
    signal.alarm(0)
