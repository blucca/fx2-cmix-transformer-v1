#!/usr/bin/env python3
"""Output-coordinate path-gradient scale over 12 already saved rows.

This uses effective positive row logits, after guard+renormalization. It does
not differentiate the original model/guard or execute a native observer.
"""
import argparse, json, math, sys
from pathlib import Path
sys.dont_write_bytecode = True
from byte_path_kl import audit, conditionals


def dot(a,b):
    return math.fsum(x*y for x,y in zip(a,b))


def bern(t,s):
    if t == s: return 0.0
    return (t*math.log(t/s) if t else 0.0) + ((1-t)*math.log((1-t)/(1-s)) if t<1 else 0.0)


def path(t,s,vocab,y):
    target=vocab[y]; g=[0.0]*len(vocab); loss=0.0
    for depth in range(8):
        prefix=[i for i,v in enumerate(vocab) if v>>(8-depth)==target>>(8-depth)]
        one=[i for i in prefix if (vocab[i]>>(7-depth))&1]
        zero=[i for i in prefix if not (vocab[i]>>(7-depth))&1]
        # Empty child is a deterministic vocab branch for both rows.
        if not one or not zero: continue
        ts=math.fsum(t[i] for i in prefix); ss=math.fsum(s[i] for i in prefix)
        t1=math.fsum(t[i] for i in one); s1=math.fsum(s[i] for i in one)
        s0=math.fsum(s[i] for i in zero)
        rt,rs=t1/ts,s1/ss; loss+=bern(rt,rs); err=rs-rt
        for i in one: g[i]+=err*s[i]/s1
        for i in zero: g[i]-=err*s[i]/s0
    assert abs(math.fsum(g))<3e-14
    return loss,g


def rows(p):
    out=[]
    for z in (p['teacher_logits'],p['student_logits']):
        e=[math.exp(v-max(z)) for v in z]; total=math.fsum(e)
        guarded=[max(v/total,1e-6) for v in e]; scale=math.fsum(guarded)
        out.append([v/scale for v in guarded])
    return out


def run(data):
    cases=[]; maxerr=0.0
    for item in data['example_rows']:
        t,s=rows(item); v=data['vocab_bytes']; fullg=[b-a for a,b in zip(t,s)]
        fullnorm2=dot(fullg,fullg); fgfull=audit(t,s,v,item['target'])['full_kl']
        label_results=[path(t,s,v,y) for y in range(len(v))]
        expected_g=[math.fsum(t[y]*label_results[y][1][i] for y in range(len(v))) for i in range(len(v))]
        expected_loss=math.fsum(t[y]*label_results[y][0] for y in range(len(v)))
        error=max(abs(a-b) for a,b in zip(fullg,expected_g)); maxerr=max(maxerr,error)
        assert error<2e-12 and abs(expected_loss-fgfull)<2e-12
        eu_norm2=math.fsum(t[y]*dot(g,g) for y,(_,g) in enumerate(label_results))
        variance=math.fsum(t[y]*dot([x-f for x,f in zip(g,fullg)],[x-f for x,f in zip(g,fullg)]) for y,(_,g) in enumerate(label_results))
        assert abs(eu_norm2-fullnorm2-variance)<2e-12
        actual_loss,actualg=label_results[item['target']]
        anorm2=dot(actualg,actualg); cosine=dot(actualg,fullg)/math.sqrt(anorm2*fullnorm2)
        ceg=list(s); ceg[item['target']]-=1.0
        cases.append(dict(regime=item['regime'],piece=item['piece'],row=item['row'],target=item['target'],
            full_kl=fgfull,actual_U=actual_loss,full_gradient_norm2=fullnorm2,
            actual_U_gradient_norm2=anorm2,actual_U_full_gradient_dot=dot(actualg,fullg),
            actual_U_full_gradient_cosine=cosine,actual_U_full_norm_ratio=math.sqrt(anorm2/fullnorm2),
            teacher_expected_U_gradient_norm2=eu_norm2,teacher_gradient_variance_norm2=variance,
            teacher_rms_U_full_norm_ratio=math.sqrt(eu_norm2/fullnorm2),
            actual_U_CE_gradient_cosine=dot(actualg,ceg)/math.sqrt(anorm2*dot(ceg,ceg)),
            full_KL_CE_gradient_cosine=dot(fullg,ceg)/math.sqrt(fullnorm2*dot(ceg,ceg)),
            expectation_gradient_error=error))
    f=math.fsum(c['full_gradient_norm2'] for c in cases)
    a=math.fsum(c['actual_U_gradient_norm2'] for c in cases)
    e=math.fsum(c['teacher_expected_U_gradient_norm2'] for c in cases)
    cross=math.fsum(c['actual_U_full_gradient_dot'] for c in cases)
    # One 4-leaf finite-difference control, covering actual analytic U gradient.
    t=[.125,.375,.25,.25]; s=[.15,.1,.3,.45]; v=[0,64,128,192];y=2
    _,g=path(t,s,v,y);z=[math.log(q) for q in s];eps=1e-5
    for i in range(4):
        vals=[]
        for h in (eps,-eps):
            zz=list(z);zz[i]+=h;weights=[math.exp(q) for q in zz];scale=math.fsum(weights)
            vals.append(path(t,[q/scale for q in weights],v,y)[0])
        assert abs((vals[0]-vals[1])/(2*eps)-g[i])<2e-10
    return dict(schema='hutter.path-gradient.saved-rows.round466.v1',status='passed',
        scope='12 saved initial8 rows, 205 labels per row; effective guarded-row logits',
        rows=len(cases),label_paths=len(cases)*len(data['vocab_bytes']),
        model_forwards=0,native_runs=0,parameter_vjps=0,optimizer_updates=0,
        actual_target_U_mean=math.fsum(c['actual_U'] for c in cases)/len(cases),
        full_KL_mean=math.fsum(c['full_kl'] for c in cases)/len(cases),
        pooled_actual_U_full_gradient_norm_ratio=math.sqrt(a/f),
        pooled_actual_U_full_gradient_cosine=cross/math.sqrt(a*f),
        pooled_teacher_rms_U_full_gradient_norm_ratio=math.sqrt(e/f),
        pooled_teacher_sampling_variance_fraction=(e-f)/e,
        maximum_expectation_gradient_error=maxerr,cases=cases)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();data=json.loads(a.evidence.read_text());result=run(data)
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
