#!/usr/bin/env python3
"""Saved-logit controls: native probability half-tail/guard, feasible path oracle.

Uses the first four saved rows of each of the three measured pieces. No models run.
The accumulation reference is serial float32; deployed fast/O3 remains its own test.
"""
import argparse,json,math,struct
from pathlib import Path
from byte_path_kl import audit,conditionals,partition,project_guarded_path


def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
FLOOR=f32(1e-6)


def native_half_row(raw):
    out=[]
    tail=len(raw)//8*8
    for i,p in enumerate(raw):
        h=struct.unpack('<H',struct.pack('<e',f32(p)))[0]
        v=struct.unpack('<e',struct.pack('<H',h))[0]
        # Preserve the source's scalar last-five subnormal decode (112-e).
        if i>=tail and h&0x7c00==0 and h&0x3ff:v*=.5
        out.append(max(FLOOR,v))
    return out


def native_conditions(p,vocab,y):
    full=[0.]*256
    for b,w in zip(vocab,p):full[b]=f32(w)
    byte=vocab[y];out=[]
    for d in range(8):
        shift=8-d;lo=(byte>>shift)<<shift;hi=lo+(1<<shift);mid=(lo+hi)//2
        num=0.
        for v in full[mid:hi]:num=f32(num+v)
        den=num
        for v in full[lo:mid]:den=f32(den+v)
        out.append(f32(num/den))
    return out


def codes(q):
    return [min(100000,int(f32(f32(min(max(v,f32(1e-4)),f32(1-1e-4)))*100001))) for v in q]


def run(data):
    vocab=data['vocab_bytes'];rows=[]
    for item in data['example_rows']:
        guarded=[]
        for z in (item['teacher_logits'],item['student_logits']):
            exp=[math.exp(v-max(z)) for v in z];scale=math.fsum(exp)
            guarded.append(native_half_row([v/scale for v in exp]))
        t,s=guarded;tz,sz=math.fsum(t),math.fsum(s);y=item['target']
        geometry=audit([v/tz for v in t],[v/sz for v in s],vocab,y)
        free=[v*tz for v in geometry['projected']]
        bounded=project_guarded_path(t,s,vocab,y,FLOOR)
        requantized=native_half_row(bounded)
        q0,q1,q2,q3=[native_conditions(p,vocab,y) for p in (t,free,bounded,requantized)]
        c0,c1,c2,c3=map(codes,(q0,q1,q2,q3))
        row=dict(regime=item['regime'],piece=item['piece'],row=item['row'],target=y,
                 full_kl=geometry['full_kl'],path_kl=geometry['path_kl'],
                 unconstrained_below_guard=sum(v<FLOOR for v in free),
                 feasible_below_guard=sum(v<FLOOR for v in bounded),
                 feasible_group_max_error=max(abs(math.fsum(bounded[i] for i in g)-math.fsum(t[i] for i in g)) for g in partition(vocab,y)),
                 free_bin_changes=sum(a!=b for a,b in zip(c0,c1)),
                 feasible_bin_changes=sum(a!=b for a,b in zip(c0,c2)),
                 rehalf_bin_changes=sum(a!=b for a,b in zip(c0,c3)),
                 teacher_codes=c0,feasible_codes=c2,rehalf_codes=c3,
                 feasible_condition_max_error=max(abs(a-b) for a,b in zip(conditionals(t,vocab,y),conditionals(bounded,vocab,y))),
                 rehalf_native_condition_max_error=max(abs(a-b) for a,b in zip(q0,q3)))
        rows.append(row)
    sums={k:sum(row[k] for row in rows) for k in ('unconstrained_below_guard','feasible_below_guard','free_bin_changes','feasible_bin_changes','rehalf_bin_changes')}
    assert rows and sums['feasible_below_guard']==0
    return dict(schema='hutter.saved-logit-native-half-path-control.round463.v1',status='passed',
                scope='12 saved logit rows; FP64 softmax->FP32->source F16C plus scalar-tail half decode and guard',
                accumulation='serial float32 source-order reference; production fast/O3 independently validated',
                floor=FLOOR,model_forwards=0,codec_runs=0,rows=len(rows),path_bits=8*len(rows),
                totals=sums,maximum_feasible_group_error=max(r['feasible_group_max_error'] for r in rows),
                maximum_feasible_condition_error=max(r['feasible_condition_max_error'] for r in rows),cases=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();r=run(json.loads(a.evidence.read_text()));body=json.dumps(r,indent=2)+'\n'
    if a.output:a.output.write_text(body)
    print(json.dumps({k:v for k,v in r.items() if k!='cases'},indent=2))
