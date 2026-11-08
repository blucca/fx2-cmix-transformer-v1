#!/usr/bin/env python3
"""A log-odds shadow at frozen native observer values; no model/native execution."""
import json, math
from pathlib import Path

N=100001
ROOT=Path(__file__).resolve().parent/'native-byte-path'


def centre(row):
    return float(row['override']) if row['override'] in (0,1) else (row['lookup_index']+.5)/N


def kl(t,s):
    if t==s:return 0.0
    assert 0<t<1 and 0<s<1
    return t*math.log(t/s)+(1-t)*math.log1p((s-t)/(1-s))


def check_logit(t,s):
    eta=math.log(s/(1-s));eps=1e-4
    sigmoid=lambda a:1/(1+math.exp(-a))
    fd=(kl(t,sigmoid(eta+eps))-kl(t,sigmoid(eta-eps)))/(2*eps)
    residual=s-t
    assert abs(fd-residual)<3e-11
    return dict(teacher_r=t,student_r=s,observed_kl=kl(t,s),
                native_log_odds_vjp=residual,finite_difference_in_relaxed_log_odds=fd)


pairs={}
for line in (ROOT/'observations-192.jsonl').read_text().splitlines():
    row=json.loads(line);key=(row['regime'],row['piece'],row['row'],row['depth_1_based'])
    pairs.setdefault(key,{})[row['side']]=row
zero=0;changed=[]
for key,pair in pairs.items():
    t,s=centre(pair['teacher']),centre(pair['oracle'])
    if t==s:
        assert kl(t,s)==0 and s-t==0;zero+=1
    else:
        changed.append(dict(key=list(key),**check_logit(t,s)))
control=dict(source='controlled clipped scalar; no native capture',**check_logit(20.5/N,10.5/N))
assert len(pairs)==96 and zero==95 and len(changed)==1
assert changed[0]['native_log_odds_vjp']>0 and control['native_log_odds_vjp']<0
result=dict(schema='hutter.native-log-odds-shadow.scalar.v1',status='passed',
    scope='96 frozen native observer pairs plus one labelled scalar control',
    native_pairs=len(pairs),zero_loss_and_carrier_pairs=zero,changed_pairs=changed,
    clip_control=control,model_forwards=0,native_runs=0,capture_runs=0,learner_implementation=0)
print(json.dumps(result,indent=2))
