#!/usr/bin/env python3
"""Saved-row check: count each actually visited bit once in teacher distillation."""
import argparse,json,math
from pathlib import Path
from byte_path_kl import audit,conditionals


def bernoulli_kl(t,s):
    return (t*math.log(t/s) if t else 0.) + ((1-t)*math.log((1-t)/(1-s)) if t<1 else 0.)


def run(data):
    vocab=data['vocab_bytes'];rows=[]
    for item in data['example_rows']:
        prob=[]
        for z in (item['teacher_logits'],item['student_logits']):
            e=[math.exp(v-max(z)) for v in z];scale=math.fsum(e)
            g=[max(v/scale,1e-6) for v in e];total=math.fsum(g);prob.append([v/total for v in g])
        t,s=prob;full=audit(t,s,vocab,item['target'])['full_kl']
        unweighted=[];coarse=[]
        for y in range(len(vocab)):
            tc,sc=conditionals(t,vocab,y),conditionals(s,vocab,y)
            terms=[bernoulli_kl(a,b) for a,b in zip(tc,sc)]
            masses=[math.fsum(t[i] for i,v in enumerate(vocab) if v>>(8-d)==vocab[y]>>(8-d)) for d in range(8)]
            unweighted.append(math.fsum(terms));coarse.append(math.fsum(a*b for a,b in zip(masses,terms)))
        expected=math.fsum(a*b for a,b in zip(t,unweighted))
        expected_coarse=math.fsum(a*b for a,b in zip(t,coarse))
        assert abs(expected-full)<2e-12 and expected_coarse<=full+2e-12
        y=item['target'];observed=audit(t,s,vocab,y)['path_kl'];assert abs(coarse[y]-observed)<2e-12
        rows.append(dict(regime=item['regime'],piece=item['piece'],row=item['row'],target=y,
                         full_kl=full,teacher_expected_unweighted_path_kl=expected,
                         teacher_expected_nine_group_kl=expected_coarse,
                         actual_target_unweighted_path_kl=unweighted[y],actual_target_nine_group_kl=coarse[y]))
    return dict(schema='hutter.byte-path-weighting.saved-rows.round463.v1',status='passed',rows=len(rows),
                scope='12 saved FP64-softmax plus guard rows; teacher-drawn target expectation enumerates 205 labels',
                model_forwards=0,codec_runs=0,
                maximum_expected_identity_error=max(abs(r['full_kl']-r['teacher_expected_unweighted_path_kl']) for r in rows),
                actual_target_means={k:math.fsum(r[k] for r in rows)/len(rows) for k in
                      ('full_kl','actual_target_unweighted_path_kl','actual_target_nine_group_kl')},cases=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();r=run(json.loads(a.evidence.read_text()));body=json.dumps(r,indent=2)+'\n'
    if a.output:a.output.write_text(body)
    print(json.dumps({k:v for k,v in r.items() if k!='cases'},indent=2))
