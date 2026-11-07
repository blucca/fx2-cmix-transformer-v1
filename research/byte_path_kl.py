#!/usr/bin/env python3
"""Target-path teacher KL for an MSB-first byte coder; probability-space audit.

This is an output geometry tool. It leaves all training and coder recipes fixed.
"""
import math


def partition(vocab, target):
    """Target leaf + one sibling subtree at each of the eight bit decisions."""
    assert 0 <= target < len(vocab) and len(set(vocab)) == len(vocab)
    assert all(isinstance(v, int) and 0 <= v <= 255 for v in vocab)
    groups = [[] for _ in range(9)]
    for i, byte in enumerate(vocab):
        groups[8 - (byte ^ vocab[target]).bit_length()].append(i)
    return [g for g in groups if g]


def kl(p, q):
    return math.fsum(a * math.log(a / b) for a, b in zip(p, q) if a)


def audit(teacher, student, vocab, target):
    assert len(teacher) == len(student) == len(vocab)
    assert all(math.isfinite(p) and p > 0 for row in (teacher, student) for p in row)
    assert all(abs(math.fsum(row) - 1) < 1e-10 for row in (teacher, student))
    groups = partition(vocab, target)
    t = [math.fsum(teacher[i] for i in g) for g in groups]
    s = [math.fsum(student[i] for i in g) for g in groups]
    projected = [0.] * len(vocab)
    # Gradient in effective-probability softmax coordinates, before a model/guard VJP.
    gradient = list(student)
    for indices, tmass, smass in zip(groups, t, s):
        for i in indices:
            projected[i] = student[i] * tmass / smass
            gradient[i] -= projected[i]
    full, coarse, remaining = kl(teacher, student), kl(t, s), kl(teacher, projected)
    assert abs(full - coarse - remaining) < 2e-12
    assert abs(projected[target] - teacher[target]) < 2e-14
    return dict(full_kl=full, path_kl=coarse, within_sibling_kl=remaining,
                groups=groups, teacher_group_mass=t, student_group_mass=s,
                projected=projected, gradient=gradient)


def project_guarded_path(teacher, student, vocab, target, floor=1e-6):
    """Keep absolute teacher group masses, with the observed codec floor.

    Each sibling minimizes relative entropy with r>=floor and fixed teacher mass.
    The output is a probability-space oracle, before physical half representability.
    """
    assert all(p >= floor for row in (teacher, student) for p in row)
    projected = [0.] * len(vocab)
    for group in partition(vocab, target):
        mass = math.fsum(teacher[i] for i in group)
        free = list(group)
        while free:
            fixed = len(group) - len(free)
            scale = (mass - fixed * floor) / math.fsum(student[i] for i in free)
            bound = [i for i in free if scale * student[i] < floor]
            if not bound:
                for i in free:
                    projected[i] = scale * student[i]
                break
            for i in bound:
                projected[i] = floor
            free = [i for i in free if i not in bound]
        assert abs(math.fsum(projected[i] for i in group)-mass) < 2e-12
    assert abs(projected[target]-teacher[target]) < 2e-14
    return projected


def conditionals(p, vocab, target):
    byte = vocab[target]
    out = []
    for depth in range(8):
        indices = [i for i, v in enumerate(vocab) if v >> (8-depth) == byte >> (8-depth)]
        ones = [i for i in indices if (vocab[i] >> (7-depth)) & 1]
        out.append(math.fsum(p[i] for i in ones) / math.fsum(p[i] for i in indices))
    return out


def check():
    # Four real byte IDs, including empty branches, check the group-KL chain rule.
    vocab, teacher, student, y = [1, 2, 4, 7], [.05, .35, .2, .4], [.2, .1, .4, .3], 2
    row = audit(teacher, student, vocab, y)
    assert row['groups'] == [[0, 1], [3], [2]]
    assert max(abs(a-b) for a,b in zip(conditionals(teacher,vocab,y),
                                      conditionals(row['projected'],vocab,y))) < 1e-14
    assert row['within_sibling_kl'] > .05 and abs(math.fsum(row['gradient'])) < 1e-14
    floored = project_guarded_path([.1,.4,.2,.3],[.01,.8,.1,.1],vocab,y,floor=.01)
    assert min(floored)>=.01 and max(abs(a-b) for a,b in zip(conditionals(floored,vocab,y),conditionals([.1,.4,.2,.3],vocab,y)))<1e-14
    return {'status':'passed', 'partition':row['groups'],
            'full_kl':row['full_kl'], 'path_kl':row['path_kl'],
            'remaining_full_kl_at_exact_path_match':row['within_sibling_kl']}


if __name__ == '__main__':
    import argparse, json
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path)
    args = p.parse_args()
    checked = check()
    if args.evidence:
        data = json.loads(args.evidence.read_text())
        errors=[]
        for row in data['example_rows']:
            logits = row['teacher_logits'], row['student_logits']
            prob=[]
            for z in logits:
                weights=[max(math.exp(v-max(z)),1e-300) for v in z]
                total=math.fsum(weights)
                raw=[w/total for w in weights]
                guarded=[max(v,1e-6) for v in raw]
                scale=math.fsum(guarded)
                prob.append([v/scale for v in guarded])
            actual=audit(*prob,data['vocab_bytes'],row['target'])
            errors.append(abs(actual['full_kl']-row['full_kl']))
            errors.append(abs(actual['path_kl']-row['path_kl']))
        assert errors and max(errors)<2e-10
        checked.update(replayed_rows=len(data['example_rows']), maximum_error=max(errors))
    print(json.dumps(checked,indent=2))
