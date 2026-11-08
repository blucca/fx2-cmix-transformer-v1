#!/usr/bin/env python3
"""Exact shared-parameter tree: visitation gain and teacher/data mismatch."""
import argparse
import json
import math
from pathlib import Path

from byte_path_kl import conditionals, kl


VOCAB = [0, 64, 128, 192]
TEACHER = [.45, .05, .45, .05]


def student(p):
    # Fixed root, two child probabilities tied in opposite directions.
    return [.5*p, .5*(1-p), .5*(1-p), .5*p]


def path_loss(data, row):
    return math.fsum(weight * math.fsum(
        kl([t, 1-t], [s, 1-s])
        for t, s in zip(conditionals(TEACHER, VOCAB, y),
                        conditionals(row, VOCAB, y)))
        for y, weight in enumerate(data))


def cross_entropy(data, row):
    return -math.fsum(q*math.log(s) for q, s in zip(data, row))


def mismatch_gain(data, old, new):
    population, observed, errors = [], [], []
    for y, weight in enumerate(data):
        local = []
        for depth, (q, t, a, b) in enumerate(zip(
                conditionals(data, VOCAB, y), conditionals(TEACHER, VOCAB, y),
                conditionals(old, VOCAB, y), conditionals(new, VOCAB, y))):
            if a != b:
                # Empty/deterministic branches stay fixed and contribute zero.
                assert 0 < a < 1 and 0 < b < 1
                delta = math.log(b)-math.log1p(-b)-math.log(a)+math.log1p(-a)
                bit = (VOCAB[y] >> (7-depth)) & 1
                population.append(weight*(q-t)*delta)
                local.append((bit-t)*delta)
        correction = math.fsum(local)
        point = [float(i == y) for i in range(len(VOCAB))]
        u_gain = path_loss(point, old)-path_loss(point, new)
        errors.append(abs(math.log(new[y]/old[y])-u_gain-correction))
        observed.append(weight*correction)
    assert max(errors) < 2e-14
    assert abs(math.fsum(population)-math.fsum(observed)) < 2e-14
    return math.fsum(observed), max(errors)


def check():
    old = student(.5)
    cases = []
    for name, data in (
            ('teacher_visitation', TEACHER),
            ('matched_child_conditionals', [.81, .09, .09, .01]),
            ('opposite_child_conditionals', [.09, .81, .01, .09])):
        alpha = data[0]+data[1]
        effective_teacher = alpha*.9+(1-alpha)*.1
        effective_data = data[0]+data[3]
        new = student(effective_teacher)
        # Both closed-form objectives are Bernoulli CE plus constants.
        assert abs(alpha*(effective_teacher-.9) +
                   (1-alpha)*(effective_teacher-.1)) < 1e-14
        assert effective_teacher/effective_teacher**2 + (
            1-effective_teacher)/(1-effective_teacher)**2 > 0
        for p in (.5, effective_teacher):
            row = student(p)
            bern = kl([.9, .1], [p, 1-p])
            complement = kl([.9, .1], [1-p, p])
            assert abs(path_loss(data, row)-(alpha*bern+(1-alpha)*complement)) < 2e-14
            assert abs(path_loss(TEACHER, row)-kl(TEACHER, row)) < 2e-14
            assert abs(cross_entropy(data, row) - math.log(2) -
                       cross_entropy([effective_data, 1-effective_data], [p, 1-p])) < 2e-14
        u_gain = path_loss(data, old)-path_loss(data, new)
        ce_gain = cross_entropy(data, old)-cross_entropy(data, new)
        correction, sample_error = mismatch_gain(data, old, new)
        assert abs(ce_gain-u_gain-correction) < 2e-14
        cases.append(dict(name=name, data=data, full_kl_optimal_p=.5,
                          path_optimal_p=effective_teacher,
                          data_ce_optimal_p=effective_data,
                          path_gain_nats=u_gain, data_ce_gain_nats=ce_gain,
                          mismatch_gain_correction_nats=correction,
                          maximum_sample_identity_error_nats=sample_error,
                          full_kl_change_nats=kl(TEACHER, new)-kl(TEACHER, old)))
    assert abs(cases[0]['path_optimal_p']-.5) < 1e-14
    assert abs(cases[1]['path_gain_nats']-.22175369374985104) < 2e-14
    assert abs(cases[1]['data_ce_gain_nats']-cases[1]['path_gain_nats']) < 2e-14
    assert cases[2]['path_gain_nats'] > 0 and cases[2]['data_ce_gain_nats'] < 0
    return dict(schema='hutter.byte-path-shared-capacity.v1', status='passed',
                scope='One constrained probability tree; closed-form optima; exact four-label enumeration',
                vocab_bytes=VOCAB, teacher=TEACHER,
                model_forwards=0, optimizer_updates=0, native_runs=0,
                cases=cases)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    body = json.dumps(check(), indent=2)+'\n'
    if a.output:
        a.output.write_text(body)
    print(body, end='')
